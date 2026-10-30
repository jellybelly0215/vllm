# SPDX-License-Identifier: Apache-2.0
"""
PolicyOffloadingConnector — applies a recompute/load policy to ALL requests.

Extends ProbeOffloadingConnector (probe requests with an explicit
kv_transfer_params.policy keep their forced behaviour). For every other request
whose lookup queues an SSD->CPU promotion, the policy decides at the hook, before
on_schedule_end() would issue the I/O.

POLICY env var (arms of the outline's ladder):
  always_load        A1  current vLLM behaviour (hit -> load)
  always_recompute       never use the offload tiers (DRAM or SSD): reference point = vLLM without offloading
  static:<tokens>    A3  recompute if h_ssd < tokens (SSD tier only; DRAM hits always load)
  table              A4  2-D winner table from profile.json (h x occ bin, batch-level)
  tokenflow          A5  recompute iff t_io > t_rc, t_rc = h * sliding-window per-token prefill latency
                         (TokenFlow EuroSys'26 §3.3/§4.2.3: observed recent prefill steps, so it tracks GPU
                         load; no Opp). t_io = the same queue-aware estimate ours uses. TF_WIN steps (32),
                         TF_MIN_PF prefill tokens for a step to count (256). Before A2-k32 this arm used the
                         profile's load-independent average — rerun older tokenflow numbers if they matter.
  tokenflow         (was) min(t_io(h, store), t_rc_avg * h): I/O load-aware, recompute
                         history-average, no Opp
  ours               A7  T_load(h, store) < T_rc(h, occ) + BETA * Opp(h, occ)
  rand:<p0>:<p1>[:<qth>]  motivation 무작위 배정: SSD hit마다 결정 시점 SSD load queue q(토큰,
                         자기 자신 제외 = io_promo_blocks*bs)를 읽어 p = p1 if q > qth else p0 (qth 기본 0),
                         u = sha1(request_id) -> [0,1), u < p 이면 recompute. DRAM hit은 항상 load.
                         결정마다 RAND_LOG(기본 PROBE_LOG의 .jsonl -> _rand.jsonl)에 한 줄 기록.
PROFILE=profile.json  BETA=1.0 (Opp weight)  PROBE_LOG as before.

A2 (I/O-busy regime) — LD_INDEX=promo switches the SSD load cost to a queue-aware model
fitted in A0 (local SSD, 2 MiB O_DIRECT, 16 read threads), used by ours AND tokenflow:
  T_load(h, Q) = max( h*B/BW1 , GAMMA * (Q + h)*B / BWSAT )
    Q = io_promo_blocks * block_size (+ LDQ_STORE_W * io_store_blocks): read work ahead of us
  Opp_io(h)    = min(J*h, Q)*B / BWSAT      J = io_promo_jobs; the delay our load adds to the
                                           loads already in flight (processor sharing)
  T_rc         = max( profile rc(h, occ) , (P + h) / rcq_thr(h) )
    P = prefill tokens still to compute for running + waiting requests (GPU work ahead of us)
  Opp_pf(h)    = [sum min(h, rem_j) over prefilling/recompute-queued + n_load_waiting * h] / RCQ_THR
                 (delay a recompute adds to other requests waiting on GPU prefill budget)
  Opp_io_fut   = (h*B/BWSAT) * lam_load * T_drain,  T_drain = (Q*B/BWSAT) / max(IOQ_EPS, 1 - rho_io)
                 (loads that will arrive before the current I/O backlog drains are all delayed by our
                 transfer too — the M/G/1 busy-period externality; ~0 when idle, grows as rho_io -> 1.
                 rho_io, lam_load from the SSD loads this connector chose in the last IOQ_WIN s)
  Opp_pf_fut   = (h/RCQ_THR) * lam_pf * T_drain_gpu,  T_drain_gpu = (P/RCQ_THR) / max(IOQ_EPS, 1 - rho_gpu)
                 (the same busy-period term on the GPU side, so neither resource can be pushed past
                 saturation; rho_gpu = prefill tokens executed in the last IOQ_WIN s / RCQ_THR,
                 lam_pf = new prefill requests scheduled per s)
  ours:  recompute  <=>  T_rc + BETA*Opp_gpu + BETA_PF*(Opp_pf + BETA_IOF*Opp_pf_fut)
                         <  T_load + BETA_IO*(Opp_io + BETA_IOF*Opp_io_fut)
Env: LDQ_BW1_GBPS=1.52 LDQ_BWSAT_GBPS=2.23 LDQ_GAMMA=1.15 LDQ_STORE_W=0 B_KV=131072
     RCQ_THR_TOKS=11500 (0 = off) BETA_IO=1.0
     IO_SIGNAL=store|promo|both : what _store_blocks() returns (the online tables' I/O axis)

Decision log line (probe_log.jsonl) adds: arm, h_ssd, h_dram, est_ld_ms, est_rc_ms,
est_opp_ms, occ_bin, store_bin.

VERIFY markers: attribute names read from vLLM 0.23.0 internals seen via grep.
"""
from __future__ import annotations

import bisect, collections, hashlib, json, math, os, time
from typing import TYPE_CHECKING, Any

from probe_connector import ProbeOffloadingConnector, logger  # same dir on PYTHONPATH
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import OffloadingConnector
import probe_connector as _pc

if TYPE_CHECKING:
    from vllm.v1.request import Request

import os
# ---- SSD store gate: env NO_SSD_STORE=1 이면 항상, 아니면 플래그 파일이 존재하는 동안 fs tier 쓰기를 건너뜀 ----
# (읽기/lookup/job 회계는 그대로. connector 모듈은 EngineCore 프로세스에서 import 되므로 fs 매니저와 같은 프로세스에 적용됨)
import vllm.v1.kv_offload.tiering.fs.manager as _fsm
_ORIG_STORE_BLOCK = _fsm.store_block
_NO_STORE_FLAG = os.environ.get("NO_SSD_STORE_FLAG", "/tmp/NO_SSD_STORE")

def _gated_store_block(*args, **kwargs):
    if os.environ.get("NO_SSD_STORE") == "1" or os.path.exists(_NO_STORE_FLAG):
        return None
    return _ORIG_STORE_BLOCK(*args, **kwargs)

_fsm.store_block = _gated_store_block
logger.warning("SSD store gated: env NO_SSD_STORE or flag file %s", _NO_STORE_FLAG)

POLICY = os.environ.get("POLICY", "always_load")
BETA = float(os.environ.get("BETA", "1.0"))
# KV-pressure guard: recompute needs h tokens of fresh GPU KV space *before* it can run,
# while load only copies into space as it frees up. When the GPU KV cache is nearly full,
# recompute therefore forces preemptions (setting B: 30 -> 52-76 with recompute share).
# KV_GUARD=0.9 -> never choose recompute when KV usage >= 90%. Unset/empty -> off.
KV_GUARD = float(os.environ["KV_GUARD"]) if os.environ.get("KV_GUARD") else None
PROFILE_PATH = os.environ.get("PROFILE", "profile.json")
# ---- A2 queue-aware cost model (see docstring). Defaults = A0 fit on the local SSD tier.
LDQ_BW1 = float(os.environ.get("LDQ_BW1_GBPS", "1.52")) * 1e9      # one load alone
LDQ_BWSAT = float(os.environ.get("LDQ_BWSAT_GBPS", "2.23")) * 1e9  # device saturated
LDQ_GAMMA = float(os.environ.get("LDQ_GAMMA", "1.15"))             # open-loop sharing correction
LDQ_STORE_W = float(os.environ.get("LDQ_STORE_W", "0"))            # weight of in-flight store blocks in Q
B_KV = int(os.environ.get("B_KV", str(128 * 1024)))
RCQ_THR = float(os.environ.get("RCQ_THR_TOKS", "11500"))            # GPU prefill tok/s at saturation
MOTIV_LOG = os.environ.get("MOTIV_LOG", "0") == "1"   # motivation용 읽기 전용 상태 로그 (결정 불변)
ONLINE_HW = os.environ.get("ONLINE_HW", "0") == "1"      # 장치 값 온라인 추정 (online_hw.py)
if ONLINE_HW:
    import online_hw as _ohw
    if _ohw.V2 or _ohw.FIFO_SSD or _ohw.IO_DONE:
        import fs_queue as _fsq                          # fs tier 스레드 풀 계측 (풀 생성 전에 감싸야 함)
        _fsq.install()
    else:
        _fsq = None
else:
    _fsq = None

# 길이별 GPU prefill 포화 처리량 (A0 길이 스윕). 예: RCQ_TABLE="8192:11500,32768:8000,61440:6000"
# 미설정이면 RCQ_THR 상수 사용 (기존 동작). h는 범위 밖에서 끝값으로 고정, 사이는 log-h 선형 보간.
_RCQ_PTS = sorted((float(k), float(v)) for k, v in
                  (x.split(":") for x in os.environ.get("RCQ_TABLE", "").replace(" ", "").split(",") if x))
def rcq_thr(h):
    if not _RCQ_PTS or h <= 0: return RCQ_THR
    if h <= _RCQ_PTS[0][0]: return _RCQ_PTS[0][1]
    if h >= _RCQ_PTS[-1][0]: return _RCQ_PTS[-1][1]
    for (h0, v0), (h1, v1) in zip(_RCQ_PTS, _RCQ_PTS[1:]):
        if h0 <= h <= h1:
            t = (math.log(h) - math.log(h0)) / (math.log(h1) - math.log(h0))
            return v0 + t * (v1 - v0)
    return RCQ_THR

def _c_pf(L):
    """길이 L 프롬프트 전체 prefill 시간 (s). 길이별 실측 처리량 rcq_thr(L) 사용."""
    return L / rcq_thr(L) if L > 0 else 0.0

BETA_IO = float(os.environ.get("BETA_IO", "1.0"))
BETA_PF = float(os.environ.get("BETA_PF", "1.0"))
# FROZEN METHOD = LIO2_k32 ours: BETA_IOF=0, PF_LANDED=0 (defaults). The busy-period terms and landed counting
# are kept for future work but OFF: in LIO2_k32f/g they made ours stop recomputing, because (1) _decided/_landed
# leak (vLLM does not call request_finished for every request: nload grew to 621 of 1183 requests) and (2) rho_io
# from *chosen* load bytes falls exactly when the system saturates (fewer decisions) — it needs a device-side
# estimate (read-queue growth) instead.
BETA_IOF = float(os.environ.get("BETA_IOF", "0.0"))  # weight of the future-arrival externalities (I/O and GPU)
PF_LANDED = os.environ.get("PF_LANDED", "0") == "1"  # count landed loads in Opp_pf (needs leak-free lifecycle tracking)
IOQ_WIN = float(os.environ.get("IOQ_WIN", "30"))     # s of load history for rho_io / lam_load
IOQ_EPS = float(os.environ.get("IOQ_EPS", "0.05"))   # floor of (1 - rho_io)
IOQ_TMAX = float(os.environ.get("IOQ_TMAX", "60"))   # cap on T_drain (s)
TF_WIN = int(os.environ.get("TF_WIN", "32"))          # tokenflow: sliding window of prefill-carrying steps
TF_MIN_PF = int(os.environ.get("TF_MIN_PF", "256"))   # a step counts if it carried >= this many prefill tokens   # weight of Opp_pf (recompute delaying other prefills); 0 = LIO2_c32 behaviour
IO_SIGNAL = os.environ.get("IO_SIGNAL", "store")

class Profile:
    def __init__(self, path: str):
        p = json.load(open(path))
        self.h = p["h_grid"]; self.bs = p["block_size"]
        self.occ_edges = p["occ_edges"]; self.store_edges = p["store_edges"]
        self.rc = {int(k): v for k, v in p["rc_ms"].items()}
        self.opp = {int(k): v for k, v in p["opp_ms"].items()}
        self.ld = {int(k): v for k, v in p["ld_ssd_ms"].items()}
        # v2 profiles also carry load cost indexed by occupancy (store signal proved
        # uninformative under a tiny CPU staging tier). LD_INDEX=occ selects it.
        self.ld_index = os.environ.get("LD_INDEX", "store")
        n_occ = len(self.occ_edges) + 1
        if "ld_ssd_by_occ_ms" in p and all(len(v) == n_occ for v in p["ld_ssd_by_occ_ms"].values()):
            self.ld_occ = {int(k): v for k, v in p["ld_ssd_by_occ_ms"].items()}
        else:
            self.ld_occ = None
            if self.ld_index == "occ":
                logger.warning("profile has no occ-indexed load table; falling back to LD_INDEX=store")
                self.ld_index = "store"
        self.ld_dram = {int(k): v for k, v in p["ld_dram_ms"].items()}
        self.winner = {int(k): v for k, v in p["winner_a4"].items()}
        self.rc_us_avg = p["rc_us_per_tok_avg"]

    def occ_bin(self, occ): return bisect.bisect_right(self.occ_edges, occ)
    def store_bin(self, s): return bisect.bisect_right(self.store_edges, s)

    def _interp(self, table, h, b=None):
        """log-linear interpolation along h; b = column index (None for 1-D tables)."""
        h = max(self.h[0], min(self.h[-1], h))
        i = bisect.bisect_left(self.h, h)
        if i == 0 or self.h[i] == h:
            v = table[self.h[i]]; return v[min(b, len(v) - 1)] if b is not None else v
        h0, h1 = self.h[i - 1], self.h[i]
        v0, v1 = table[h0], table[h1]
        if b is not None: v0, v1 = v0[min(b, len(v0) - 1)], v1[min(b, len(v1) - 1)]
        t = (math.log(h) - math.log(h0)) / (math.log(h1) - math.log(h0))
        return v0 + t * (v1 - v0) if (v0 <= 0 or v1 <= 0) else math.exp(math.log(v0) + t * (math.log(v1) - math.log(v0)))

    def _rc_ext_scale(self, h):

        """h > h_max이면 rc0(h)/rc0(h_max) (RC_EXT, log-h 선형 보간, 끝 너머는 마지막 두 점으로 log-log 외삽). 아니면 1."""

        pts = getattr(self, "_rc_ext_pts", None)

        if pts is None:

            raw = os.environ.get("RC_EXT", "").replace(" ", "")

            pts = sorted((float(k), float(v)) for k, v in (x.split(":") for x in raw.split(",") if x))

            if pts and not any(abs(k - self.h[-1]) < 1 for k, _ in pts):

                logger.warning("RC_EXT에 h_max=%d 점이 없음 → 무시", self.h[-1]); pts = []

            self._rc_ext_pts = pts

        if not pts or h <= self.h[-1]: return 1.0

        def f(x):

            if x <= pts[0][0]: return pts[0][1]

            for (a0, b0), (a1, b1) in zip(pts, pts[1:]):

                if a0 <= x <= a1:

                    t = (math.log(x) - math.log(a0)) / (math.log(a1) - math.log(a0)); return b0 + t * (b1 - b0)

            (a0, b0), (a1, b1) = pts[-2], pts[-1]

            k = math.log(b1 / b0) / math.log(a1 / a0); return b1 * (x / a1) ** k

        return f(h) / f(self.h[-1])

    def rc_ms(self, h, occ):   return self._interp(self.rc, h, self.occ_bin(occ)) * self._rc_ext_scale(h) if h > 0 else 0.0

    def opp_ms(self, h, occ):  return self._interp(self.opp, h, self.occ_bin(occ)) * self._rc_ext_scale(h) if h > 0 else 0.0

    def ld_ssd_ms(self, h, s, occ=None):
        if h <= 0: return 0.0
        if self.ld_index == "occ" and occ is not None and self.ld_occ is not None:
            return self._interp(self.ld_occ, h, self.occ_bin(occ))
        return self._interp(self.ld, h, self.store_bin(s))
    def ld_dram_ms(self, h):   return self._interp(self.ld_dram, h) if h > 0 else 0.0
    def a4_winner(self, h, occ):
        hg = min(self.h, key=lambda g: abs(math.log(g) - math.log(max(h, 1))))
        return self.winner[hg][self.occ_bin(occ)]

class PolicyOffloadingConnector(ProbeOffloadingConnector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profile = Profile(PROFILE_PATH)
        self.arm = POLICY
        # request_id -> "load" | "recompute": the scheduler re-queries a request every step while
        # it waits for KV space or for its promotion to finish; decide once and repeat that answer.
        self._decided: dict[str, str] = {}
        self._dec_h: dict[str, int] = {}          # rid -> tokens it will load (for the prefill backlog)
        self._cur_backlog = 0
        self._tf_win = collections.deque(maxlen=TF_WIN)   # (step_ms, tokens) of recent prefill-carrying steps
        self._ld_hist = collections.deque()                # (t, h_ssd) of SSD loads chosen, last IOQ_WIN s
        self._pf_hist = collections.deque()                # (t, prefill tokens, new reqs) per step, last IOQ_WIN s
        self._landed: set[str] = set()                     # load-decided requests whose KV reached the CPU tier (GPU-ready)
        self._io_fut = {}
        self._tf_prev = None                               # (t_call, tokens, prefill_tokens) of the step scheduled last call
        self._pf_rem: list[int] = []              # remaining prefill of requests competing for GPU prefill budget
        self._pf_nload = 0                        # waiting requests that chose load (need GPU once their KV lands)
        self._hw = _ohw.OnlineHW() if ONLINE_HW else None
        self._hw_log_t = 0.0
        self.static_thr = int(POLICY.split(":")[1]) if POLICY.startswith("static") else None
        self.rand_cfg = None
        if POLICY.startswith("rand"):
            _a = POLICY.split(":")
            self.rand_cfg = (float(_a[1]), float(_a[2]), float(_a[3]) if len(_a) > 3 else 0.0)   # (p0, p1, qth)
            logger.info("rand arm: p0=%.3f p1=%.3f qth=%.0f", *self.rand_cfg)
        self._rand_fh = None
        self._cur_rid = None
        logger.info("PolicyOffloadingConnector arm=%s beta=%.2f profile=%s ld_index=%s kv_guard=%s",
                    self.arm, BETA, PROFILE_PATH, self.profile.ld_index, KV_GUARD)
        logger.info("A2 model: io_signal=%s bw1=%.2f bwsat=%.2f GB/s gamma=%.2f store_w=%.2f rcq_thr=%.0f beta_io=%.2f beta_pf=%.2f beta_iof=%.2f pf_landed=%d ioq_win=%.0f",
                    IO_SIGNAL, LDQ_BW1 / 1e9, LDQ_BWSAT / 1e9, LDQ_GAMMA, LDQ_STORE_W, RCQ_THR, BETA_IO, BETA_PF, BETA_IOF, int(PF_LANDED), IOQ_WIN)

    # ---- state helpers ----------------------------------------------------
    def _occ_tokens(self):
        s = _pc._SCHEDULER
        return sum(getattr(r, "num_computed_tokens", 0) for r in getattr(s, "running", [])) if s else 0

    def _store_blocks(self):
        io = self._io_state()
        if IO_SIGNAL == "promo": return io["io_promo_blocks"]
        if IO_SIGNAL == "both": return io["io_promo_blocks"] + io["io_store_blocks"]
        return io["io_store_blocks"]

    # ---- A2 helpers ---------------------------------------------------------
    def _prefill_backlog(self, exclude_rid=None) -> int:
        """Prefill tokens the GPU has committed to: running requests' remaining prefill + waiting
        requests that already chose recompute. Waiting requests not yet looked up are NOT counted —
        in a long queue most of them will be offload hits, and counting their full prompts made
        est_rc explode so ours never recomputed (LIO2 pilot). VERIFY: waiting is iterable."""
        s = _pc._SCHEDULER
        self._pf_rem = []; self._pf_nload = 0; self._pf_items = []
        if s is None: return 0
        tot = 0
        try:
            running = list(getattr(s, "running", []))
            run_ids = {getattr(r, "request_id", None) for r in running}
            # load-decided, not yet running: all of them (nload, for the log) and those whose KV has landed
            # (GPU-ready; only these are delayed by our recompute — the rest wait on the SSD, not the GPU).
            # LIO2_k32f counted all nload as landed -> Opp_pf of tens of seconds -> ours never recomputed.
            self._landed -= run_ids
            self._pf_nload = sum(1 for rid, d in self._decided.items()
                                 if d == "load" and rid not in run_ids and rid != exclude_rid)
            self._pf_nlanded = sum(1 for rid in self._landed if rid != exclude_rid)
            for r in running:
                L_ = getattr(r, "num_prompt_tokens", 0); a_ = getattr(r, "num_computed_tokens", 0)
                rem = max(0, L_ - a_)
                tot += rem
                if rem > 0: self._pf_rem.append(rem); self._pf_items.append((a_, L_))
            for r in list(getattr(s, "waiting", [])):
                rid = getattr(r, "request_id", None)
                if rid == exclude_rid: continue
                d = self._decided.get(rid)
                if d == "recompute":
                    L_ = getattr(r, "num_prompt_tokens", 0); a_ = getattr(r, "num_computed_tokens", 0)
                    rem = max(0, L_ - a_)
                    tot += rem; self._pf_rem.append(rem)
                    if rem > 0: self._pf_items.append((a_, L_))
                # (load-decided requests are counted above from self._decided)
        except Exception:
            pass
        return tot

    def _opp_pf_ms(self, h, io, t_rc_ms) -> float:
        """Delay a recompute of h adds to other requests waiting on GPU prefill budget (running first,
        so a long recompute holds the budget for ~h/RCQ_THR): prefilling / recompute-queued requests by
        min(h, their remaining); load-waiting ones by h, but only those whose KV lands before our
        recompute ends — already-landed ones (load-waiting minus in-flight promotion jobs) plus the
        share of in-flight ones that the device (PS at BWSAT) completes within t_rc."""
        if h <= 0 or RCQ_THR <= 0: return 0.0
        J = io["io_promo_jobs"]; q_bytes = self._q_tokens(io) * B_KV
        landed = getattr(self, "_pf_nlanded", 0) if PF_LANDED else 0
        soon = J if q_bytes <= 0 else min(J, J * LDQ_BWSAT * (t_rc_ms / 1000.0) / q_bytes)
        self._pf_neff = landed + soon
        return 1000.0 * (sum(_c_pf(min(h, L_ - a_)) for a_, L_ in getattr(self, "_pf_items", []))
                         + self._pf_neff * _c_pf(h))

    def _q_tokens(self, io) -> float:
        return (io["io_promo_blocks"] + LDQ_STORE_W * io["io_store_blocks"]) * self.profile.bs

    def _ld_ssd_q_ms(self, h, io) -> float:
        if h <= 0: return 0.0
        solo = h * B_KV / LDQ_BW1
        shared = LDQ_GAMMA * (self._q_tokens(io) + h) * B_KV / LDQ_BWSAT
        return 1000.0 * max(solo, shared)

    def _opp_io_fut_ms(self, h, io) -> float:
        now = time.time(); hist = self._ld_hist
        while hist and hist[0][0] < now - IOQ_WIN: hist.popleft()
        if h <= 0 or not hist:
            self._io_fut = {"rho_io": 0.0, "t_drain_s": 0.0}; return 0.0
        span = max(5.0, min(IOQ_WIN, now - hist[0][0]))
        lam = len(hist) / span
        rho = sum(x for _, x in hist) * B_KV / span / LDQ_BWSAT
        t_drain = min(IOQ_TMAX, (self._q_tokens(io) * B_KV / LDQ_BWSAT) / max(IOQ_EPS, 1.0 - rho))
        self._io_fut = {"rho_io": round(rho, 3), "lam_load": round(lam, 3), "t_drain_s": round(t_drain, 2)}
        return 1000.0 * (h * B_KV / LDQ_BWSAT) * lam * t_drain

    def _opp_pf_fut_ms(self, h) -> float:
        now = time.time(); hist = self._pf_hist
        while hist and hist[0][0] < now - IOQ_WIN: hist.popleft()
        if h <= 0 or not hist or RCQ_THR <= 0:
            self._gpu_fut = {"rho_gpu": 0.0, "t_drain_gpu_s": 0.0}; return 0.0
        span = max(5.0, min(IOQ_WIN, now - hist[0][0]))
        rho = sum(p for _, p, _ in hist) / span / RCQ_THR
        lam = sum(n for _, _, n in hist) / span
        t_drain = min(IOQ_TMAX, (self._cur_backlog / RCQ_THR) / max(IOQ_EPS, 1.0 - rho))
        self._gpu_fut = {"rho_gpu": round(rho, 3), "lam_pf": round(lam, 3), "t_drain_gpu_s": round(t_drain, 2)}
        return 1000.0 * (h / rcq_thr(h)) * lam * t_drain

    def _opp_io_ms(self, h, io) -> float:
        if h <= 0: return 0.0
        return 1000.0 * min(io["io_promo_jobs"] * h, self._q_tokens(io)) * B_KV / LDQ_BWSAT

    def _load_queue_blocks(self) -> int:
        """[A0] SSD->CPU promotion blocks sitting in _pending_load_submissions (all requests).
        Logging only — the decision does not use it yet. A0 checks whether this tracks the real
        read queue (device aqu-sz) or only the not-yet-issued slice of it. VERIFY."""
        try:
            mgr = self.connector_scheduler.manager
            return sum(len(sub.keys) for d in mgr._pending_load_submissions.values() for sub in d.values())
        except Exception:
            return -1

    def _kv_usage(self):
        """Fraction of GPU KV blocks in use, or None if unavailable. VERIFY attribute names."""
        s = _pc._SCHEDULER
        if s is None: return None
        try:
            km = s.kv_cache_manager
            u = getattr(km, "usage", None)
            if u is not None: return float(u)
            bp = getattr(km, "block_pool", None)
            if bp is not None and hasattr(bp, "get_usage"): return float(bp.get_usage())
        except Exception:
            pass
        return None

    def _dram_prefix_blocks(self, request, num_computed_tokens) -> int:
        """Leading blocks (beyond the HBM prefix) already resident in the CPU tier.
        VERIFY: group_states[0].offload_keys / req_context / primary_tier.lookup."""
        try:
            st = self.connector_scheduler._req_status[request.request_id]
            gs = st.group_states[0]
            keys = gs.offload_keys
            bs = self.profile.bs
            start = num_computed_tokens // bs
            mgr = self.connector_scheduler.manager
            n = 0
            for k in keys[start:]:
                if mgr.primary_tier.lookup(k, st.req_context) is True: n += 1
                else: break
            return n
        except Exception:
            return 0

    # ---- the policy -------------------------------------------------------
    def _policy_decide(self, h_ssd, h_dram, occ, store):
        """Return (decision, est_ld, est_rc, est_opp)."""
        P = self.profile; h_tot = h_ssd + h_dram
        if self.rand_cfg is not None:
            return self._rand_decide(h_ssd, h_dram)
        if MOTIV_LOG and self.arm == "always_load":
            self._last_a2 = None
            return "load", 0.0, 0.0, 0.0
        if self._hw is not None and self.arm == "ours":
            return self._online_decide(h_ssd, h_dram)
        est_ld = P.ld_ssd_ms(h_ssd, store, occ) + P.ld_dram_ms(h_dram)
        est_rc = P.rc_ms(h_tot, occ)
        est_opp = P.opp_ms(h_tot, occ)
        est_opp_io = 0.0; est_opp_pf = 0.0; est_opp_io_fut = 0.0; est_opp_pf_fut = 0.0
        self._gpu_fut = {}
        self._last_a2 = None
        if P.ld_index == "promo":
            io = self._io_state()
            est_ld = self._ld_ssd_q_ms(h_ssd, io) + P.ld_dram_ms(h_dram)
            est_opp_io = self._opp_io_ms(h_ssd, io)
            est_opp_io_fut = self._opp_io_fut_ms(h_ssd, io)
            est_opp_pf_fut = self._opp_pf_fut_ms(h_tot)
            rc_q = 1000.0 * (sum(_c_pf(L_) - _c_pf(a_) for a_, L_ in getattr(self, "_pf_items", []))
                             + _c_pf(h_tot)) if RCQ_THR > 0 else 0.0
            est_rc = max(est_rc, rc_q)
            est_opp_pf = self._opp_pf_ms(h_tot, io, est_rc)
            self._last_a2 = {"ld_model": "promo", "q_tok": int(self._q_tokens(io)), "promo_jobs": io["io_promo_jobs"],
                             "rc_backlog_tok": int(self._cur_backlog), "est_rc_q_ms": round(rc_q, 1),
                             "est_opp_io_ms": round(est_opp_io, 1), "est_opp_pf_ms": round(est_opp_pf, 1),
                             "pf_n": len(self._pf_rem), "pf_nload": self._pf_nload,
                             "pf_neff": round(getattr(self, "_pf_neff", 0.0), 2), "pf_nlanded": getattr(self, "_pf_nlanded", 0),
                             "est_opp_io_fut_ms": round(est_opp_io_fut, 1), **self._io_fut,
                             "est_opp_pf_fut_ms": round(est_opp_pf_fut, 1), **self._gpu_fut}
        if self.arm == "always_load":
            d = "load"
        elif self.arm == "always_recompute":
            d = "recompute"
        elif h_ssd == 0 and self.arm in ("table",):
            d = "load"                      # winner table is built for the SSD tier only
        elif self.static_thr is not None:
            # tier-specific static thresholds: SSD uses the zero-load crossover; DRAM's crossover is
            # below the measured range on this hardware, so its threshold is 0 (always load).
            d = ("recompute" if h_ssd < self.static_thr else "load") if h_ssd > 0 else "load"
        elif self.arm == "table":
            d = P.a4_winner(h_tot, occ)
        elif self.arm == "tokenflow":
            us = self._tf_us_per_tok()
            t_rc = (us if us is not None else P.rc_us_avg) * h_tot / 1000.0
            d = "recompute" if t_rc < est_ld else "load"
            est_rc, est_opp = t_rc, 0.0
            if self._last_a2 is not None:
                self._last_a2["tf_us_per_tok"] = None if us is None else round(us, 1)
                self._last_a2["tf_win_n"] = len(self._tf_win)
        elif self.arm == "ours":
            d = "recompute" if (est_rc + BETA * est_opp + BETA_PF * (est_opp_pf + BETA_IOF * est_opp_pf_fut)
                                < est_ld + BETA_IO * (est_opp_io + BETA_IOF * est_opp_io_fut)) else "load"
        else:
            raise ValueError(f"unknown POLICY {self.arm}")
        return d, est_ld, est_rc, est_opp

    def _rand_decide(self, h_ssd, h_dram):
        """motivation 무작위 배정. 결정에 쓴 q와 p, u를 그대로 기록 (분석 구간 = 결정 변수)."""
        p0, p1, qth = self.rand_cfg
        rid = str(self._cur_rid)
        try:
            q = int(self._io_state()["io_promo_blocks"]) * self.profile.bs
        except Exception:
            q = -1
        if h_ssd <= 0:
            d, p, u = "load", 0.0, 1.0
        else:
            p = p1 if q > qth else p0
            u = int.from_bytes(hashlib.sha1(rid.encode()).digest()[:8], "big") / 2.0 ** 64
            d = "recompute" if u < p else "load"
        self._last_a2 = {"rand_q_tok": q, "rand_p": p, "rand_u": round(u, 6)}
        try:
            if self._rand_fh is None:      # 결정은 스케줄러 프로세스에서만 → 첫 결정 때 연다 (서버 기동마다 새 파일)
                path = os.environ.get("RAND_LOG") or os.environ.get("PROBE_LOG", "probe_log.jsonl").replace(".jsonl", "") + "_rand.jsonl"
                self._rand_fh = open(path, "w", buffering=1)
            self._rand_fh.write(json.dumps({"rid": rid, "t": round(time.time(), 4), "h_dram": h_dram, "h_ssd": h_ssd,
                                            "q_tok": q, "p": p, "u": round(u, 6), "arm": d}) + "\n")
        except Exception as e:
            logger.warning("rand log write failed: %s", e)
        return d, 0.0, 0.0, 0.0

    def _motiv_state(self):
        """결정 순간의 상태 (읽기만 함)."""
        try:
            s = _pc._SCHEDULER
            run = list(getattr(s, "running", []) or []); wait = list(getattr(s, "waiting", []) or [])
            def _done(r): return getattr(r, "num_computed_tokens", 0) >= getattr(r, "num_prompt_tokens", 1 << 30)
            pf = [max(0, getattr(r, "num_prompt_tokens", 0) - getattr(r, "num_computed_tokens", 0))
                  for r in run if not _done(r)]
            st = {"m_t": round(time.time(), 4), "m_n_dec": sum(1 for r in run if _done(r)),
                  "m_n_run": len(run), "m_n_wait": len(wait), "m_n_pf": len(pf), "m_pf_rem_tok": sum(pf)}
            try:
                st["m_io"] = dict(self._io_state())
            except Exception:
                pass
            return st
        except Exception as e:
            return {"m_err": str(e)[:80]}

    def _n_dec(self):
        try:
            import probe_connector as _pcm
            return sum(1 for r in list(getattr(_pcm._SCHEDULER, "running", []) or [])
                       if getattr(r, "num_computed_tokens", 0) >= getattr(r, "num_prompt_tokens", 1 << 30))
        except Exception:
            return 0

    def _budget_cfg(self):
        """스케줄러 설정의 스텝당 토큰 예산 (vLLM V1: max_num_scheduled_tokens). 못 읽으면 None → 관찰 최대값 사용."""
        sch = _pc._SCHEDULER
        for get in (lambda: sch.max_num_scheduled_tokens,                    # VERIFY 0.23 속성 이름
                    lambda: sch.scheduler_config.max_num_batched_tokens):
            try:
                v = int(get())
                if v > 0: return v
            except Exception:
                pass
        return None

    def _online_decide(self, h_ssd, h_dram):
        """온라인 장치 모델로 다섯 비용 계산. 준비 전에는 load (vLLM 기본 동작)."""
        hw = self._hw; P = self.profile; bs = P.bs
        if h_ssd <= 0:                                   # CPU tier만: PCIe 복사가 압도적으로 빠름 → 비교 없이 load
            self._last_a2 = {"online": 1, "dram_only": True}
            return "load", 0.0, 0.0, 0.0
        io = self._io_state()
        qr = io["io_promo_blocks"] * bs; qw = io["io_store_blocks"] * bs; jobs = io["io_promo_jobs"]
        items = list(getattr(self, "_pf_items", []))
        n_dec = 0
        try:
            import probe_connector as _pcm
            for r in list(getattr(_pcm._SCHEDULER, "running", []) or []):
                if getattr(r, "num_computed_tokens", 0) >= getattr(r, "num_prompt_tokens", 1 << 30): n_dec += 1
        except Exception:
            pass
        base = {"online": 1, "n_dec": n_dec, "q_tok": int(qr + qw), "promo_jobs": jobs,
                "rc_backlog_tok": int(sum(max(0, L - a) for a, L in items)), "pf_n": len(items), **hw.params()}
        if _ohw.C_CFG and hw.gpu.C_cfg is None:
            hw.gpu.C_cfg = self._budget_cfg()
        if not hw.ready:
            self._last_a2 = base
            return "load", 0.0, 0.0, 0.0
        a0 = getattr(self, "_cur_a", 0); self._cur_a = 0                  # 한 번 쓰고 비움 (다른 경로에 남지 않게)
        s_tok = getattr(self, "_cur_s", 1); self._cur_s = 1
        new_keys = getattr(self, "_cur_new_keys", set()); self._cur_new_keys = set()
        qv = None
        if _fsq is not None:
            try:
                pend = self.connector_scheduler.manager._pending_load_submissions
                other = [len(d_[k].keys) for tier, d_ in pend.items() for k in d_ if (id(tier), k) not in new_keys]
                v = _fsq.view(other)
                if v is not None:
                    qv = {"ahead_tok": v["ahead"] * bs, "last_tok": [(e + 1) * bs if e >= 0 else 0 for e in v["last"]],
                          "store_busy": v["store_busy"], "lam": _fsq.lam(), "h_avg_tok": _fsq.h_avg() * bs,
                          "store_frac": _fsq.store_frac()}
                    base.update({"q_ahead_tok": qv["ahead_tok"], "q_jobs": len(v["last"]), "store_busy": v["store_busy"], "store_frac": round(qv["store_frac"], 3),
                                 "lam_load": round(qv["lam"], 3)})
            except Exception as e:
                base["qv_err"] = str(e)[:80]
        c = hw.costs(h_ssd, h_dram, P.ld_dram_ms(h_dram), qr, qw, jobs, items, n_dec, a0=a0, qv=qv, s_tok=s_tok)
        self._pf_neff = c["n_eff"]
        rc_side = c["est_rc"] + BETA * c["est_opp"] + BETA_PF * c["est_opp_pf"]
        ld_side = c["est_ld"] + BETA_IO * c["est_opp_io"]
        d = "recompute" if rc_side < ld_side else "load"
        self._last_pred = (c["est_ld_raw"], c["est_rc_raw"])
        self._last_a2 = {**base, "est_opp_io_ms": round(c["est_opp_io"], 1), "est_opp_pf_ms": round(c["est_opp_pf"], 1),
                         "est_ld_raw_ms": round(c["est_ld_raw"], 1), "est_rc_raw_ms": round(c["est_rc_raw"], 1),
                         "est_rc_own_ms": round(c["own_rc"], 1), "pf_neff": round(c["n_eff"], 2),
                         "ld_ssd_ms": round(c["ld_ssd_ms"], 1), "ld_gpu_wait_ms": round(c["ld_gpu_wait_ms"], 1),
                         "s_tok": s_tok, "mix_steps": c.get("mix_steps"), "v2_q": c.get("v2_q")}
        return d, c["est_ld"], c["est_rc"], c["est_opp"]

    # ---- decision hook ----------------------------------------------------
    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int):
        if self._policy(request) in ("load", "recompute"):
            return super().get_num_new_matched_tokens(request, num_computed_tokens)  # probe path

        prior = self._decided.get(request.request_id)
        before = self._pending_snapshot()
        t0 = time.perf_counter()
        h, is_async = OffloadingConnector.get_num_new_matched_tokens(self, request, num_computed_tokens)
        t1 = time.perf_counter()
        new_keys = self._pending_snapshot() - before
        if prior == "miss":
            # a miss re-queried while waiting for KV space: stay silent unless blocks have since appeared
            if h is None or h <= 0:
                return (h if h is not None else None), is_async
            prior = None                                   # blocks arrived (stored by another request): decide below
        if prior is not None:
            # Re-query of a request we already decided: repeat the answer, no new decision/log row.
            if prior == "recompute":
                if new_keys: self._purge_new_promotions(new_keys)
                h_dram = self._dram_prefix_blocks(request, num_computed_tokens) * self.profile.bs
                self.connector_scheduler._req_status[request.request_id].update_num_hit_blocks(num_computed_tokens + h_dram)
                return h_dram, h_dram > 0
            if h is not None and h > 0:
                if MOTIV_LOG and request.request_id not in self._landed:
                    self._emit(request, self.arm, "landed", h=h, num_computed=num_computed_tokens, tier="cpu",
                               promo=0, released=0, lookup_us=(t1 - t0) * 1e6, decide_us=0.0,
                               extra=self._motiv_state())
                self._landed.add(request.request_id)      # KV is in the CPU tier: from now on it waits on the GPU
                if self._hw is not None: self._hw.load_done(request.request_id)
                if is_async: self._inflight[request.request_id] = h
            return h, is_async

        # Case A: parent returned a number -> miss (0) or a hit already resident in the CPU tier
        # (or promoted earlier). DRAM hits go through the SAME cost comparison as SSD hits, using
        # the DRAM load table: on this hardware the profile says load for every h, but that is a
        # measured fact about A100 + PCIe 4.0 + an 8B model, not a rule (a faster GPU or a larger
        # model can move the DRAM crossover into range).
        if h is not None:
            if h <= 0:
                self._decided[request.request_id] = "miss"
                if self._hw is not None:
                    try:
                        self._prefill_backlog(request.request_id)
                        _L = getattr(request, "num_prompt_tokens", 0)
                        self._hw.track_miss(request.request_id, _L - num_computed_tokens, num_computed_tokens,
                                            list(getattr(self, "_pf_items", [])), self._n_dec())
                    except Exception:
                        pass
                self._emit(request, self.arm, "miss", h=h, num_computed=num_computed_tokens, tier="none",
                           promo=0, released=0, lookup_us=(t1 - t0) * 1e6, decide_us=0.0,
                           **({"extra": self._motiv_state()} if MOTIV_LOG else {}))
                return h, is_async
            occ = self._occ_tokens(); store = self._store_blocks()
            kv_use = self._kv_usage()
            self._cur_backlog = self._prefill_backlog(request.request_id)
            if self.arm == "always_recompute":
                d, est_ld, est_rc, est_opp = self._policy_decide(0, h, occ, store)   # all of h is in DRAM
            else:
                # CPU tier hit은 항상 load: CPU→GPU 복사는 PCIe라 빠르고 SSD load queue를 거치지 않음
                d, est_ld, est_rc, est_opp = "load", 0.0, 0.0, 0.0
            t2 = time.perf_counter()
            extra = {"arm": self.arm, "h_ssd": 0, "h_dram": h, "est_ld_ms": round(est_ld, 1),
                     "est_rc_ms": round(est_rc, 1), "est_opp_ms": round(est_opp, 1),
                     "occ_bin": self.profile.occ_bin(occ), "store_bin": self.profile.store_bin(store),
                     "kv_usage": None if kv_use is None else round(kv_use, 3)}
            if MOTIV_LOG: extra.update(self._motiv_state())
            self._decided[request.request_id] = d
            if d == "load": self._dec_h[request.request_id] = h
            if d == "load": self._landed.add(request.request_id)     # DRAM hit: KV already in the CPU tier
            if d == "recompute":
                st = self.connector_scheduler._req_status[request.request_id]
                st.update_num_hit_blocks(num_computed_tokens)
                self._emit(request, self.arm, "recompute", h=h, num_computed=num_computed_tokens, tier="dram",
                           promo=0, released=0, lookup_us=(t1 - t0) * 1e6, decide_us=(t2 - t1) * 1e6, extra=extra)
                return 0, False
            self._emit(request, self.arm, "load", h=h, num_computed=num_computed_tokens, tier="dram",
                       promo=0, released=0, lookup_us=(t1 - t0) * 1e6, decide_us=(t2 - t1) * 1e6, extra=extra)
            if is_async: self._inflight[request.request_id] = h
            return h, is_async

        # Case B: deferred but nothing new queued -> someone else's promotion in flight; wait.
        if not new_keys:
            self._emit(request, self.arm, "pending", h=None, num_computed=num_computed_tokens,
                       tier="?", promo=0, released=0, lookup_us=(t1 - t0) * 1e6, decide_us=0.0)
            return None, is_async

        # Case C: this lookup queued SSD->CPU promotions -> SSD hit. Decide here.
        try:
            return self._case_c(request, num_computed_tokens, h, is_async, new_keys, t0, t1)
        except Exception as e:
            self._errors = getattr(self, "_errors", 0) + 1
            if self._errors <= 20 or self._errors % 100 == 0:
                logger.exception("policy decision failed (%d); falling back to load", self._errors)
            self._promoted.add(request.request_id)
            return None, is_async

    def _case_c(self, request, num_computed_tokens, h, is_async, new_keys, t0, t1):
        mgr = self.connector_scheduler.manager
        promo_blocks = sum(len(d[k].keys) for tier, d in mgr._pending_load_submissions.items()
                           for k in d if (id(tier), k) in new_keys)
        h_ssd = promo_blocks * self.profile.bs
        h_dram = self._dram_prefix_blocks(request, num_computed_tokens) * self.profile.bs
        occ = self._occ_tokens(); store = self._store_blocks()
        self._cur_backlog = self._prefill_backlog(request.request_id)
        self._cur_a = num_computed_tokens                                  # 위치 오프셋 (GPU에 남은 앞부분)
        self._cur_rid = request.request_id                                 # rand arm의 배정 해시용
        self._cur_new_keys = new_keys                                      # v2: 이 요청의 pending promotion (큐 순서 계산에서 제외)
        self._cur_s = max(1, getattr(request, "num_prompt_tokens", 0) - num_computed_tokens - h_ssd - h_dram)   # 나머지 프롬프트
        d, est_ld, est_rc, est_opp = self._policy_decide(h_ssd, h_dram, occ, store)
        kv_use = self._kv_usage()
        guarded = False
        if d == "recompute" and KV_GUARD is not None and kv_use is not None and kv_use >= KV_GUARD:
            d, guarded = "load", True            # no room for h fresh tokens -> recompute would preempt
        released = 0
        self._decided[request.request_id] = d
        if d != "recompute": self._dec_h[request.request_id] = h_ssd + h_dram
        if d == "load" and h_ssd > 0: self._ld_hist.append((time.time(), h_ssd))
        if _fsq is not None and d == "load" and h_ssd > 0:
            try:
                pend = mgr._pending_load_submissions
                _fsq.map_request(request.request_id, [d_[k].keys[0] for tier, d_ in pend.items() for k in d_
                                                      if (id(tier), k) in new_keys and d_[k].keys])
            except Exception:
                pass
        if self._hw is not None and getattr(self, "_last_pred", None) is not None:
            self._hw.track(request.request_id, "rc" if d == "recompute" else "ld",
                           self._last_pred[1] if d == "recompute" else self._last_pred[0])
            self._last_pred = None
        if self._hw is not None and d == "load" and h_ssd > 0:
            _io = self._io_state()
            self._hw.load_started(request.request_id, h_ssd, _io["io_promo_blocks"] * self.profile.bs,
                                  _io["io_store_blocks"] * self.profile.bs)
        if d == "recompute":
            st = self.connector_scheduler._req_status[request.request_id]
            released = self._purge_new_promotions(new_keys)          # SSD→CPU promotion 취소
            st.update_num_hit_blocks(num_computed_tokens + h_dram)    # CPU에 있는 앞부분은 load
            ret = (h_dram, h_dram > 0)
            if h_dram > 0: self._dec_h[request.request_id] = h_dram
        else:
            self._promoted.add(request.request_id)
            ret = (None, is_async)
        t2 = time.perf_counter()
        rec_extra = {"arm": self.arm, "h_ssd": h_ssd, "h_dram": h_dram, "est_ld_ms": round(est_ld, 1),
                     "est_rc_ms": round(est_rc, 1), "est_opp_ms": round(est_opp, 1),
                     "occ_bin": self.profile.occ_bin(occ), "store_bin": self.profile.store_bin(store),
                     "kv_usage": None if kv_use is None else round(kv_use, 3), "kv_guarded": guarded}
        # [A0] read-queue candidates: pending promotions of OTHER requests (own new_keys are already
        # queued at this point, so subtract them), in-flight async loads, and the full _io_state().
        if MOTIV_LOG: rec_extra.update(self._motiv_state())
        q = self._load_queue_blocks()
        rec_extra["ldq_pending_blocks"] = (q - promo_blocks) if q >= 0 else -1
        rec_extra["ldq_inflight_tokens"] = sum(getattr(self, "_inflight", {}).values())
        rec_extra["ldq_promoted_reqs"] = len(getattr(self, "_promoted", ()))
        if getattr(self, "_last_a2", None): rec_extra.update(self._last_a2)
        try:
            rec_extra["io_state"] = dict(self._io_state())
        except Exception:
            pass
        self._emit(request, self.arm, d if d == "recompute" else "load_pending",
                   h=h_ssd + h_dram, num_computed=num_computed_tokens, tier="ssd",
                   promo=len(new_keys), released=released,
                   lookup_us=(t1 - t0) * 1e6, decide_us=(t2 - t1) * 1e6, extra=rec_extra)
        return ret

    # ---- tokenflow: sliding-window per-token prefill latency from observed steps ------------------
    def _tf_us_per_tok(self):
        if not self._tf_win: return None
        ms = sum(m for m, _ in self._tf_win); tok = sum(t for _, t in self._tf_win)
        return 1000.0 * ms / tok if tok > 0 else None

    def build_connector_meta(self, scheduler_output):
        if self._hw is not None:
            self._hw.on_step(scheduler_output)
            if _fsq is not None:
                try:
                    self._hw.io_samples(_fsq.tick(), self.profile.bs)
                except Exception:
                    pass
            if time.time() - self._hw_log_t > 30.0:
                self._hw_log_t = time.time(); logger.info("OHW %s", self._hw.params())
        # The wall time between two calls is the execution of the step scheduled at the earlier call,
        # so attribute it to that step's tokens (parent's step log pairs it with the current step).
        now = time.time()
        try:
            nst = getattr(scheduler_output, "num_scheduled_tokens", {}) or {}
            tok = sum(nst.values()); pf = sum(max(0, n - 1) for n in nst.values())   # decode = 1 token/req
        except Exception:
            tok = pf = 0
        try:
            n_new = len(getattr(scheduler_output, "scheduled_new_reqs", []) or [])
        except Exception:
            n_new = 0
        if pf > 0 or n_new > 0:
            self._pf_hist.append((now, pf, n_new))
        prev = self._tf_prev
        if prev is not None and prev[2] >= TF_MIN_PF and prev[1] > 0:
            self._tf_win.append(((now - prev[0]) * 1000.0, prev[1]))
        self._tf_prev = (now, tok, pf)
        return super().build_connector_meta(scheduler_output)

    def request_finished(self, request, block_ids):
        self._decided.pop(request.request_id, None)
        self._dec_h.pop(request.request_id, None)
        self._landed.discard(request.request_id)
        if self._hw is not None: self._hw.forget(request.request_id)
        return super().request_finished(request, block_ids)
