# SPDX-License-Identifier: Apache-2.0
"""online_hw — 결정에 필요한 장치 값을 서빙 중에 관찰해 갱신 (사전 측정·시간표 없음).

load 쪽 내 대기 (①):
    ① = max( SSD 읽기 시간 ,  GPU 앞 prefill 대기열 소진 시간 ) + 한 스텝 (suffix)
    SSD에서 읽는 동안 GPU도 앞 요청들을 처리하므로 둘은 동시에 진행된다. 다 읽어도 GPU 차례가 안 왔으면 기다린다.
    (이 항이 없으면 바쁠 때 load를 과소평가: 실측 고정 시간 load +450 ms vs recompute +190 ms @ LEval 1.25/s)
④·⑤ 피해 대상 (OHW_FIFO=1, 기본):
    vLLM은 실행 중 요청의 prefill을 먼저 처리하고 새 요청(내 recompute)은 남은 예산으로 처리한다.
    → 내 앞의 prefill 요청들은 나보다 먼저 끝나 내 계산 동안 decode 중: ④ = (생성 중 + 앞 prefill 수) × lin(h)
    → 앞 요청은 내가 밀어내지 않음: ⑤ = 곧 도착할 load 수 × 내 계산 시간
    (실측: 이전 식은 ④를 1.4~21배 과소평가. 결정 순간 prefill 중이던 요청이 실제 피해 대상)
④ 겹침 (OHW_REM=1일 때, 기본 끔 — 실측상 효과 미미):
    내 계산은 앞 prefill 대기 Q 스텝 뒤에 시작해 S 스텝. 요청 j는 남은 출력 r_j 스텝 동안만 생성한다.
      생성 중인 요청: [0, r_j],  앞 prefill 요청: [그 요청 prefill이 끝나는 스텝 t_j, t_j + r_j]
      겹침_j = max(0, min(끝_j, Q+S) − max(시작_j, Q)) / S,   ④ = lin(h) × Σ 겹침_j
    (실측: 남은 출력을 무시하면 긴 문서(LooGLE)에서 ④+⑤를 2~6배 과대평가)
SSD 읽기 모델 (load가 끝날 때마다 1표본):
    T_load(h, Qr, Qw) = max( h·B / BW1 ,  (Qr + w·Qw + h)·B / BWeff )
    Qr = 결정 시점 다른 요청의 promotion 대기 토큰, Qw = 저장 대기 토큰, BWeff = 합계 속도 / 손실배율
    BW1 : 줄이 거의 없을 때 끝난 load들의 h·B/T 중앙값
    BWeff, w : 줄이 있을 때의 load들로 T = α·(Qr+h) + β·Qw 를 가중 최소제곱(IRLS) → BWeff = B/α, w = β/α
GPU 스텝 모델에 '섞인 스텝' 항 (OHW_MIX=1, 기본):
    prefill이 하나라도 섞인 스텝은 생성 전용 스텝(CUDA graph)보다 고정적으로 느림 → b_mix·[prefill 있음]
    ④ = 영향받는 생성 요청 수 × (내 계산 스텝 수 × b_mix + lin(h)),  ③·내 계산 시간도 스텝마다 b_mix 포함
    (실측: 이 항이 없으면 짧은 recompute + 생성 요청 다수(ShareGPT)에서 ④를 7~23배 과소평가)
GPU 스텝 모델 (스텝마다 1표본):
    step_ms = b0 + b1·n_dec + b2·Σn + b3·Σ n·(ctx + n/2)        n = prefill 조각 토큰, ctx = 조각 시작 위치
    lin(x)  = b2·x + b3·x²/2         길이 x 프롬프트의 순수 계산 시간
    C       = 관찰된 스텝당 최대 prefill 토큰 (chunk 예산)
예측 vs 실제 기록 (결정 → 그 요청의 prefill 완료):
    load·recompute·miss마다 (실제 ÷ 예측)을 기록만 한다 (로그·분석용). 결정에는 쓰지 않는다 (OHW_CALIB=0, 기본).
    이유: 실측에서 실제 ≈ 1.0 × 장치 모델 예측 + 약 285 ms 이고, 고정 오버헤드는 load·recompute에 똑같이 붙어
    비교에서 상쇄된다. 곱셈 보정은 이 고정 시간을 비율로 바꿔 예측이 큰 쪽을 과대평가하므로 쓰지 않는다.
    (OHW_CALIB=1이면 이전 실험과 같은 곱셈 보정 k_ld·k_rc 사용)
준비 전(SSD 표본 < N_SSD 또는 prefill 스텝 < N_GPU)에는 decide()가 None → 호출 측이 load(vLLM 기본)로 처리.
"""
from __future__ import annotations

import json, math, os, time
from collections import deque

import numpy as np

B_KV = int(os.environ.get("B_KV", "131072"))   # bytes/token: Llama-3.1-8B 131072, Qwen2.5-7B 57344
N_SSD = int(os.environ.get("OHW_N_SSD", "20"))
N_GPU = int(os.environ.get("OHW_N_GPU", "200"))
SSD_WIN = int(os.environ.get("OHW_SSD_WIN", "300"))
GPU_WIN = int(os.environ.get("OHW_GPU_WIN", "4000"))
SSD_REFIT = 10
GPU_REFIT = 200
K_WIN = int(os.environ.get("OHW_K_WIN", "100"))
K_MIN = 10
CALIB = os.environ.get("OHW_CALIB", "0") == "1"      # 기본: 보정 안 함 (비율은 기록만)
# 스케줄러 순서(FIFO)와 일관: 내 앞의 prefill은 먼저 끝나고 내 계산 동안 decode 중 → ④에 포함, ⑤에서는 제외
FIFO = os.environ.get("OHW_FIFO", "1") == "1"      # 기본 켬 (최종 디자인). 0이면 이전 실행(_ourso)과 같은 식
# ④를 요청별 '남은 출력' 기준으로: 내 계산 스텝과 그 요청의 생성 스텝이 겹치는 비율만큼만 피해
REM = os.environ.get("OHW_REM", "0") == "1"      # 남은 출력 겹침 (효과 미미해 기본 끔)
MIX = os.environ.get("OHW_MIX", "1") == "1"       # 섞인 스텝 고정 비용 반영 (기본 켬)
OHW_LOG = os.environ.get("OHW_LOG", "")
SKIPIDLE = os.environ.get("OHW_IDLE", "1") == "1"       # 처리 토큰이 없는 빈 스텝(유휴 간격)은 GPU 학습에서 제외
LDEXCL = os.environ.get("OHW_LDEXCL", "1") == "1"     # recompute와 겹친 load는 SSD 학습에서 제외 (재진입 대기 흡수 방지)
GUARD = os.environ.get("OHW_GUARD", "1") == "1"         # b2·b3가 0인 재학습 결과는 거부하고 이전 계수 유지
OFFSET = os.environ.get("OHW_OFF", "1") == "1"         # 내 계산 비용을 위치 a0부터: lin(a0+h) − lin(a0)
PARTIAL = os.environ.get("OHW_PARTIAL", "1") == "1"    # recompute 시 CPU tier에 있는 앞부분(h_dram)은 load, SSD 부분만 계산
                                                       # (policy_connector가 CPU 앞부분을 load로 넘김). 0이면 이전 식(h_ssd+h_dram 전체 계산)
LAG = int(os.environ.get("OHW_LAG", "1"))              # 비동기 스케줄링: 호출 간격은 LAG 스텝 앞 구성의 실행 시간
TOK = os.environ.get("OHW_TOK", "0") == "1"            # 생성·prefill 토큰이 토큰당 비용 공유 (b2 식별)
STEP_LOG = os.environ.get("OHW_STEP_LOG", "")          # 스텝별 원시 특징 로그 (결정에는 영향 없음)
# ---- v2 (fs tier 구조 반영). OHW_V2=1이면 아래 플래그가 모두 켜지고, 하나씩 끌 수 있다. 끄면 이전 동작과 같다.
V2 = os.environ.get("OHW_V2", "0") == "1"
def _flag(name): return os.environ.get(name, "1" if V2 else "0") == "1"
FIFO_SSD = _flag("OHW_FIFO_SSD")      # ① SSD 모델: (앞 load 블록 + h) × 읽기 시간, 읽기 속도는 store 비율로. 표본은 실제 job 완료
IO_DONE = _flag("OHW_IO_DONE")        # 이전 SSD 모델도 표본 끝을 실제 job 완료로 (FIFO_SSD가 꺼져 있을 때 의미 있음)
OPP_IO_FUT = _flag("OHW_OPP_IO_FUT")  # ② 내 뒤에 들어올 load가 받는 지연
PF_QUEUE = _flag("OHW_PF_QUEUE")      # ⑤ load 큐 순서로 끝날 load를 세고 도착 시점별 대기
MIX_DIFF = _flag("OHW_MIX_DIFF")      # ④ 추가 지연은 load 대비 늘어난 섞인 스텝 수만큼
C_CFG = _flag("OHW_C_CFG")            # 스텝 예산 = 스케줄러 설정값
PF_FUT = os.environ.get("OHW_PF_FUT", "0") == "1"   # ⑤ 결정 뒤 새 load도 셈. 검증에서 그런 load가 0개라 기본 끔
F_AVG = _flag("OHW_F_AVG")            # 읽기 속도에 쓰는 store 비율: 결정 순간 0/1 대신 최근 F_WIN 초 평균


class SSDModel:
    def __init__(self):
        self.s = deque(maxlen=SSD_WIN)          # (h, qr, qw, T_s)
        self.bw1 = self.bweff = None; self.w = 1.0
        self.n = 0; self._since = 0

    @property
    def ready(self): return self.bw1 is not None and self.bweff is not None and self.n >= N_SSD

    def add(self, h, qr, qw, T):
        if h <= 0 or T <= 0: return
        self.s.append((float(h), float(max(qr, 0)), float(max(qw, 0)), float(T)))
        self.n += 1; self._since += 1
        if self._since >= SSD_REFIT or (self.bw1 is None and self.n >= N_SSD):
            self._since = 0; self.fit()

    def fit(self):
        if len(self.s) < 5: return
        a = np.array(self.s); h, qr, qw, T = a.T
        solo = (qr + qw) < 0.25 * h
        if solo.sum() >= 3: self.bw1 = float(np.median(h[solo] * B_KV / T[solo]))
        elif self.bw1 is None: self.bw1 = float(np.percentile(h * B_KV / T, 90))
        q = ~solo
        if q.sum() >= 5:
            X = np.stack([(qr[q] + h[q]) * B_KV, qw[q] * B_KV], 1); y = T[q]
            wt = np.ones(len(y))
            for _ in range(5):                                    # IRLS: 상대오차 기준, 이상치 완화
                W = wt / np.maximum(y, 1e-3)
                coef, *_ = np.linalg.lstsq(X * W[:, None], y * W, rcond=None)
                coef = np.maximum(coef, [1e-12, 0.0])
                r = np.abs(X @ coef - y) / np.maximum(y, 1e-3)
                wt = 1.0 / np.maximum(r, 0.05)
            self.bweff = float(1.0 / coef[0]); self.w = float(coef[1] / coef[0])
        elif self.bweff is None and self.bw1 is not None:
            self.bweff = self.bw1                                 # 줄 표본이 없으면 혼자 속도로 시작
        if self.bweff is not None and self.bw1 is not None and self.bweff < 0.5 * self.bw1:
            self.bweff = 0.5 * self.bw1                           # 비정상 적합 방지

    def t_load(self, h, qr, qw):
        if h <= 0: return 0.0
        return max(h * B_KV / self.bw1, (qr + self.w * qw + h) * B_KV / self.bweff)

    def params(self):
        return {"bw1_gbps": None if self.bw1 is None else round(self.bw1 / 1e9, 3),
                "bweff_gbps": None if self.bweff is None else round(self.bweff / 1e9, 3),
                "w_store": round(self.w, 2), "n_ssd": self.n}


class SSDModelF:
    """FIFO 읽기 큐 모델 (OHW_FIFO_SSD). 표본은 fs_queue가 잰 실제 job (제출 → I/O 완료).
    T = max( h·B / BW1 ,  (q + h)·B·(a0 + a1·f) )
      q = 제출 순간 load 큐에서 앞에 있던 블록(토큰), f = job이 진행되는 동안 store 큐가 차 있던 비율
      BW_r0 = 1/a0 : store가 없을 때 읽기 속도,  BW_rs = 1/(a0 + a1) : store가 걸려 있을 때 읽기 속도
    store는 load 앞에 서지 않고(별도 큐) 읽기 스레드와 SSD를 나눠 써 속도만 낮춘다 → w 없음."""

    def __init__(self):
        self.s = deque(maxlen=SSD_WIN)          # (h, q, f, T_s)
        self.bw1 = None; self.a = None
        self.n = 0; self._since = 0

    @property
    def ready(self): return self.bw1 is not None and self.a is not None and self.n >= N_SSD

    def add(self, h, q, f, T):
        if h <= 0 or T <= 0: return
        self.s.append((float(h), float(max(q, 0)), float(min(1.0, max(f, 0.0))), float(T)))
        self.n += 1; self._since += 1
        if self._since >= SSD_REFIT or (self.bw1 is None and self.n >= N_SSD):
            self._since = 0; self.fit()

    def fit(self):
        if len(self.s) < 5: return
        a = np.array(self.s); h, q, f, T = a.T
        solo = q < 0.25 * h
        if solo.sum() >= 3: self.bw1 = float(np.median(h[solo] * B_KV / T[solo]))
        elif self.bw1 is None: self.bw1 = float(np.percentile(h * B_KV / T, 90))
        m = ~solo
        if m.sum() >= 5:
            x = (q[m] + h[m]) * B_KV; y = T[m]; fm = f[m]
            two = fm.std() > 0.05                                   # store 비율이 고르면 a1은 식별 불가 → a0만
            X = np.stack([x, x * fm], 1) if two else x[:, None]
            wt = np.ones(len(y))
            for _ in range(5):                                      # IRLS: 상대오차 기준, 이상치 완화
                W = wt / np.maximum(y, 1e-3)
                coef, *_ = np.linalg.lstsq(X * W[:, None], y * W, rcond=None)
                coef = np.maximum(coef, 1e-15 if not two else [1e-15, 0.0])
                r = np.abs(X @ coef - y) / np.maximum(y, 1e-3)
                wt = 1.0 / np.maximum(r, 0.05)
            self.a = [float(coef[0]), float(coef[1]) if two else (self.a[1] if self.a else 0.0)]
        elif self.a is None and self.bw1 is not None:
            self.a = [1.0 / self.bw1, 0.0]                          # 줄 표본이 없으면 단일 속도로 시작
        if self.a is not None and self.bw1 is not None:             # 비정상 적합 방지: 읽기 속도 ≥ 0.5 × 단일
            cap = 2.0 / self.bw1
            self.a[0] = min(self.a[0], cap); self.a[1] = min(self.a[1], cap - self.a[0])

    def sec_per_tok(self, f):
        return B_KV * (self.a[0] + self.a[1] * f)

    def t_load(self, h, q, f):
        if h <= 0: return 0.0
        return max(h * B_KV / self.bw1, (q + h) * self.sec_per_tok(f))

    def params(self):
        g = lambda v: None if v is None or v <= 0 else round(1.0 / v / 1e9, 3)
        return {"f_bw1_gbps": None if self.bw1 is None else round(self.bw1 / 1e9, 3),
                "f_bwr0_gbps": None if self.a is None else g(self.a[0]),
                "f_bwrs_gbps": None if self.a is None else g(self.a[0] + self.a[1]), "n_ssdf": self.n}


class GPUModel:
    def __init__(self):
        self.X = deque(maxlen=GPU_WIN); self.y = deque(maxlen=GPU_WIN)
        self.b = None; self.C = 0; self.n_pf = 0; self._since = 0
        self.C_cfg = None                   # 스케줄러 설정의 스텝 예산 (OHW_C_CFG, connector가 채움)

    @property
    def ready(self): return self.b is not None and self.n_pf >= N_GPU and self.budget > 0

    @property
    def budget(self):                                             # 스텝당 prefill+decode 토큰 예산
        return self.C_cfg if (C_CFG and self.C_cfg) else self.C

    def add(self, n_dec, chunks, step_ms, dctx=0):
        if step_ms <= 0 or step_ms > 5000: return
        if SKIPIDLE and n_dec == 0 and not chunks: return            # 빈 스텝 = 유휴 시간, 계산 표본 아님
        sn = sum(n for n, _ in chunks)
        sq = sum(n * (c + n / 2.0) for n, c in chunks)
        self.X.append((float(n_dec), float(sn), float(sq), float(dctx))); self.y.append(float(step_ms))
        if sn > 0:
            self.n_pf += 1; self.C = max(self.C, int(sn))
        self._since += 1
        if self._since >= GPU_REFIT and self.n_pf >= N_GPU // 2:
            self._since = 0; self.fit()

    @staticmethod
    def _nnls(X, y):                                              # 음수 계수는 0으로 고정 후 재적합
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        keep = np.ones(X.shape[1], bool)
        for _ in range(4):
            neg = (coef < 0) & keep
            if not neg.any(): break
            keep &= ~neg; c2, *_ = np.linalg.lstsq(X[:, keep], y, rcond=None)
            coef = np.zeros(X.shape[1]); coef[keep] = c2
        return coef

    def fit(self):
        R = np.array(self.X); y = np.array(self.y)                # 열: n_dec, Σn, Σq, Σ생성 문맥
        nd, sn, sq, dc = R[:, 0], R[:, 1], R[:, 2], R[:, 3]
        if (sn > 0).sum() < 20 or (sn == 0).sum() < 20: return
        one = np.ones(len(y)); pf = (sn > 0).astype(float)
        if TOK:
            coef = self._nnls(np.column_stack([one, pf, nd + sn, dc, sq]), y)   # [c0, c_mix, c_tok, c_dctx, c3]
            dbar = float(dc.sum() / max(1.0, nd.sum()))           # 생성 요청 1개의 평균 문맥
            self.c_tok = [float(v) for v in coef]; self.dctx_mean = dbar
            self.b = [float(coef[0]), float(coef[2] + coef[3] * dbar), float(coef[1]), float(coef[2]), float(coef[4])]
        else:
            coef = self._nnls(np.column_stack([one, nd, pf, sn, sq]), y)
            nb = [float(v) for v in coef]
            if GUARD and (nb[3] <= 0 or nb[4] <= 0):                 # 물리적으로 말이 안 되는 결과는 거부
                self.n_reject = getattr(self, 'n_reject', 0) + 1; return
            self.b = nb                     # [b0, b1, b_mix, b2, b3]

    def lin(self, x):
        b2, b3 = self.b[3], self.b[4]
        return b2 * x + b3 * x * x / 2.0 if x > 0 else 0.0

    @property
    def bmix(self):
        return self.b[2] if MIX else 0.0

    def overhead(self, n_dec):                                    # 생성 전용 스텝의 기본 시간
        return self.b[0] + self.b[1] * n_dec

    def steps(self, tok, n_dec=0):
        return math.ceil(tok / max(1, self.budget - n_dec)) if tok > 0 else 0

    def c_alone(self, L, n_dec):                                  # 길이 L 프롬프트 전체 prefill 시간 (ms), 스텝은 섞인 스텝
        return self.steps(L, n_dec) * (self.overhead(n_dec) + self.bmix) + self.lin(L)

    def thr(self, L, n_dec=0):                                    # tok/s (로그·검증용)
        t = self.c_alone(L, n_dec)
        return L / (t / 1000.0) if t > 0 else None

    def params(self):
        if self.b is None: return {"gpu_b": None, "chunk": self.C, "n_pf_steps": self.n_pf}
        return {"gpu_b": [round(v, 6) for v in self.b], "b_mix": round(self.b[2], 2), "mix": MIX, "tok": TOK, "lag": LAG, "off": OFFSET, "idle": SKIPIDLE, "guard": GUARD, "n_reject": getattr(self, "n_reject", 0), "c_tok": getattr(self, "c_tok", None), "dctx_mean": round(getattr(self, "dctx_mean", 0.0), 1), "chunk": self.C, "n_pf_steps": self.n_pf,
                "thr8k": round(self.thr(8192) or 0), "thr32k": round(self.thr(32768) or 0),
                "thr60k": round(self.thr(61440) or 0)}


class OnlineHW:
    """connector가 호출하는 진입점. 모든 메서드는 실패해도 예외를 밖으로 내지 않는다."""

    def __init__(self):
        self.ssd = SSDModel(); self.gpu = GPUModel(); self.ssdf = SSDModelF()
        self._land = {}                    # rid -> KV가 CPU tier에 도착한 시각 (fs_queue, 기록용)
        self._rc_iv = deque(maxlen=500); self.n_ld_skip = 0   # 끝난 recompute 계산 구간 (시작, 끝)
        self._ld = {}                      # rid -> (t0, h, qr, qw)
        self._plen = {}                    # rid -> prompt 길이 (new req에서 기록)
        self._prev = None                  # (t, n_dec, chunks) : 이전 스텝 구성
        self._last_log = 0.0
        self.errors = 0
        self._trk = {}                     # rid -> (t0, kind 'ld'|'rc', 예측 ms)
        self._fin = []                     # 이번 스텝에서 prefill이 끝나는 추적 요청
        self.kr = {"ld": deque(maxlen=K_WIN), "rc": deque(maxlen=K_WIN), "rcm": deque(maxlen=K_WIN)}
        self._t_chunk = {}                 # rid -> 첫 prefill 조각이 스케줄된 스텝 시각
        self._logf = open(OHW_LOG, "a") if OHW_LOG else None
        self._stepf = open(STEP_LOG, "a", buffering=1) if STEP_LOG else None

    def k(self, kind):
        if not CALIB: return 1.0
        return self.k_obs(kind)

    def k_obs(self, kind):                                         # 관찰된 비율 (기록용, 결정과 무관)
        d = self.kr[kind]
        if kind == "rc" and len(d) < K_MIN: d = self.kr["rcm"]          # recompute 결정 표본 부족 → miss로 대신
        return float(min(3.0, max(0.5, np.median(d)))) if len(d) >= K_MIN else 1.0

    def track(self, rid, kind, pred_ms):
        if pred_ms and pred_ms > 0: self._trk[rid] = (time.time(), kind, float(pred_ms))

    def track_miss(self, rid, rem, a, items, n_dec):
        """miss = 처음부터(또는 GPU prefix 이후부터) 계산 → recompute와 같은 예측식으로 k_rc 표본."""
        if not self.gpu.ready or rem <= 0: return
        G = self.gpu; rem_tot = sum(max(0, L - x) for x, L in items)
        pred = (G.steps(rem_tot + rem, n_dec) * G.overhead(n_dec) + sum(G.lin(L) - G.lin(x) for x, L in items)
                + G.lin(a + rem) - G.lin(a))
        self.track(rid, "rcm", pred)

    def _complete(self, rid, now):
        x = self._trk.pop(rid, None)
        tc = self._t_chunk.pop(rid, None)
        if x:
            self.kr[x[1]].append(((now - x[0]) * 1000.0) / x[2])
            if x[1] == "rc" and tc is not None: self._rc_iv.append((tc, now))
            if self._logf is not None:
                try:
                    self._logf.write(json.dumps({"rid": rid, "kind": x[1], "t_dec": round(x[0], 4),
                                                 "t_first_chunk": None if tc is None else round(tc, 4),
                                                 "t_done": round(now, 4), "pred_ms": round(x[2], 1),
                                                 "t_land": None if self._land.get(rid) is None else round(self._land[rid], 4)}) + "\n")
                    self._logf.flush()
                except Exception:
                    pass

    @property
    def ready(self): return (self.ssdf.ready if FIFO_SSD else self.ssd.ready) and self.gpu.ready

    def io_samples(self, samples, bs):
        """fs_queue.tick()의 끝난 job들: (블록 수, 앞 블록 수, store 비율, 걸린 시간, rid, 완료 시각)."""
        for n, ahead, f, T, rid, td in samples:
            try:
                self.ssdf.add(n * bs, ahead * bs, f, T)
                if rid is not None:
                    self._land[rid] = td
                    if IO_DONE and not FIFO_SSD:
                        x = self._ld.pop(rid, None)
                        if x: self.ssd.add(x[1], x[2], x[3], td - x[0])
                if len(self._land) > 20000: self._land.clear()
            except Exception:
                self.errors += 1

    # ---- SSD 표본
    def load_started(self, rid, h, qr, qw):
        if h > 0: self._ld[rid] = (time.time(), h, qr, qw)

    def _rc_overlap(self, t0, t1):                                # [t0, t1]에 recompute 계산이 있었나
        for a, b in self._rc_iv:
            if b > t0 and a < t1: return True
        for rid, v in self._trk.items():                          # 지금 계산 중인 recompute
            if v[1] == "rc":
                tc = self._t_chunk.get(rid)
                if tc is not None and tc < t1: return True
        return False

    def load_done(self, rid):
        if IO_DONE or FIFO_SSD:                                    # 표본은 fs_queue의 실제 완료 시각으로 (io_samples)
            if FIFO_SSD: self._ld.pop(rid, None)
            return
        x = self._ld.pop(rid, None)
        if not x: return
        now = time.time()
        if LDEXCL and self._rc_overlap(x[0], now):                # 재진입 대기가 섞인 표본 → SSD 학습에서 제외
            self.n_ld_skip += 1; return
        self.ssd.add(x[1], x[2], x[3], now - x[0])

    def forget(self, rid):
        self._land.pop(rid, None)
        self._ld.pop(rid, None); self._plen.pop(rid, None); self._trk.pop(rid, None); self._t_chunk.pop(rid, None)

    # ---- GPU 표본: 스텝 구성은 '이 스텝 전에 계산된 토큰 수'(SchedulerOutput)로 판단
    def _lag_done(self, now):                                     # LAG=1: 호출 m의 구성은 호출 m+1~m+2에 실행됨
        if LAG == 0: return list(self._fin)
        for r, v in list(self._t_chunk.items()):                  # 지난 호출에서 첫 조각이 스케줄된 요청 → 지금이 실제 시작
            if v is None: self._t_chunk[r] = now
        done = getattr(self, "_fin_prev", []); self._fin_prev = list(self._fin)
        return done                                               # 두 호출 전에 마지막 조각이 스케줄된 요청 → 지금이 실제 끝

    def on_step(self, so):
        now = time.time()
        try:
            nst = getattr(so, "num_scheduled_tokens", {}) or {}
            before = {}
            for nr in (getattr(so, "scheduled_new_reqs", None) or []):
                rid = getattr(nr, "req_id", None)
                if rid is None: continue
                before[rid] = getattr(nr, "num_computed_tokens", 0) or 0
                p = getattr(nr, "prompt_token_ids", None)
                if p is not None: self._plen[rid] = len(p)
            cr = getattr(so, "scheduled_cached_reqs", None)
            if cr is not None:
                ids = getattr(cr, "req_ids", None)
                ncs = getattr(cr, "num_computed_tokens", None)
                if ids is not None and ncs is not None:
                    for rid, nc in zip(ids, ncs): before[rid] = nc
                else:                                             # 구버전: 객체 목록
                    for c in cr:
                        before[getattr(c, "req_id", None)] = getattr(c, "num_computed_tokens", 0)
            for rid in self._lag_done(now): self._complete(rid, now)      # 이전 스텝에서 prefill이 끝난 요청
            self._fin = []
            n_dec, chunks, dctx = 0, [], 0
            for rid, n in nst.items():
                if rid in self._trk:
                    L0 = self._plen.get(rid); a0 = before.get(rid)
                    if rid not in self._t_chunk and (n > 1 or (L0 is not None and a0 is not None and a0 < L0)):
                        self._t_chunk[rid] = (None if LAG else now)                                           # 첫 prefill 조각 스텝
                    if L0 is not None and a0 is not None and a0 + n >= L0: self._fin.append(rid)
                L = self._plen.get(rid); a = before.get(rid)
                if L is None or a is None:
                    if n == 1: n_dec += 1
                    elif n > 1: chunks.append((n, 0))
                    continue
                if a >= L: n_dec += 1; dctx += a
                else: chunks.append((n, a))
            if self._prev is not None:
                ms = (now - self._prev[0]) * 1000.0
                src = self._prev if LAG == 0 else getattr(self, "_pp", None)
                if src is not None:
                    self.gpu.add(src[1], src[2], ms, src[3])
                if self._stepf is not None:
                    p = self._prev[2]
                    self._stepf.write(json.dumps({"t": round(self._prev[0], 4), "ms": round(ms, 3), "nd": self._prev[1],
                        "sn": sum(n for n, _ in p), "sq": round(sum(n * (c + n / 2.0) for n, c in p), 1),
                        "dc": self._prev[3], "nch": len(p)}) + "\n")
            self._pp = self._prev; self._prev = (now, n_dec, chunks, dctx)
        except Exception:
            self.errors += 1; self._prev = None; self._pp = None

    # ---- 결정용 비용 (ms). items = [(a_j, L_j)] 앞에서 prefill 중·recompute 대기 중인 요청
    def costs(self, h_ssd, h_dram, ld_dram_ms, qr, qw, promo_jobs, items, n_dec, dec_rem=None, ahead_out=None, a0=0,
              qv=None, s_tok=1):
        """qv (v2): fs_queue.view()를 토큰 단위로 바꾼 것 {ahead_tok, last_tok(정렬), store_busy, lam}.
        s_tok: hit 뒤 나머지 프롬프트 토큰 수 (④ MIX_DIFF)."""
        S, G = self.ssd, self.gpu
        F = self.ssdf
        use_q = FIFO_SSD and F.ready
        if use_q and qv is None:                                   # 풀 계측이 없으면 이전 신호로 근사
            qv = {"ahead_tok": qr, "last_tok": [], "store_busy": int(qw > 0), "lam": 0.0}
        # recompute 쪽에서 실제로 계산하는 토큰과 시작 위치.
        # PARTIAL: CPU에 있는 앞부분(h_dram)은 load되므로 SSD 부분(h_ssd)만, 위치 a0 + h_dram부터 계산.
        h_tot = h_ssd if PARTIAL else h_ssd + h_dram
        a0 = a0 if OFFSET else 0
        a_rc = a0 + (h_dram if PARTIAL else 0)
        lin_h = G.lin(a_rc + h_tot) - G.lin(a_rc)                               # 위치 a_rc부터 h_tot 계산
        rem_tot = sum(max(0, L - a) for a, L in items)
        ovh = G.overhead(n_dec) + G.bmix                                         # prefill이 섞인 스텝의 기본 시간
        gpu_wait = G.steps(rem_tot, n_dec) * ovh + sum(G.lin(L) - G.lin(a) for a, L in items)   # 앞 prefill 소진
        if use_q:
            fnow = qv.get("store_frac", qv["store_busy"]) if F_AVG else qv["store_busy"]
            spt = F.sec_per_tok(fnow)                             # 앞으로의 읽기 속도 (store 비율에 따라)
            ssd_ms = 1000.0 * F.t_load(h_ssd, qv["ahead_tok"], fnow)
        else:
            ssd_ms = 1000.0 * S.t_load(h_ssd, qr, qw)
        est_ld_raw = max(ssd_ms, gpu_wait) + ld_dram_ms + ovh                                  # + suffix 한 스텝
        k_ld, k_rc = self.k("ld"), self.k("rc")
        est_ld = k_ld * est_ld_raw
        q_tok = qr + S.w * qw
        if OPP_IO_FUT and use_q:
            # FIFO: 진행 중 load는 내 앞이라 안 늦어짐. 내 블록이 큐에 남아 있는 동안 들어오는 load가
            # 내 남은 블록만큼 늦어진다. 앞 블록 처리 중 도착 → h 전체, 내 블록 처리 중 도착 → 평균 h/2.
            T_my = ssd_ms / 1000.0; T_ah = min(T_my, qv["ahead_tok"] * spt)
            est_opp_io = 1000.0 * qv["lam"] * (T_ah + (T_my - T_ah) / 2.0) * h_ssd * spt if h_ssd > 0 else 0.0
        else:
            if use_q:   # 이전 정의(진행 중 load가 늦어짐)를 새 SSD 모델 속도로: 상한은 읽기 큐만
                est_opp_io = 1000.0 * min(promo_jobs * h_ssd, qv["ahead_tok"]) * spt if h_ssd > 0 else 0.0
            else:
                est_opp_io = 1000.0 * min(promo_jobs * h_ssd, q_tok) * B_KV / S.bweff if h_ssd > 0 else 0.0
        est_rc_raw = G.steps(rem_tot + h_tot, n_dec) * ovh + sum(G.lin(L) - G.lin(a) for a, L in items) + lin_h
        if PARTIAL and h_dram > 0:
            est_rc_raw += ld_dram_ms                                              # CPU 앞부분 복사를 기다린 뒤 계산 시작
        est_rc = k_rc * est_rc_raw
        own = k_rc * (G.steps(h_tot, n_dec) * ovh + lin_h)
        n_ahead = sum(1 for a, L in items if L - a > 0)
        if REM and FIFO and dec_rem is not None and ahead_out is not None and len(ahead_out) == len(items):
            Qs = G.steps(rem_tot, n_dec); Ss = max(1, G.steps(h_tot, n_dec))     # 스텝 단위 (S·Q는 다른 뜻이라 이름 구분)
            ov = sum(max(0.0, min(r, Qs + Ss) - Qs) / Ss for r in dec_rem)       # 생성 중: [0, r]
            t = 0
            for (a, L), r in zip(items, ahead_out):                               # 앞 prefill: 끝난 뒤 r 스텝 생성
                t += G.steps(max(0, L - a), n_dec)
                ov += max(0.0, min(t + r, Qs + Ss) - max(t, Qs)) / Ss
            est_opp = ov * (Ss * G.bmix + lin_h)
        else:
            ov = None
            if MIX_DIFF:   # load도 나머지 프롬프트용 섞인 스텝을 만든다 → recompute로 늘어난 섞인 스텝만 b_mix
                s_ = max(1, int(s_tok))
                x_mix = max(0, G.steps(rem_tot + h_tot + s_, n_dec) - G.steps(rem_tot, n_dec) - G.steps(s_, n_dec))
            else:
                x_mix = G.steps(h_tot, n_dec)
            per = x_mix * G.bmix + lin_h                                  # 생성 요청 1개가 겪는 지연
            est_opp = (n_dec + (n_ahead if FIFO else 0)) * per
        if PF_QUEUE and use_q:
            # load 큐 순서대로 끝나는 시각 t_j = (그 job 마지막 블록까지의 토큰) × 읽기 시간.
            # ③ 창 안에 끝나는 job만: 앞 prefill 계산 중 도착 → 내 계산 전체(own), 내 계산 중 도착 → 남은 만큼.
            win = est_rc / 1000.0; t_pr = max(0.0, (est_rc - own) / 1000.0)
            soon = 0; est_opp_pf = 0.0
            for e in qv["last_tok"]:
                t = max(0.0, e) * spt
                if t > win: break
                soon += 1
                est_opp_pf += own if t <= t_pr else 1000.0 * (win - t)
            if PF_FUT and qv.get("lam", 0) > 0:
                # 시각 u에 들어온 load: 앞 큐가 비는 시각 A와 u 중 늦은 때부터 자기 블록(평균 크기 Hs)을 읽어 도착
                A = qv["ahead_tok"] * spt; Hs = qv.get("h_avg_tok", 0.0) * spt; K = 20; du = win / K
                for i in range(K):
                    land = max((i + 0.5) * du, A) + Hs
                    if land <= win:
                        p_ = qv["lam"] * du; soon += p_
                        est_opp_pf += p_ * (own if land <= t_pr else 1000.0 * (win - land))
        else:
            if use_q:   # 이전 정의(고르게 끝남)를 새 SSD 모델 속도로
                qs = qv["ahead_tok"] * spt
                soon = promo_jobs if qs <= 0 else min(promo_jobs, promo_jobs * (est_rc / 1000.0) / qs)
            else:
                q_bytes = q_tok * B_KV
                soon = promo_jobs if q_bytes <= 0 else min(promo_jobs, promo_jobs * S.bweff * (est_rc / 1000.0) / q_bytes)
            est_opp_pf = (0.0 if FIFO else sum(G.lin(min(h_tot, max(0, L - a))) for a, L in items)) + soon * own
        return dict(est_ld=est_ld, est_opp_io=est_opp_io, est_rc=est_rc, est_opp=est_opp,
                    est_opp_pf=est_opp_pf, own_rc=own, n_eff=soon, est_ld_raw=est_ld_raw, est_rc_raw=est_rc_raw,
                    k_ld=k_ld, k_rc=k_rc, ld_ssd_ms=ssd_ms, ld_gpu_wait_ms=gpu_wait, opp_overlap=ov,
                    v2_q=use_q, mix_steps=None if ov is not None else x_mix)

    def params(self):
        return {**self.ssd.params(), **self.ssdf.params(), **self.gpu.params(), "v2": V2, "fifo_ssd": FIFO_SSD,
                "io_done": IO_DONE, "partial": PARTIAL, "opp_io_fut": OPP_IO_FUT, "pf_queue": PF_QUEUE, "pf_fut": PF_FUT, "f_avg": F_AVG, "mix_diff": MIX_DIFF, "c_cfg": C_CFG,
                "budget": self.gpu.budget, "ohw_ready": self.ready, "ohw_err": self.errors, "ldexcl": LDEXCL, "n_ld_skip": self.n_ld_skip,
                "calib": CALIB, "fifo": FIFO, "rem": REM, "k_ld": round(self.k("ld"), 3), "k_rc": round(self.k("rc"), 3),
                "k_ld_obs": round(self.k_obs("ld"), 3), "k_rc_obs": round(self.k_obs("rc"), 3),
                "n_kld": len(self.kr["ld"]), "n_krc": len(self.kr["rc"]), "n_krcm": len(self.kr["rcm"]),
                "k_rc_miss": round(float(np.median(self.kr["rcm"])), 3) if len(self.kr["rcm"]) >= K_MIN else None}