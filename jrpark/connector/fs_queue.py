# SPDX-License-Identifier: Apache-2.0
"""fs_queue — fs tier 스레드 풀(DualQueueThreadPool) 계측. 결정은 바꾸지 않는다.

vLLM fs tier 구조 (vllm/v1/kv_offload/tiering/fs/thread_pool.py):
  * load 큐와 store 큐가 따로 있고, 둘 다 블록 단위 FIFO (deque, popleft).
  * load 우선 스레드(n_read)는 load 큐를 먼저, store 우선 스레드(n_write)는 store 큐를 먼저 비운다.
  → 내 load 블록은 이미 load 큐에 있는 블록 뒤에 선다. store는 앞에 서지 않고, store 큐가 차 있는
    동안 SSD를 나눠 써서 읽기 속도를 낮춘다.

여기서 하는 일:
  * job 제출 시각, 블록 수, 제출 순간 load 큐에 앞서 있던 블록 수 기록 (enqueue_load 감쌈)
  * job 완료 시각 기록 (워커가 _finished_q에 넣는 순간, 실제 I/O 완료)
  * 스케줄러 스텝마다 store 큐가 차 있었는지 세어 job별 'store가 걸려 있던 비율' 계산
  * 결정 시점의 load 큐 순서 보기 (job별 마지막 블록 위치)
  * load job 도착률 (최근 LAM_WIN 초)
connector 모듈은 EngineCore 프로세스에서 import 되므로 fs 매니저·스레드 풀과 같은 프로세스에서 동작한다.
"""
from __future__ import annotations

import collections, json, os, threading, time

LAM_WIN = float(os.environ.get("OHW_LAM_WIN", "30"))     # load 도착률 창 (초)
F_WIN = float(os.environ.get("OHW_F_WIN", "10"))          # store 비율 평균 창 (초)
IO_LOG = os.environ.get("OHW_IO_LOG", "")                # job별 기록 (분석용, 결정 불변)

POOLS: list = []                 # 계측된 DualQueueThreadPool 인스턴스
JOBS: dict = {}                  # job_id -> {t_sub, n, ahead, st0, steps, st_steps, t_done, rid}
_DONE: dict = {}                 # job_id -> 완료 시각 (워커 스레드가 기록)
# 결정마다 부르는 lam()·h_avg()·store_frac()이 가볍도록 창 안 값만 남기고 합계를 따로 유지한다
_SUB = collections.deque()      # 최근 LAM_WIN 초의 load job (제출 시각, 블록 수)
_sub_blk = 0; _T0 = None        # 창 안 블록 합, 첫 제출 시각
_BUSY = collections.deque()     # 최근 F_WIN 초의 (시각, store 큐가 차 있었나) 스텝마다
_busy_sum = 0
_STEP = 0; _VCACHE = (-1, None) # 큐 보기 결과는 스케줄러 스텝마다 한 번만 계산
_KEY2RID: dict = {}              # 첫 OffloadKey -> request id (기록용)
_JOB2KEY: dict = {}              # job_id -> 첫 OffloadKey
_lock = threading.Lock()
_logf = open(IO_LOG, "a", buffering=1) if IO_LOG else None
installed = False


class _TimedDeque(collections.deque):
    """워커가 (job_id, success)를 넣는 순간을 완료 시각으로 남긴다."""
    def append(self, x):
        try:
            _DONE[x[0]] = time.time()
        except Exception:
            pass
        super().append(x)


def install() -> bool:
    """DualQueueThreadPool과 fs 매니저의 submit_load를 감싼다. 여러 번 불러도 한 번만 적용."""
    global installed
    if installed:
        return True
    try:
        from vllm.v1.kv_offload.tiering.fs import thread_pool as _tp
        import vllm.v1.kv_offload.tiering.fs.manager as _fsm
    except Exception:
        return False
    P = _tp.DualQueueThreadPool
    _init, _enq = P.__init__, P.enqueue_load

    def init(self, *a, **kw):
        _init(self, *a, **kw)
        self._finished_q = _TimedDeque(self._finished_q)
        POOLS.append(self)

    def enqueue_load(self, job_id, n_tasks, tasks):
        global _sub_blk, _T0
        with self._condition:
            ahead = len(self._load_q); st = len(self._store_q)
        now = time.time()
        with _lock:
            JOBS[job_id] = {"t_sub": now, "n": int(n_tasks), "ahead": ahead, "st0": int(st > 0),
                            "steps": 0, "st_steps": 0, "t_done": None,
                            "rid": _KEY2RID.pop(_JOB2KEY.pop(job_id, None), None)}
            _SUB.append((now, int(n_tasks))); _sub_blk += int(n_tasks)
            if _T0 is None: _T0 = now
        return _enq(self, job_id, n_tasks, tasks)

    P.__init__ = init
    P.enqueue_load = enqueue_load
    # fs 매니저의 submit_load: job_id와 첫 key를 이어 요청 id를 찾는다 (기록용)
    for name in dir(_fsm):
        C = getattr(_fsm, name)
        if isinstance(C, type) and "submit_load" in C.__dict__ and C.__module__ == _fsm.__name__:
            _sub = C.submit_load

            def submit_load(self, job_metadata, _sub=_sub):
                try:
                    keys = getattr(job_metadata, "keys", None)
                    if keys: _JOB2KEY[job_metadata.job_id] = keys[0]
                except Exception:
                    pass
                return _sub(self, job_metadata)
            C.submit_load = submit_load
    installed = True
    return True


def map_request(rid, first_keys) -> None:
    """load로 결정한 요청의 pending 항목 첫 key들을 기록 (job과 요청을 잇는 데만 씀)."""
    for k in first_keys:
        try: _KEY2RID[k] = rid
        except Exception: pass
    if len(_KEY2RID) > 20000: _KEY2RID.clear()


def _store_busy() -> int:
    return int(any(len(p._store_q) > 0 for p in POOLS))


def tick() -> list:
    """스케줄러 스텝마다 호출. 진행 중 job의 store 비율을 세고, 끝난 job 표본을 돌려준다.
    반환: [(n_blocks, ahead_blocks, f_store, T_sec, rid, t_done)]"""
    if not POOLS:
        return []
    global _busy_sum, _STEP
    busy = _store_busy(); out = []; now = time.time()
    _BUSY.append((now, busy)); _busy_sum += busy
    while _BUSY and now - _BUSY[0][0] > F_WIN:
        _busy_sum -= _BUSY.popleft()[1]
    _STEP += 1
    with _lock:
        for jid, j in list(JOBS.items()):
            td = _DONE.pop(jid, None)
            if td is None:
                j["steps"] += 1; j["st_steps"] += busy
                continue
            f = j["st_steps"] / j["steps"] if j["steps"] else float(j["st0"])
            out.append((j["n"], j["ahead"], f, td - j["t_sub"], j["rid"], td))
            if _logf is not None:
                try:
                    _logf.write(json.dumps({"job": str(jid), "rid": j["rid"], "n": j["n"], "ahead": j["ahead"],
                                            "f_store": round(f, 3), "t_sub": round(j["t_sub"], 4),
                                            "t_done": round(td, 4)}) + "\n")
                except Exception:
                    pass
            del JOBS[jid]
    return out


def _trim_sub(now):
    global _sub_blk
    while _SUB and now - _SUB[0][0] > LAM_WIN:
        _sub_blk -= _SUB.popleft()[1]


def lam(now=None) -> float:
    """최근 LAM_WIN 초의 load job 도착률 (job/s)."""
    now = now or time.time()
    with _lock:
        _trim_sub(now); n = len(_SUB); first = _T0 or now
    return n / max(1.0, min(LAM_WIN, now - first))


def store_frac(now=None) -> float:
    """최근 F_WIN 초 동안 스텝마다 본 'store 큐가 차 있음'의 비율."""
    return _busy_sum / len(_BUSY) if _BUSY else float(_store_busy())


def h_avg(now=None) -> float:
    """최근 LAM_WIN 초에 들어온 load job의 평균 블록 수."""
    now = now or time.time()
    with _lock:
        _trim_sub(now)
        return _sub_blk / len(_SUB) if _SUB else 0.0


def view(pending_other_blocks=()) -> dict | None:
    """결정 시점의 load 큐. 앞에서부터 job별 마지막 블록 위치(0부터)를 돌려준다.
    pending_other_blocks: 아직 큐에 안 들어간(같은 스케줄 패스에서 먼저 load로 정해진) job의 블록 수들.
    이들은 flush 때 큐 뒤에 이 순서대로 붙고, 이 요청의 블록은 그 뒤에 선다."""
    global _VCACHE
    if not POOLS:
        return None
    if _VCACHE[0] == _STEP:                                    # 같은 스텝의 다른 결정이 이미 계산함
        pos0, last0 = _VCACHE[1]
    else:
        ends, pos0 = {}, 0
        for p in POOLS:
            with p._condition:
                ids = [s.job_id for _, s in p._load_q]
            for jid in ids:
                ends[jid] = pos0; pos0 += 1
        queued = set(ends)
        with _lock:
            in_service = [jid for jid in JOBS if jid not in queued and jid not in _DONE]
        last0 = sorted(ends.values()) + [-1] * len(in_service)  # 남은 블록이 없으면 곧 끝남
        _VCACHE = (_STEP, (pos0, last0))
    pos, last = pos0, list(last0)
    for n in pending_other_blocks:
        pos += int(n); last.append(pos - 1)
    return {"ahead": pos, "last": sorted(last), "store_busy": _store_busy()}