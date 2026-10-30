# SPDX-License-Identifier: Apache-2.0
"""
ProbeOffloadingConnector — experiment wrapper around vLLM's native OffloadingConnector.

Adds, at the recompute-vs-load decision point (get_num_new_matched_tokens):
  * a per-request force switch via request.kv_transfer_params["policy"] in
    {"load", "recompute"}  (absent -> default vLLM behaviour = load when hit)
  * a JSONL log line per decision with:
      h (hit tokens found in offload tiers), num_computed (GPU-HBM prefix hit),
      tier (dram / ssd / none, best effort), occ_reqs, occ_tokens, q_wait,
      q_io_reqs, q_io_tokens, policy, decision

Lines marked VERIFY reference vLLM 0.23.0 internals inferred from grep output in
this session; check them against the installed source before running.

Later, the runtime selector (§3.3 of the outline) replaces `_decide()`.
"""
from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)

LOG_PATH = os.environ.get("PROBE_LOG", "probe_log.jsonl")

# --- scheduler handle -------------------------------------------------------
# The scheduler-side connector is created inside Scheduler.__init__ but has no
# back-reference to it. We patch Scheduler.__init__ once to register itself so
# the connector can read `running` / `waiting`.  VERIFY: attribute names.
_SCHEDULER: Any = None


def _install_scheduler_hook() -> None:
    global _hooked
    try:
        from vllm.v1.core.sched.scheduler import Scheduler  # VERIFY path
    except Exception as e:  # pragma: no cover
        logger.warning("ProbeConnector: cannot import Scheduler (%s)", e)
        return
    if getattr(Scheduler, "_probe_hooked", False):
        return
    orig_init = Scheduler.__init__

    def patched(self, *a, **kw):
        orig_init(self, *a, **kw)
        global _SCHEDULER
        _SCHEDULER = self

    Scheduler.__init__ = patched  # type: ignore[assignment]
    Scheduler._probe_hooked = True  # type: ignore[attr-defined]


_install_scheduler_hook()


class ProbeOffloadingConnector(OffloadingConnector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._log = open(LOG_PATH, "a", buffering=1)
        self._step_log = open(os.environ.get("STEP_LOG", LOG_PATH.replace(".jsonl", "_steps.jsonl")), "a", buffering=1)
        # in-flight async loads issued by us: req_id -> tokens
        self._inflight: dict[str, int] = {}
        # req_id -> number of None ("ask again later") returns seen before a decision
        self._none_count: dict[str, int] = {}
        # req_ids that queued a secondary->primary promotion during any lookup call
        self._promoted: set[str] = set()

    # ------------------------------------------------------------------ util
    def _io_state(self) -> dict[str, int]:
        """In-flight secondary-tier jobs from the tiering manager, both directions.
        promotions = SSD->CPU loads, cascades = CPU->SSD stores. Blocks = sum of keys.
        Zero for CPUOffloadingSpec (no secondary tiers)."""
        out = {"io_promo_jobs": 0, "io_promo_blocks": 0, "io_store_jobs": 0, "io_store_blocks": 0}
        mgr = getattr(self.connector_scheduler, "manager", None)
        jobs = getattr(mgr, "_transfer_jobs", None)
        if not jobs:
            return out
        for jm in list(jobs.values()):
            n = len(getattr(jm, "keys", ()) or ())
            if getattr(jm, "is_promotion", False):
                out["io_promo_jobs"] += 1; out["io_promo_blocks"] += n
            else:
                out["io_store_jobs"] += 1; out["io_store_blocks"] += n
        return out

    def _sched_state(self) -> dict[str, int]:
        s = _SCHEDULER
        if s is None:
            return {"occ_reqs": -1, "occ_tokens": -1, "q_wait": -1}
        running = getattr(s, "running", [])            # VERIFY
        waiting = getattr(s, "waiting", [])            # VERIFY (may be a RequestQueue)
        occ_tokens = 0
        for r in running:
            occ_tokens += getattr(r, "num_computed_tokens", 0)
        return {
            "occ_reqs": len(running),
            "occ_tokens": occ_tokens,
            "q_wait": len(waiting),
        }

    def _policy(self, request: "Request") -> str | None:
        p = getattr(request, "kv_transfer_params", None) or {}   # VERIFY attr name
        return p.get("policy")

    def _pending_snapshot(self) -> set[tuple[int, str]]:
        """{(id(tier), ctx_key)} of all promotions currently queued in the tiering
        manager. Empty for CPUOffloadingSpec (no _pending_load_submissions)."""
        mgr = getattr(self.connector_scheduler, "manager", None)          # confirmed: scheduler.py:264
        pend = getattr(mgr, "_pending_load_submissions", None)
        if not pend:
            return set()
        return {(id(tier), k) for tier, d in pend.items() for k in d}

    def _purge_new_promotions(self, new_keys: set[tuple[int, str]]) -> int:
        """Drop promotions queued by the parent call we are overriding, and release
        the CPU slots promote() allocated eagerly (prepare_write, ref_cnt=-1) via the
        manager's own failure path: primary_tier.complete_write(keys, ctx, False).
        Returns number of blocks released."""
        released = 0
        try:
            mgr = self.connector_scheduler.manager
            pend = mgr._pending_load_submissions
            for tier, d in list(pend.items()):
                for k in list(d):
                    if (id(tier), k) in new_keys:
                        entry = d.pop(k)
                        mgr.primary_tier.complete_write(entry.keys, entry.req_context, False)
                        released += len(entry.keys)
                if not d:
                    pend.pop(tier, None)
        except Exception as e:  # never break scheduling because of experiment code
            logger.warning("purge_new_promotions failed: %s", e)
        return released

    def _decide(self, policy: str | None, h: int) -> int:
        """Returns tokens to load. Experiment: forced policy only.
        Later: replace with cost-model selector using self._sched_state()."""
        if policy == "recompute":
            return 0
        return h

    # ---------------------------------------------------- decision point
    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        policy = self._policy(request)

        if policy == "recompute":
            # Bypass the parent's _lookup(): in the tiering manager, lookup() itself
            # queues the SSD->CPU promotion (and returns None until it completes),
            # so any decision made after the parent call comes after the load.
            # Mirror the parent's bookkeeping (offloading/scheduler.py:559-571)
            # minus _lookup/_touch so the store path still works at request end.
            t0 = time.perf_counter()
            st = self.connector_scheduler._req_status[request.request_id]
            for gs in st.group_states:
                gs.block_ids.clear()
            st.update_offload_keys()
            st.num_locally_computed_tokens = num_computed_tokens
            st.update_num_hit_blocks(num_computed_tokens)
            self._emit(request, policy, "recompute", h=-1, num_computed=num_computed_tokens,
                       tier="skipped", promo=0, released=0,
                       lookup_us=0.0, decide_us=(time.perf_counter() - t0) * 1e6)
            return 0, False

        before = self._pending_snapshot()
        t0 = time.perf_counter()
        h, is_async = super().get_num_new_matched_tokens(request, num_computed_tokens)
        t1 = time.perf_counter()

        new_keys = self._pending_snapshot() - before
        if new_keys:
            self._promoted.add(request.request_id)

        if h is None:               # async lookup / promotion in progress -> asked again later
            self._none_count[request.request_id] = self._none_count.get(request.request_id, 0) + 1
            self._emit(request, policy, "pending", h=None, num_computed=num_computed_tokens,
                       tier="ssd" if new_keys else "?", promo=len(new_keys), released=0,
                       lookup_us=(t1 - t0) * 1e6, decide_us=0.0)
            return None, is_async

        # ssd if this request ever queued a promotion (usually during an earlier
        # pending call); dram if the blocks were already resident in the CPU tier.
        tier = "none" if h == 0 else ("ssd" if request.request_id in self._promoted else "dram")
        load = self._decide(policy, h)
        released = 0

        if load == 0 and h > 0:      # selector-driven recompute (not forced): roll back
            st = self.connector_scheduler._req_status[request.request_id]
            st.update_num_hit_blocks(num_computed_tokens)
            released = self._purge_new_promotions(new_keys)
            is_async = False

        if load > 0 and is_async:
            self._inflight[request.request_id] = load

        t2 = time.perf_counter()
        self._emit(request, policy,
                   "load" if load > 0 else ("recompute" if h > 0 else "miss"),
                   h=h, num_computed=num_computed_tokens, tier=tier, promo=len(new_keys),
                   released=released, lookup_us=(t1 - t0) * 1e6, decide_us=(t2 - t1) * 1e6)
        return load, is_async

    def _emit(self, request, policy, decision, *, h, num_computed, tier, promo, released,
              lookup_us, decide_us, extra=None):
        rid = request.request_id
        rec = {
            "ts": time.time(),
            "req_id": rid,
            "policy": policy,
            "decision": decision,          # load / recompute / miss / pending
            "h": h,                        # -1 when lookup was bypassed (forced recompute)
            "num_computed": num_computed,
            "prompt_len": getattr(request, "num_prompt_tokens", -1),
            "tier": tier,                  # dram / ssd / none / skipped / ?
            "promo_jobs": promo,           # (tier, request) promotion jobs queued by this lookup
            "released_blocks": released,   # CPU slots freed on rollback
            "pending_calls": self._none_count.get(rid, 0),  # how many None returns preceded
            "q_io_reqs": len(self._inflight),
            "q_io_tokens": sum(self._inflight.values()),
            "lookup_us": round(lookup_us, 1),   # parent lookup (tier probing) cost
            "decide_us": round(decide_us, 1),   # our decision + bookkeeping cost
        }
        rec.update(self._sched_state())
        rec.update(self._io_state())
        if extra: rec.update(extra)
        self._log.write(json.dumps(rec) + "\n")
        if decision != "pending":
            self._none_count.pop(rid, None)
            self._promoted.discard(rid)

    # ---------------------------------------------- scheduler step timing
    # build_connector_meta is called once per scheduler step with the SchedulerOutput.
    # Logging its wall-clock gives the step period (~TBT for in-flight decodes) plus
    # how many prefill tokens the step carried -> direct Opp measurement without the
    # client-side proxy: compare step periods during a probe vs. around it.
    def build_connector_meta(self, scheduler_output: "SchedulerOutput"):
        now = time.time()
        try:
            nst = getattr(scheduler_output, "num_scheduled_tokens", {}) or {}
            n_req = len(nst); n_tok = sum(nst.values())
            n_new = len(getattr(scheduler_output, "scheduled_new_reqs", []) or [])
        except Exception:
            n_req = n_tok = n_new = -1
        last = getattr(self, "_last_step_ts", None)
        self._last_step_ts = now
        if last is not None and (n_req or n_new):
            self._step_log.write(json.dumps({"ts": now, "step_ms": round((now - last) * 1000, 2),
                                             "n_req": n_req, "n_tok": n_tok, "n_new": n_new}) + "\n")
        return super().build_connector_meta(scheduler_output)

    # ---------------------------------------------- q_io bookkeeping
    def update_connector_output(self, connector_output: "KVConnectorOutput"):
        super().update_connector_output(connector_output)
        done = getattr(connector_output, "finished_recving", None)   # VERIFY name/type
        if done:
            for rid in done:
                self._inflight.pop(rid, None)

    def request_finished(self, request: "Request", block_ids: list[int]):
        self._inflight.pop(request.request_id, None)
        return super().request_finished(request, block_ids)