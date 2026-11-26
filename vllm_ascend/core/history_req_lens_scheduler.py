#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Waiting-queue scheduling policy driven by historical decode lengths.

Ported from the SGLang history-req-lens feature (MindSpeed_for_lingqu PR#6):
during RL rollout the same prompt is sampled repeatedly (rollout.n paths and
cross-step dataset reuse), so the observed decode length of a finished request
is a good predictor for the next request with the same prompt.

The strategy lives entirely inside the waiting queue:
- on ``pop_request``, requests are dispatched longest-predicted-first (LJF)
  so long decoders start early and converge in parallel with short ones;
- requests without history keep FCFS order at the tail of the ordering;
- preempted / partially executed requests always resume first.

Decode-length records are backfilled when requests finish, via a per-scheduler
``update_from_output`` wrapper installed by :func:`install_history_req_lens_waiting_queue`.
"""

from __future__ import annotations

import time
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING

from vllm.logger import logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.request_queue import RequestQueue, SchedulingPolicy
from vllm.v1.request import Request, RequestStatus

if TYPE_CHECKING:
    from vllm.v1.core.sched.scheduler import Scheduler


def prompt_key(prompt_token_ids) -> int:
    """Hash the exact prompt token sequence (same key semantics as the SGLang implementation)."""
    return hash(tuple(prompt_token_ids))


class HistoryReqLensDB:
    """LRU store of historical decode lengths, keyed by prompt hash.

    Each prompt keeps the last ``window_size`` observed output lengths and
    predicts the decode length as their rolling mean. The store is bounded by
    ``max_size`` entries (least-recently-used eviction).
    """

    RECORDED_IDS_LIMIT_MULTIPLIER = 8

    def __init__(self, max_size: int = 1024, window_size: int = 6) -> None:
        self.max_size = max_size
        self.window_size = window_size
        self._data: OrderedDict[int, list[int]] = OrderedDict()
        # Guard against double-recording: ``finished_req_ids`` persists until
        # the scheduler frees finished states on the next schedule() call.
        self._recorded_request_ids: OrderedDict[str, None] = OrderedDict()
        self.record_count = 0
        self.hit_count = 0
        self.miss_count = 0

    def record(self, key: int, output_len: int) -> None:
        if key in self._data:
            self._data.move_to_end(key)
            lens = self._data[key]
            if len(lens) >= self.window_size:
                lens.pop(0)
            lens.append(output_len)
        else:
            while len(self._data) >= self.max_size:
                self._data.popitem(last=False)
            self._data[key] = [output_len]
        self.record_count += 1

    def record_request(self, request_id: str, request: Request) -> None:
        if request_id in self._recorded_request_ids:
            return
        self._recorded_request_ids[request_id] = None
        limit = self.max_size * self.RECORDED_IDS_LIMIT_MULTIPLIER
        while len(self._recorded_request_ids) > limit:
            self._recorded_request_ids.popitem(last=False)
        output_len = getattr(request, "num_output_tokens", None)
        if output_len is None:
            output_len = len(getattr(request, "output_token_ids", ()) or ())
        self.record(prompt_key(request.prompt_token_ids), int(output_len))

    def predict(self, key: int) -> float | None:
        lens = self._data.get(key)
        if not lens:
            return None
        return sum(lens) / len(lens)


class HistoryReqLensRequestQueue(RequestQueue):
    """Waiting queue ordered by predicted decode length (longest first).

    Implements the standard vLLM ``RequestQueue`` contract. Selection is
    O(waiting_size) per pop: the request with the highest predicted decode
    length wins; ties and history misses keep FCFS order (stable selection).
    ``peek_request`` pins the selected request so the following ``pop_request``
    removes the same request even if the queue changed in between.
    """

    STATS_LOG_INTERVAL_S = 5.0

    def __init__(
        self,
        db: HistoryReqLensDB,
        immediate_predicate: Callable[[Request], bool] | None = None,
    ) -> None:
        self.db = db
        self.immediate_predicate = immediate_predicate
        # Preempted / partially executed requests resume ahead of fresh work.
        self._immediate_queue: deque[Request] = deque()
        # Fresh requests in FCFS arrival order; pop selects by prediction.
        self._waiting: list[Request] = []
        self._pinned_request_id: str | None = None
        self._last_stats_log_at = time.monotonic()

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _select_best_request(self) -> Request | None:
        if self._immediate_queue:
            return self._immediate_queue[0]
        if not self._waiting:
            return None
        best_request: Request | None = None
        best_pred = -1.0
        for request in self._waiting:
            pred = self.db.predict(prompt_key(request.prompt_token_ids))
            if pred is None:
                pred = 0.0
            # Strictly greater keeps the earliest request on ties (FCFS).
            if pred > best_pred:
                best_request = request
                best_pred = pred
        return best_request

    def _count_hit_or_miss(self, request: Request) -> None:
        """Count exactly once per dispatched request (pop semantics, not scans)."""
        if self.db.predict(prompt_key(request.prompt_token_ids)) is None:
            self.db.miss_count += 1
        else:
            self.db.hit_count += 1

    def _remove_from_waiting(self, request: Request) -> None:
        for index, candidate in enumerate(self._waiting):
            if candidate.request_id == request.request_id:
                self._waiting.pop(index)
                return

    def _remove_from_immediate(self, request_id: str) -> bool:
        for index, candidate in enumerate(self._immediate_queue):
            if candidate.request_id == request_id:
                del self._immediate_queue[index]
                return True
        return False

    def _pop_specific(self, request: Request) -> None:
        if not self._remove_from_immediate(request.request_id):
            self._remove_from_waiting(request)

    def _find_request(self, request_id: str) -> Request | None:
        for request in self._immediate_queue:
            if request.request_id == request_id:
                return request
        for request in self._waiting:
            if request.request_id == request_id:
                return request
        return None

    def _maybe_log_stats(self) -> None:
        now = time.monotonic()
        if (now - self._last_stats_log_at) < self.STATS_LOG_INTERVAL_S:
            return
        self._last_stats_log_at = now
        logger.info(
            "HistoryReqLens stats: db_size=%d window_size=%d "
            "queue=(immediate=%d waiting=%d) "
            "pop_hits=%d pop_misses=%d records=%d",
            len(self.db._data),
            self.db.window_size,
            len(self._immediate_queue),
            len(self._waiting),
            self.db.hit_count,
            self.db.miss_count,
            self.db.record_count,
        )

    # ------------------------------------------------------------------
    # RequestQueue contract
    # ------------------------------------------------------------------

    def add_request(self, request: Request) -> None:
        self._pinned_request_id = None
        if self.immediate_predicate is not None and self.immediate_predicate(request):
            self._immediate_queue.append(request)
        else:
            self._waiting.append(request)
        self._maybe_log_stats()

    def pop_request(self) -> Request:
        pinned_id = self._pinned_request_id
        if pinned_id is not None:
            request = self._find_request(pinned_id)
            if request is not None:
                self._count_hit_or_miss(request)
                self._pop_specific(request)
                self._pinned_request_id = None
                self._maybe_log_stats()
                return request
            self._pinned_request_id = None
        request = self._select_best_request()
        if request is None:
            raise IndexError("pop from empty HistoryReqLens queue")
        self._count_hit_or_miss(request)
        self._pop_specific(request)
        self._pinned_request_id = None
        self._maybe_log_stats()
        return request

    def peek_request(self) -> Request:
        request = self._select_best_request()
        if request is None:
            raise IndexError("peek from an empty HistoryReqLens queue")
        self._pinned_request_id = request.request_id
        return request

    def prepend_request(self, request: Request) -> None:
        self._pinned_request_id = None
        if self.immediate_predicate is not None and self.immediate_predicate(request):
            self._immediate_queue.appendleft(request)
        else:
            self._waiting.insert(0, request)
        self._maybe_log_stats()

    def prepend_requests(self, requests: RequestQueue) -> None:
        for request in requests:
            self.prepend_request(request)

    def remove_request(self, request: Request) -> None:
        self._pinned_request_id = None
        self._remove_request_by_id(request.request_id)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        self._pinned_request_id = None
        for request in requests:
            self._remove_request_by_id(request.request_id)

    def _remove_request_by_id(self, request_id: str) -> None:
        if not self._remove_from_immediate(request_id):
            self._remove_from_waiting_by_id(request_id)

    def _remove_from_waiting_by_id(self, request_id: str) -> None:
        for index, candidate in enumerate(self._waiting):
            if candidate.request_id == request_id:
                self._waiting.pop(index)
                return

    def __bool__(self) -> bool:
        return len(self) > 0

    def __len__(self) -> int:
        return len(self._immediate_queue) + len(self._waiting)

    def __iter__(self) -> Iterator[Request]:
        yield from self._immediate_queue
        yield from self._waiting

    def __contains__(self, request: object) -> bool:
        request_id = getattr(request, "request_id", None)
        return request_id is not None and self._find_request(request_id) is not None


def is_recovery_request(request: Request) -> bool:
    """Return whether a request must resume ahead of fresh waiting work."""
    return (
        request.status == RequestStatus.PREEMPTED
        or getattr(request, "num_computed_tokens", 0) > 0
        or getattr(request, "num_output_tokens", 0) > 0
    )


def _install_record_hook(scheduler: Scheduler, db: HistoryReqLensDB) -> None:
    """Wrap the scheduler's ``update_from_output`` to backfill decode lengths.

    Instance-level wrapping (instead of class-level patching) covers both the
    async path and the DP BalanceScheduler path (which overrides ``schedule``
    but inherits ``update_from_output``). Finished Request objects are still
    alive after ``update_from_output`` returns; the scheduler frees them on
    the next ``schedule()`` call. Recording failures must never break
    inference, hence the catch-all.
    """
    original_update_from_output = scheduler.update_from_output

    def update_from_output_with_record(*args, **kwargs):
        outputs = original_update_from_output(*args, **kwargs)
        try:
            for request_id in scheduler.finished_req_ids:
                request = scheduler.requests.get(request_id)
                if request is not None:
                    db.record_request(request_id, request)
        except Exception:
            logger.exception("HistoryReqLens decode-length backfill failed; ignoring.")
        return outputs

    scheduler.update_from_output = update_from_output_with_record


def install_history_req_lens_waiting_queue(
    scheduler: Scheduler,
    *,
    max_db_size: int,
    window_size: int,
) -> HistoryReqLensRequestQueue:
    """Install the HistoryReqLens queue on a newly constructed FCFS scheduler."""
    policy = scheduler.policy
    if policy != SchedulingPolicy.FCFS:
        raise ValueError(f"HistoryReqLens requires FCFS scheduling policy, got {policy!s}")

    if getattr(scheduler, "_history_req_lens_db", None) is not None:
        return scheduler.waiting

    waiting = scheduler.waiting
    if isinstance(waiting, HistoryReqLensRequestQueue):
        return waiting
    if waiting:
        raise RuntimeError("HistoryReqLens waiting queue must be installed before request admission.")

    db = HistoryReqLensDB(max_size=max_db_size, window_size=window_size)
    queue = HistoryReqLensRequestQueue(db=db, immediate_predicate=is_recovery_request)
    scheduler._history_req_lens_db = db
    scheduler.waiting = queue
    _install_record_hook(scheduler, db)
    logger.info(
        "HistoryReqLens waiting queue installed: max_db_size=%d, window_size=%d",
        max_db_size,
        window_size,
    )
    return queue


class HistoryReqLensAsyncScheduler(AsyncScheduler):
    """Async scheduler that installs the HistoryReqLens waiting queue."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        from vllm_ascend.ascend_config import init_ascend_config

        history_req_lens_config = init_ascend_config(self.vllm_config).scheduler_config.history_req_lens_config
        if history_req_lens_config.enabled:
            install_history_req_lens_waiting_queue(
                self,
                max_db_size=history_req_lens_config.max_db_size,
                window_size=history_req_lens_config.window_size,
            )
