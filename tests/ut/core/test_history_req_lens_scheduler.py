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
"""Tests for the HistoryReqLens waiting queue (LJF by historical decode lengths)."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.request import RequestStatus

from vllm_ascend.core.history_req_lens_scheduler import (
    HistoryReqLensAsyncScheduler,
    HistoryReqLensDB,
    HistoryReqLensRequestQueue,
    install_history_req_lens_waiting_queue,
    prompt_key,
)


def make_scheduler(policy: SchedulingPolicy = SchedulingPolicy.FCFS):
    return SimpleNamespace(
        policy=policy,
        waiting=create_request_queue(policy),
        finished_req_ids=set(),
        requests={},
    )


def make_request(
    request_id: str,
    prompt_token_ids: list[int],
    *,
    status: RequestStatus = RequestStatus.WAITING,
    num_computed_tokens: int = 0,
    num_output_tokens: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        num_computed_tokens=num_computed_tokens,
        num_output_tokens=num_output_tokens,
        status=status,
    )


def seed_history(db: HistoryReqLensDB, prompt_token_ids: list[int], output_lens: list[int]) -> None:
    for length in output_lens:
        db.record(prompt_key(prompt_token_ids), length)


# ---------------------------------------------------------------------------
# DB behavior
# ---------------------------------------------------------------------------


def test_db_predict_returns_rolling_mean():
    db = HistoryReqLensDB(max_size=16, window_size=6)
    key = prompt_key([1, 2, 3])

    for length in (100, 200, 300):
        db.record(key, length)

    assert db.predict(key) == pytest.approx(200.0)


def test_db_respects_window_size():
    db = HistoryReqLensDB(max_size=16, window_size=2)
    key = prompt_key([1, 2, 3])

    for length in (100, 200, 400):
        db.record(key, length)

    assert db.predict(key) == pytest.approx(300.0)


def test_db_lru_eviction():
    db = HistoryReqLensDB(max_size=2, window_size=6)
    key_a = prompt_key([1])
    key_b = prompt_key([2])
    key_c = prompt_key([3])

    db.record(key_a, 10)
    db.record(key_b, 20)
    assert db.predict(key_a) is not None

    db.record(key_a, 11)  # refresh a -> b becomes LRU
    db.record(key_c, 30)  # evicts b

    assert db.predict(key_b) is None
    assert db.predict(key_a) is not None
    assert db.predict(key_c) is not None


# ---------------------------------------------------------------------------
# Queue ordering behavior
# ---------------------------------------------------------------------------


def test_pop_orders_by_predicted_decode_length():
    db = HistoryReqLensDB()
    scheduler = make_scheduler()
    queue = install_history_req_lens_waiting_queue(scheduler, max_db_size=64, window_size=6)

    short_prompt = [10, 11]
    long_prompt = [20, 21]
    fresh_prompt = [30, 31]
    seed_history(db, short_prompt, [100])
    seed_history(db, long_prompt, [4000])

    requests = [
        make_request("short", short_prompt),
        make_request("long", long_prompt),
        make_request("fresh", fresh_prompt),
    ]
    for request in requests:
        queue.add_request(request)

    assert queue.pop_request() is requests[1]  # predicted 4000 first
    assert queue.pop_request() is requests[0]  # predicted 100 second
    assert queue.pop_request() is requests[2]  # no history keeps FCFS tail


def test_no_history_degrades_to_fcfs():
    db = HistoryReqLensDB()
    scheduler = make_scheduler()
    queue = install_history_req_lens_waiting_queue(scheduler, max_db_size=64, window_size=6)

    requests = [make_request(f"r{i}", [i]) for i in range(3)]
    for request in requests:
        queue.add_request(request)

    for request in requests:
        assert queue.pop_request() is request


def test_ties_keep_fcfs_order():
    db = HistoryReqLensDB()
    scheduler = make_scheduler()
    queue = install_history_req_lens_waiting_queue(scheduler, max_db_size=64, window_size=6)

    prompt_a = [1]
    prompt_b = [2]
    seed_history(db, prompt_a, [500])
    seed_history(db, prompt_b, [500])

    first = make_request("a", prompt_a)
    second = make_request("b", prompt_b)
    queue.add_request(first)
    queue.add_request(second)

    assert queue.pop_request() is first  # equal prediction -> arrival order


def test_peek_pin_then_pop_returns_same_request():
    db = HistoryReqLensDB()
    scheduler = make_scheduler()
    queue = install_history_req_lens_waiting_queue(scheduler, max_db_size=64, window_size=6)

    long_prompt = [9]
    seed_history(db, long_prompt, [3000])
    target = make_request("target", long_prompt)
    other = make_request("other", [8])
    queue.add_request(other)
    queue.add_request(target)

    assert queue.peek_request() is target
    assert queue.pop_request() is target
    assert queue.pop_request() is other


def test_recovery_requests_pop_first():
    db = HistoryReqLensDB()
    scheduler = make_scheduler()
    queue = install_history_req_lens_waiting_queue(scheduler, max_db_size=64, window_size=6)

    long_prompt = [7]
    seed_history(db, long_prompt, [5000])
    fresh_long = make_request("fresh-long", long_prompt)
    recovery = make_request("recovery", [6], num_computed_tokens=10)
    queue.add_request(fresh_long)
    queue.add_request(recovery)

    assert queue.pop_request() is recovery
    assert queue.pop_request() is fresh_long


def test_queue_contract_methods():
    db = HistoryReqLensDB()
    scheduler = make_scheduler()
    queue = install_history_req_lens_waiting_queue(scheduler, max_db_size=64, window_size=6)

    request_a = make_request("a", [1])
    request_b = make_request("b", [2])
    queue.add_request(request_a)
    queue.add_request(request_b)

    assert len(queue) == 2
    assert bool(queue)
    assert request_a in queue
    assert request_b in queue
    assert list(queue) == [request_a, request_b]

    queue.remove_request(request_a)
    assert request_a not in queue
    assert len(queue) == 1

    queue.prepend_request(request_a)  # back at the front of the FCFS list
    assert list(queue) == [request_a, request_b]


# ---------------------------------------------------------------------------
# Record backfill hook
# ---------------------------------------------------------------------------


def test_record_hook_backfills_finished_requests():
    scheduler = make_scheduler()
    db = install_history_req_lens_waiting_queue(scheduler, max_db_size=64, window_size=6).db

    finished = make_request("done", [1, 2, 3], num_output_tokens=3000)
    scheduler.requests["done"] = finished
    scheduler.finished_req_ids = {"done"}

    recorded = []
    scheduler.update_from_output = lambda *args, **kwargs: recorded.append(args)
    scheduler._history_req_lens_db = db
    from vllm_ascend.core.history_req_lens_scheduler import _install_record_hook

    _install_record_hook(scheduler, db)
    scheduler.update_from_output(SimpleNamespace(), SimpleNamespace())

    assert db.predict(prompt_key([1, 2, 3])) == pytest.approx(3000.0)


def test_record_hook_never_double_records():
    scheduler = make_scheduler()
    db = install_history_req_lens_waiting_queue(scheduler, max_db_size=64, window_size=6).db

    finished = make_request("done", [4], num_output_tokens=100)
    scheduler.requests["done"] = finished
    scheduler.finished_req_ids = {"done"}

    from vllm_ascend.core.history_req_lens_scheduler import _install_record_hook

    _install_record_hook(scheduler, db)
    scheduler.update_from_output(SimpleNamespace(), SimpleNamespace())
    scheduler.update_from_output(SimpleNamespace(), SimpleNamespace())

    assert db.record_count == 1


def test_record_hook_swallows_errors():
    scheduler = make_scheduler()
    db = install_history_req_lens_waiting_queue(scheduler, max_db_size=64, window_size=6).db
    from vllm_ascend.core.history_req_lens_scheduler import _install_record_hook

    _install_record_hook(scheduler, db)
    # requests access raising must not break the wrapped call
    scheduler.finished_req_ids = {"gone"}
    scheduler.requests = None
    assert scheduler.update_from_output(SimpleNamespace(), SimpleNamespace()) is not None


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------


def test_install_is_idempotent():
    scheduler = make_scheduler()

    queue = install_history_req_lens_waiting_queue(scheduler, max_db_size=64, window_size=6)

    assert (
        install_history_req_lens_waiting_queue(scheduler, max_db_size=128, window_size=3)
        is queue
    )


def test_install_rejects_non_fcfs_scheduler():
    scheduler = make_scheduler(SchedulingPolicy.PRIORITY)

    with pytest.raises(ValueError, match="requires FCFS"):
        install_history_req_lens_waiting_queue(scheduler, max_db_size=64, window_size=6)


def test_install_rejects_after_request_admission():
    scheduler = make_scheduler()
    scheduler.waiting.add_request(make_request("waiting", [1]))

    with pytest.raises(RuntimeError, match="before request admission"):
        install_history_req_lens_waiting_queue(scheduler, max_db_size=64, window_size=6)


@pytest.mark.parametrize("enabled", [False, True])
def test_async_scheduler_only_installs_queue_when_enabled(enabled):
    vllm_config = SimpleNamespace()
    ascend_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            history_req_lens_config=SimpleNamespace(
                enabled=enabled,
                max_db_size=64,
                window_size=6,
            )
        )
    )

    def initialize_async_scheduler(scheduler, *args, **kwargs):
        scheduler.vllm_config = kwargs["vllm_config"]
        scheduler.policy = SchedulingPolicy.FCFS
        scheduler.waiting = create_request_queue(SchedulingPolicy.FCFS)
        scheduler.finished_req_ids = set()
        scheduler.requests = {}

    with (
        patch.object(AsyncScheduler, "__init__", initialize_async_scheduler),
        patch("vllm_ascend.ascend_config.init_ascend_config", return_value=ascend_config),
    ):
        scheduler = HistoryReqLensAsyncScheduler(vllm_config=vllm_config)

    assert isinstance(scheduler.waiting, HistoryReqLensRequestQueue) is enabled
