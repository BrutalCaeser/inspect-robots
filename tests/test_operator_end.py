"""Signalled End (SIGUSR1): interrupt a blocked policy and never execute its late reply."""

from __future__ import annotations

import os
import signal
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from inspect_robots import _operator_end
from inspect_robots.approver import AutoApprover
from inspect_robots.controller import DefaultController
from inspect_robots.logging.sink import NullSink
from inspect_robots.mock import CubePickEmbodiment, ScriptedPolicy
from inspect_robots.rollout import TrialRecord, rollout
from inspect_robots.scene import Scene
from inspect_robots.types import OPERATOR_END, Action, ActionChunk, Observation, StepResult

pytestmark = pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="POSIX signals only")

_SCENE = Scene(id="s", instruction="reach", init_seed=0)


@pytest.fixture
def end_signal() -> Iterator[None]:
    """Install the handler for one test, then restore the process-wide state."""
    previous = signal.getsignal(signal.SIGUSR1)
    saved = (_operator_end._State.installed, _operator_end._State.requested)
    signal.signal(signal.SIGUSR1, signal.SIG_DFL)
    _operator_end._State.installed = False
    _operator_end._State.requested = False
    assert _operator_end.install()
    try:
        yield
    finally:
        signal.signal(signal.SIGUSR1, previous)
        _operator_end._State.installed, _operator_end._State.requested = saved
        _operator_end._State.interruptible = False


def _send_end() -> None:
    os.kill(os.getpid(), signal.SIGUSR1)


def _send_end_when(event: threading.Event) -> threading.Thread:
    """Send End from another thread once ``event`` is set, as the dashboard would."""

    def run() -> None:
        assert event.wait(5)
        _send_end()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


class _BlockingPolicy:
    """Scripted chunks, but the given inference waits for ``release`` like a slow model call."""

    def __init__(self, *, block_on: int = 1, swallow: bool = False) -> None:
        self._inner = ScriptedPolicy(chunk_size=3)
        self.info = self._inner.info
        self.config = self._inner.config
        self.block_on = block_on
        self.swallow = swallow
        self.calls = 0
        self.waiting = threading.Event()
        self.release = threading.Event()
        self.replies = 0

    def reset(self, scene: Scene) -> None:
        self._inner.reset(scene)

    def act(self, observation: Observation) -> ActionChunk:
        self.calls += 1
        if self.calls == self.block_on:
            self.waiting.set()
            try:
                # Short sleeps keep the wait interruptible on every platform.
                deadline = time.monotonic() + 5
                while not self.release.is_set() and time.monotonic() < deadline:
                    time.sleep(0.005)
            except BaseException:
                if not self.swallow:
                    raise
                # A policy that swallows the interrupt still returns a reply,
                # which the rollout must reject as late.
        self.replies += 1
        return self._inner.act(observation)


class _CountingEmbodiment(CubePickEmbodiment):
    def __init__(self, on_step: Callable[[int], None] | None = None) -> None:
        super().__init__()
        self.executed = 0
        self._on_step = on_step

    def step(self, action: Action) -> StepResult:
        self.executed += 1
        if self._on_step is not None:
            self._on_step(self.executed)
        return super().step(action)


def _run(policy: Any, embodiment: Any, max_steps: int = 20) -> TrialRecord:
    return rollout(
        policy,
        embodiment,
        _SCENE,
        max_steps=max_steps,
        seed=0,
        epoch=0,
        controller=DefaultController(),
        approver=AutoApprover(),
        sink=NullSink(),
    )


@pytest.mark.usefixtures("end_signal")
def test_end_during_blocked_model_call_executes_nothing() -> None:
    policy = _BlockingPolicy()
    embodiment = _CountingEmbodiment()
    sender = _send_end_when(policy.waiting)

    record = _run(policy, embodiment)
    sender.join(5)

    assert record.terminated
    assert record.termination_reason == OPERATOR_END
    assert record.metadata["operator_end_source"] == "signal"
    assert record.status != "cancelled"
    assert policy.replies == 0
    assert embodiment.executed == 0
    assert record.steps == []


@pytest.mark.usefixtures("end_signal")
def test_late_reply_from_policy_that_swallows_interrupt_is_discarded() -> None:
    policy = _BlockingPolicy(swallow=True)
    embodiment = _CountingEmbodiment()
    sender = _send_end_when(policy.waiting)

    record = _run(policy, embodiment)
    sender.join(5)

    assert policy.replies == 1
    assert embodiment.executed == 0
    assert record.termination_reason == OPERATOR_END


@pytest.mark.usefixtures("end_signal")
def test_end_mid_trial_skips_rest_of_chunk_and_next_call() -> None:
    policy = _BlockingPolicy(block_on=2)
    embodiment = _CountingEmbodiment()
    sender = _send_end_when(policy.waiting)

    record = _run(policy, embodiment)
    sender.join(5)

    # The first chunk (3 steps) ran before End; the blocked second call is discarded.
    assert embodiment.executed == 3
    assert len(record.steps) == 3
    assert policy.replies == 1
    assert record.termination_reason == OPERATOR_END


@pytest.mark.usefixtures("end_signal")
def test_end_during_motion_finishes_that_step_then_stops() -> None:
    policy = _BlockingPolicy(block_on=0)
    embodiment = _CountingEmbodiment(on_step=lambda n: _send_end() if n == 1 else None)

    record = _run(policy, embodiment)

    # Motion in flight is never interrupted; the remaining buffered actions are skipped.
    assert embodiment.executed == 1
    assert len(record.steps) == 1
    assert record.termination_reason == OPERATOR_END


@pytest.mark.usefixtures("end_signal")
def test_request_before_trial_is_discarded() -> None:
    _send_end()
    assert _operator_end._State.requested
    policy = _BlockingPolicy(block_on=0)
    embodiment = _CountingEmbodiment()

    record = _run(policy, embodiment, max_steps=2)

    assert embodiment.executed == 2
    assert record.termination_reason == "max_steps"


@pytest.mark.usefixtures("end_signal")
def test_worker_thread_rollout_ignores_signal_state() -> None:
    _operator_end._State.requested = True
    results: list[TrialRecord] = []

    def run() -> None:
        results.append(_run(_BlockingPolicy(block_on=0), _CountingEmbodiment(), max_steps=2))

    worker = threading.Thread(target=run)
    worker.start()
    worker.join(10)

    assert results[0].termination_reason == "max_steps"
    # The worker neither consumed nor cleared the main thread's request.
    assert _operator_end._State.requested


def test_without_handler_end_helpers_are_inert() -> None:
    saved = _operator_end._State.installed
    _operator_end._State.installed = False
    try:
        assert not _operator_end.active()
        _operator_end.check()
        with _operator_end.interruptible():
            pass
    finally:
        _operator_end._State.installed = saved


@pytest.mark.usefixtures("end_signal")
def test_install_is_idempotent() -> None:
    assert _operator_end.install()
    assert signal.getsignal(signal.SIGUSR1) is _operator_end._handle


def test_install_refuses_existing_handler_off_main_thread_and_without_sigusr1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_operator_end._State, "installed", False)
    previous = signal.getsignal(signal.SIGUSR1)

    def other(signum: int, frame: object) -> None:
        return None

    signal.signal(signal.SIGUSR1, other)
    try:
        assert not _operator_end.install()
        assert signal.getsignal(signal.SIGUSR1) is other
    finally:
        signal.signal(signal.SIGUSR1, previous)

    outcome: list[bool] = []
    worker = threading.Thread(target=lambda: outcome.append(_operator_end.install()))
    worker.start()
    worker.join(5)
    assert outcome == [False]

    monkeypatch.delattr(signal, "SIGUSR1")
    assert not _operator_end.install()
    assert not _operator_end._State.installed
