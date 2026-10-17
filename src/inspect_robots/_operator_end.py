"""Process-wide ``SIGUSR1`` end-of-episode request that does not wait for the policy.

A keypress End is read between steps, so a trial that is blocked on a slow model
call first receives the reply and executes it. Remote supervisors (the rig
dashboard) can instead send ``SIGUSR1`` to the evaluator. While the rollout
thread is waiting on the policy, the handler raises ``OperatorEndRequested``
into that wait, so the reply is discarded unexecuted. At any other time the
signal only sets a flag that the rollout checks before the next policy call and
before every ``embodiment.step()``; it never interrupts motion already in flight.

The handler is installed once, on the main thread, and only when ``SIGUSR1``
still has its default disposition, so it never replaces another owner. A
supervisor can confirm installation by reading the caught-signal mask
(``SigCgt`` in ``/proc/<pid>/status``) before sending: without the handler,
``SIGUSR1`` terminates the process without parking.
"""

from __future__ import annotations

import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from types import FrameType


class OperatorEndRequested(BaseException):
    """Unwind a policy wait to end the trial; not an ``Exception``, so retries skip it."""


class _State:
    installed = False
    requested = False
    interruptible = False


def _on_main_thread() -> bool:
    return threading.current_thread() is threading.main_thread()


def _handle(signum: int, frame: FrameType | None) -> None:
    _State.requested = True
    if _State.interruptible:
        # Disarm before raising so a repeated signal cannot interrupt the unwind.
        _State.interruptible = False
        raise OperatorEndRequested()


def install() -> bool:
    """Install the handler once; return whether this process now honors the signal.

    Returns ``False`` without side effects off POSIX, off the main thread, or
    when ``SIGUSR1`` already has a non-default handler.
    """
    if _State.installed:
        return True
    usr1 = getattr(signal, "SIGUSR1", None)
    if usr1 is None or not _on_main_thread():
        return False
    if signal.getsignal(usr1) is not signal.SIG_DFL:
        return False
    signal.signal(usr1, _handle)
    _State.installed = True
    return True


def active() -> bool:
    """True when the handler is installed and the caller can receive its interrupt.

    Rollouts on worker threads (parallel simulation trials) ignore the signal
    entirely, so one trial cannot clear or consume another's request.
    """
    return _State.installed and _on_main_thread()


def begin_trial() -> None:
    """Discard a request that arrived before this trial started."""
    _State.requested = False


def check() -> None:
    """Raise ``OperatorEndRequested`` if an end request arrived; no-op when inactive."""
    if active() and _State.requested:
        raise OperatorEndRequested()


@contextmanager
def interruptible() -> Iterator[None]:
    """Let an end request abort the enclosed wait, and reject its result afterwards."""
    if not active():
        yield
        return
    # Arm before checking: a signal landing between the two then raises from
    # the handler, so no request can slip into the wait as a flag only.
    _State.interruptible = True
    try:
        check()
        yield
    finally:
        _State.interruptible = False
    if _State.requested:
        # The wait finished (or a policy swallowed the interrupt) after the
        # request arrived: its result is late and must not be executed.
        raise OperatorEndRequested()
