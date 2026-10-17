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
refusal caused by another owner, or one a supervisor asked about, warns.

Without this handler ``SIGUSR1`` either terminates the process unparked or
reaches some other owner that will not end the trial, so a supervisor must
verify this specific handler before sending. The caught-signal mask (``SigCgt``
in ``/proc/<pid>/status``) cannot tell the two apart. Instead, a supervisor sets
``INSPECT_ROBOTS_END_SIGNAL_READY_FILE`` to a per-launch path when it starts the
evaluator. After installing, the process writes its PID there; it deletes the
file again if it later finds another handler has replaced this one. Send
``SIGUSR1`` only when the file exists and names the evaluator's PID.
"""

from __future__ import annotations

import os
import signal
import threading
import warnings
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from types import FrameType

READY_FILE_ENV = "INSPECT_ROBOTS_END_SIGNAL_READY_FILE"


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


def _refuse(reason: str, *, always: bool) -> bool:
    if always or os.environ.get(READY_FILE_ENV):
        warnings.warn(
            f"signalled End (SIGUSR1) unavailable: {reason}; End waits for the policy",
            RuntimeWarning,
            stacklevel=3,
        )
    return False


def _write_ready() -> None:
    path = os.environ.get(READY_FILE_ENV)
    if not path:
        return
    try:
        partial = f"{path}.{os.getpid()}.tmp"
        with open(partial, "w", encoding="utf-8") as handle:
            handle.write(f"{os.getpid()}\n")
        os.replace(partial, path)
    except OSError as exc:
        warnings.warn(
            f"signalled End installed but its ready file {path!r} could not be written "
            f"({exc}); a supervisor will treat End as unsupported",
            RuntimeWarning,
            stacklevel=3,
        )


def _retract_ready() -> None:
    path = os.environ.get(READY_FILE_ENV)
    if not path:
        return
    # Delete only a file this process wrote, never one a later launch reused.
    owned = False
    with suppress(OSError), open(path, encoding="utf-8") as handle:
        owned = handle.read().strip() == str(os.getpid())
    if owned:
        with suppress(OSError):
            os.remove(path)


def install() -> bool:
    """Install the handler once; return whether this process now honors the signal.

    Returns ``False`` without installing off POSIX, off the main thread, or when
    ``SIGUSR1`` already has a non-default handler (which also warns). On success,
    writes the PID to the ``READY_FILE_ENV`` path when a supervisor set one.
    """
    if _State.installed:
        return True
    usr1 = getattr(signal, "SIGUSR1", None)
    if usr1 is None:
        return _refuse("this platform has no SIGUSR1", always=False)
    if not _on_main_thread():
        return _refuse("not installed from the main thread", always=False)
    if signal.getsignal(usr1) is not signal.SIG_DFL:
        return _refuse("another handler already owns SIGUSR1", always=True)
    signal.signal(usr1, _handle)
    _State.installed = True
    _write_ready()
    return True


def active() -> bool:
    """True when this handler still owns ``SIGUSR1`` and the caller can be interrupted.

    Rollouts on worker threads (parallel simulation trials) ignore the signal
    entirely, so one trial cannot clear or consume another's request. If some
    other component replaced the handler after installation, End is withdrawn:
    the ready file is deleted and a warning is issued once.
    """
    if not _State.installed or not _on_main_thread():
        return False
    if signal.getsignal(signal.SIGUSR1) is not _handle:
        _State.installed = False
        _retract_ready()
        warnings.warn(
            "signalled End (SIGUSR1) withdrawn: another handler replaced it; "
            "End waits for the policy",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    return True


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
