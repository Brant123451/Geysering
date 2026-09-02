#!/usr/bin/env python3
"""Run one Linux command under a real single-CPU duty-cycle quota.

The Case-3 host has neither a writable cgroup v2 hierarchy nor ``cpulimit``.
This wrapper therefore enforces the declared CPU budget itself:

* the child starts a new process group;
* the whole group inherits one Linux CPU affinity and nice level 19;
* SIGSTOP/SIGCONT duty cycling limits the group to 20 percent by default;
* timeout and wrapper signals resume and terminate the entire group cleanly.

The wrapper is intentionally Linux-only.  It exits closed when affinity,
process-group signalling, or the requested quota cannot be established.
"""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import BinaryIO, Sequence


TIMEOUT_EXIT = 124
INTERNAL_EXIT = 125


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not (parsed > 0.0):
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _normalise_returncode(returncode: int) -> int:
    if returncode >= 0:
        return returncode
    return 128 + min(-returncode, 127)


class QuotaRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.process: subprocess.Popen[bytes] | None = None
        self.stopped = False
        self.requested_signal: int | None = None
        self.log_handle: BinaryIO | None = None

    def _status(self, message: str) -> None:
        line = f"[case3-cpu-quota] {message}\n".encode("utf-8", errors="replace")
        if self.log_handle is not None:
            self.log_handle.write(line)
            self.log_handle.flush()
        else:
            sys.stderr.buffer.write(line)
            sys.stderr.buffer.flush()

    def _handle_signal(self, signum: int, _frame: object) -> None:
        if self.requested_signal is None:
            self.requested_signal = signum

    def _kill_group(self, signum: int) -> None:
        if self.process is None:
            return
        try:
            os.killpg(self.process.pid, signum)
        except ProcessLookupError:
            pass

    def _resume(self) -> None:
        if self.stopped:
            self._kill_group(signal.SIGCONT)
            self.stopped = False

    def _wait_for_exit(self, seconds: float) -> int | None:
        assert self.process is not None
        deadline = time.monotonic() + max(seconds, 0.0)
        while True:
            returncode = self.process.poll()
            if returncode is not None:
                return returncode
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return None
            time.sleep(min(0.05, remaining))

    def _terminate_group(self, initial_signal: int, reason: str) -> int | None:
        assert self.process is not None
        self._resume()
        self._status(f"terminating process group: {reason}; signal={initial_signal}")
        self._kill_group(initial_signal)
        returncode = self._wait_for_exit(self.args.grace_seconds)
        if returncode is not None:
            return returncode

        self._kill_group(signal.SIGTERM)
        returncode = self._wait_for_exit(min(5.0, max(self.args.grace_seconds, 0.25)))
        if returncode is not None:
            return returncode

        self._kill_group(signal.SIGKILL)
        return self._wait_for_exit(5.0)

    def _child_setup(self) -> None:
        # This runs in the child immediately before exec.  Both settings are
        # inherited by ordinary descendants; the new session below supplies a
        # private process group for duty-cycle signals.
        os.sched_setaffinity(0, {self.args.cpu})
        os.setpriority(os.PRIO_PROCESS, 0, 19)

    def _open_log(self) -> None:
        if self.args.log is None:
            return
        log_path = Path(self.args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = log_path.open("wb", buffering=0)

    def _start(self) -> None:
        self._open_log()
        command_text = shlex.join(self.args.command)
        self._status(
            "start "
            f"cpu={self.args.cpu} nice=19 quota={self.args.quota_percent:g}% "
            f"period={self.args.period_seconds:g}s timeout={self.args.timeout_seconds:g}s "
            f"command={command_text}"
        )
        stdout = self.log_handle if self.log_handle is not None else None
        stderr: int | BinaryIO | None = subprocess.STDOUT if stdout is not None else None
        self.process = subprocess.Popen(
            self.args.command,
            stdin=None,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            preexec_fn=self._child_setup,
        )

        # Verify the kernel state instead of trusting an environment marker.
        # A very short command may already have exited; its pre-exec setup still
        # had to succeed or Popen itself would have raised.
        if self.process.poll() is None:
            affinity = os.sched_getaffinity(self.process.pid)
            nice_value = os.getpriority(os.PRIO_PROCESS, self.process.pid)
            if affinity != {self.args.cpu} or nice_value != 19:
                raise RuntimeError(
                    "kernel did not establish the requested CPU affinity/nice level: "
                    f"affinity={sorted(affinity)}, nice={nice_value}"
                )

    def run(self) -> int:
        allowed = os.sched_getaffinity(0)
        if self.args.cpu not in allowed:
            raise RuntimeError(
                f"CPU {self.args.cpu} is unavailable; allowed CPUs are {sorted(allowed)}"
            )

        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(signum, self._handle_signal)

        self._start()
        assert self.process is not None

        run_slice = self.args.period_seconds * self.args.quota_percent / 100.0
        stop_slice = self.args.period_seconds - run_slice
        deadline = time.monotonic() + self.args.timeout_seconds

        try:
            while True:
                if self.requested_signal is not None:
                    signum = self.requested_signal
                    self._terminate_group(signum, "wrapper received a signal")
                    return 128 + signum

                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    self._terminate_group(signal.SIGINT, "wall timeout")
                    return TIMEOUT_EXIT

                try:
                    returncode = self.process.wait(timeout=min(run_slice, remaining))
                except subprocess.TimeoutExpired:
                    returncode = None
                if returncode is not None:
                    normalised = _normalise_returncode(returncode)
                    self._status(f"exit child_returncode={returncode} wrapper_returncode={normalised}")
                    return normalised

                if self.requested_signal is not None:
                    continue
                if time.monotonic() >= deadline:
                    continue

                self._kill_group(signal.SIGSTOP)
                self.stopped = True
                stop_deadline = min(deadline, time.monotonic() + stop_slice)
                while time.monotonic() < stop_deadline and self.requested_signal is None:
                    time.sleep(min(0.05, stop_deadline - time.monotonic()))
                self._resume()
        except BaseException:
            self._terminate_group(signal.SIGTERM, "wrapper internal failure")
            raise
        finally:
            self._resume()
            if self.log_handle is not None:
                self.log_handle.close()
                self.log_handle = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one command on one CPU at a real duty-cycle quota."
    )
    parser.add_argument("--cpu", required=True, type=_nonnegative_int)
    parser.add_argument("--quota-percent", type=_positive_float, default=20.0)
    parser.add_argument("--period-seconds", type=_positive_float, default=0.5)
    parser.add_argument("--timeout-seconds", required=True, type=_positive_float)
    parser.add_argument("--grace-seconds", type=_nonnegative_float, default=60.0)
    parser.add_argument("--log", help="combined child stdout/stderr log (truncated)")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    if args.quota_percent > 100.0:
        parser.error("--quota-percent cannot exceed 100")
    if args.period_seconds * args.quota_percent / 100.0 < 0.005:
        parser.error("the running slice must be at least 0.005 seconds")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        return QuotaRunner(args).run()
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"run_with_cpu_quota.py: {exc}", file=sys.stderr)
        return INTERNAL_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
