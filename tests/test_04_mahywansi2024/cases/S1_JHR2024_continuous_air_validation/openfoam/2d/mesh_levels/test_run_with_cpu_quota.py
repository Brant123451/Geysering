#!/usr/bin/env python3
"""No-OpenFOAM self-tests for the Case-3 CPU quota wrapper."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
WRAPPER = HERE / "run_with_cpu_quota.py"


@unittest.skipUnless(hasattr(os, "sched_getaffinity"), "Linux affinity is required")
class QuotaWrapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cpu = min(os.sched_getaffinity(0))

    def wrapper_command(self, *child: str, **options: float) -> list[str]:
        quota = options.get("quota", 20.0)
        period = options.get("period", 0.1)
        timeout = options.get("timeout", 3.0)
        grace = options.get("grace", 0.05)
        return [
            sys.executable,
            str(WRAPPER),
            "--cpu",
            str(self.cpu),
            "--quota-percent",
            str(quota),
            "--period-seconds",
            str(period),
            "--timeout-seconds",
            str(timeout),
            "--grace-seconds",
            str(grace),
            "--",
            *child,
        ]

    def test_kernel_affinity_nice_log_and_child_exit_code(self) -> None:
        payload = (
            "import json,os; "
            "print(json.dumps({'affinity':sorted(os.sched_getaffinity(0)),"
            "'nice':os.getpriority(os.PRIO_PROCESS,0)})); "
            "raise SystemExit(7)"
        )
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "child.log"
            command = self.wrapper_command(sys.executable, "-c", payload, quota=100.0)
            command[2:2] = ["--log", str(log)]
            completed = subprocess.run(command, check=False)
            self.assertEqual(completed.returncode, 7)
            lines = log.read_text(encoding="utf-8").splitlines()
            data = json.loads(next(line for line in lines if line.startswith("{")))
            self.assertEqual(data["affinity"], [self.cpu])
            self.assertEqual(data["nice"], 19)
            self.assertTrue(any("wrapper_returncode=7" in line for line in lines))

    def test_default_style_twenty_percent_duty_cycle_is_real(self) -> None:
        payload = (
            "import time; start=time.process_time(); "
            "\nwhile time.process_time()-start < 0.08: pass"
        )
        started = time.monotonic()
        completed = subprocess.run(
            self.wrapper_command(sys.executable, "-c", payload),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        elapsed = time.monotonic() - started
        self.assertEqual(completed.returncode, 0)
        self.assertGreaterEqual(elapsed, 0.25)

    def test_timeout_returns_124_and_terminates_group(self) -> None:
        completed = subprocess.run(
            self.wrapper_command(
                sys.executable,
                "-c",
                "import time; time.sleep(10)",
                timeout=0.2,
            ),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        self.assertEqual(completed.returncode, 124)

    def test_wrapper_signal_is_forwarded_with_shell_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "signal.log"
            command = self.wrapper_command(
                sys.executable,
                "-c",
                "import time; time.sleep(10)",
            )
            command[2:2] = ["--log", str(log)]
            process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            ready_deadline = time.monotonic() + 2.0
            while time.monotonic() < ready_deadline:
                if log.is_file() and "[case3-cpu-quota] start" in log.read_text(
                    encoding="utf-8"
                ):
                    break
                time.sleep(0.02)
            else:
                self.fail("quota wrapper did not reach its signal-ready state")
            process.send_signal(signal.SIGTERM)
            self.assertEqual(process.wait(timeout=3.0), 128 + signal.SIGTERM)


if __name__ == "__main__":
    unittest.main(verbosity=2)
