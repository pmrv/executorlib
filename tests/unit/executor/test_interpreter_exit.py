"""The interpreter must be able to exit when a task fails and shutdown() is never called.

Run in subprocesses: the failure mode is a deadlock in ``threading._shutdown``, which can
only be observed by letting a whole interpreter try to exit. Each subprocess runs this very
file with a scenario name; the scenarios live under ``__main__`` below.
"""

import subprocess
import sys
import unittest

from executorlib import SingleNodeExecutor

TIMEOUT = 60


def _run(scenario):
    return subprocess.run(
        [sys.executable, __file__, scenario],
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
    )


class TestInterpreterExit(unittest.TestCase):
    def test_exits_when_exception_escapes(self):
        """The traceback is printed and the interpreter then actually exits."""
        try:
            proc = _run("escaping_exception")
        except subprocess.TimeoutExpired:
            self.fail(
                "interpreter did not exit within %ds after an uncaught exception" % TIMEOUT
            )
        self.assertEqual(proc.returncode, 1)
        self.assertIn("ValueError", proc.stderr)

    def test_exits_without_shutdown(self):
        """A successful run that never calls shutdown() must still terminate."""
        try:
            proc = _run("no_shutdown")
        except subprocess.TimeoutExpired:
            self.fail("interpreter did not exit within %ds without shutdown()" % TIMEOUT)
        self.assertEqual(proc.returncode, 0)


def raise_error():
    raise ValueError("deliberate failure")


def escaping_exception():
    # No context manager and no shutdown(); the exception is deliberately not caught, so
    # it propagates out of the program exactly as it does from a console-script entry
    # point.
    exe = SingleNodeExecutor(max_workers=1)
    exe.submit(raise_error).result()
    return exe


def no_shutdown():
    # Same, but the task succeeds and shutdown() is still never called. A builtin on
    # purpose: with a function defined in this module the executor happens to be torn
    # down early enough that the hang only shows intermittently; with a builtin it
    # hangs every time on a scheduler without the exit hook.
    exe = SingleNodeExecutor(max_workers=1)
    assert exe.submit(sum, [20, 22]).result() == 42
    return exe


SCENARIOS = {"escaping_exception": escaping_exception, "no_shutdown": no_shutdown}


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] in SCENARIOS:
        # Bound at module scope on purpose: as in a real script, the executor must live
        # until interpreter finalization, so that its __del__ -> shutdown() runs there.
        # Left local to the scenario function it is collected early and the hang is missed.
        exe = SCENARIOS[sys.argv[1]]()
    else:
        unittest.main()
