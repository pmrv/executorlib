"""The interpreter must be able to exit when a task fails and shutdown() is never called.

Run in subprocesses: the failure mode is a deadlock in ``threading._shutdown``, which can
only be observed by letting a whole interpreter try to exit.
"""

import subprocess
import sys
import unittest

TIMEOUT = 60

# No context manager and no shutdown(); the exception is deliberately not caught, so it
# propagates out of the program exactly as it does from a console-script entry point.
ESCAPING_EXCEPTION = """
from executorlib import SingleNodeExecutor


def raise_error():
    raise ValueError("deliberate failure")


exe = SingleNodeExecutor(max_workers=1)
exe.submit(raise_error).result()
"""

# Same, but the task succeeds and shutdown() is still never called.
NO_SHUTDOWN = """
from executorlib import SingleNodeExecutor


def double(i):
    return 2 * i


exe = SingleNodeExecutor(max_workers=1)
assert exe.submit(double, 21).result() == 42
"""


def _run(script):
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=TIMEOUT
    )


class TestInterpreterExit(unittest.TestCase):
    def test_exits_when_exception_escapes(self):
        """The traceback is printed and the interpreter then actually exits."""
        try:
            proc = _run(ESCAPING_EXCEPTION)
        except subprocess.TimeoutExpired:
            self.fail(
                "interpreter did not exit within %ds after an uncaught exception" % TIMEOUT
            )
        self.assertEqual(proc.returncode, 1)
        self.assertIn("ValueError", proc.stderr)

    def test_exits_without_shutdown(self):
        """A successful run that never calls shutdown() must still terminate."""
        try:
            proc = _run(NO_SHUTDOWN)
        except subprocess.TimeoutExpired:
            self.fail("interpreter did not exit within %ds without shutdown()" % TIMEOUT)
        self.assertEqual(proc.returncode, 0)


if __name__ == "__main__":
    unittest.main()
