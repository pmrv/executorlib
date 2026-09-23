"""The interpreter-exit hook must not deadlock a later shutdown()."""

import unittest

from executorlib import SingleNodeExecutor
from executorlib.task_scheduler.base import _python_exit, _task_scheduler_set


class TestShutdownAtInterpreterExit(unittest.TestCase):
    def test_shutdown_after_hook_does_not_block(self):
        """A shutdown() following the hook must be a no-op, not a deadlock.

        The hook stops the task threads. If it did not also mark the scheduler shut
        down, this shutdown() would enqueue a second shutdown message that no thread
        is left to acknowledge, and its queue.join() would block forever -- which is
        the original bug by a new route. __del__ calls shutdown(), so this is the
        ordinary path at interpreter exit, not a contrived one.
        """
        exe = SingleNodeExecutor(max_workers=1)
        self.assertEqual(exe.submit(sum, [1, 2, 3]).result(), 6)
        scheduler = exe._task_scheduler if hasattr(exe, "_task_scheduler") else exe
        scheduler._shutdown_at_interpreter_exit()
        scheduler.shutdown(wait=True)  # must return promptly
        scheduler.shutdown(wait=True)  # and stay idempotent

    def test_hook_is_safe_with_no_schedulers(self):
        """The hook must tolerate an empty or already-cleaned registry."""
        _task_scheduler_set.clear()
        _python_exit()


if __name__ == "__main__":
    unittest.main()
