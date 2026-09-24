import atexit
import contextlib
import queue
import time
import weakref
from concurrent.futures import (
    Executor as FutureExecutor,
)
from concurrent.futures import (
    Future,
)
from threading import Thread
from typing import Callable, Optional, Union

from executorlib.standalone.inputcheck import check_resource_dict
from executorlib.standalone.queue import cancel_items_in_queue
from executorlib.standalone.serialize import cloudpickle_register

# How long the interpreter-exit hook waits for the workers to go away. The hook runs while
# the interpreter is shutting down, so it must be bounded: blocking here would recreate the
# very deadlock the daemon threads were meant to prevent.
_SHUTDOWN_AT_EXIT_TIMEOUT: float = 5.0

# Task schedulers which may still have running workers. Weak, like the standard library's
# ``concurrent.futures.thread._threads_queues``: a scheduler which has been garbage
# collected has already run ``__del__`` -> ``shutdown(wait=False)``, so it needs nothing
# from us.
_task_scheduler_set: "weakref.WeakSet" = weakref.WeakSet()


def _python_exit() -> None:
    """Tell the workers of every live task scheduler to shut down.

    The task scheduler threads are daemon threads, so the interpreter can exit without
    them -- but a worker *process* which is still running would simply be abandoned, and
    under a batch scheduler an orphaned worker keeps the whole allocation alive. So before
    the interpreter goes, send each scheduler the same shutdown message ``shutdown()``
    sends, which is what makes the worker processes exit on their own.

    Registered with the public ``atexit.register``. CPython runs ``atexit`` callbacks
    *after* ``threading._shutdown()`` has joined every non-daemon thread, so this only
    runs at all because the task scheduler threads are daemon threads: were any of them
    non-daemon again, the interpreter would block in that join first and never get here.
    ``concurrent.futures.thread`` uses the private ``threading._register_atexit`` because
    its worker threads are non-daemon; ours are not, so the public hook is enough.
    """
    for task_scheduler in list(_task_scheduler_set):
        with contextlib.suppress(Exception):
            task_scheduler._shutdown_at_interpreter_exit()


atexit.register(_python_exit)


def validate_resource_dict(resource_dict: dict):
    """
    No-op resource dict validator used as the default when no validation is required.

    Args:
        resource_dict (dict): Dictionary of resource requirements (ignored).
    """
    pass


class TaskSchedulerBase(FutureExecutor):
    """
    Base class for the executor.

    Args:
        max_cores (int): defines the number cores which can be used in parallel
    """

    def __init__(
        self,
        max_cores: Optional[int] = None,
        validator: Callable = validate_resource_dict,
    ):
        """
        Initialize the TaskSchedulerBase.

        Args:
            max_cores (int, optional): Maximum number of cores available to the scheduler.
                Tasks requesting more cores than this will be rejected. Defaults to None (unlimited).
            validator (Callable): Function used to validate per-task resource dicts before
                submission. Defaults to the no-op validate_resource_dict.
        """
        cloudpickle_register(ind=3)
        self._process_kwargs: dict = {}
        self._max_cores = max_cores
        self._future_queue: Optional[queue.Queue] = queue.Queue()
        self._process: Optional[Union[Thread, list[Thread]]] = None
        self._validator = validator
        _task_scheduler_set.add(self)

    @property
    def max_workers(self) -> Optional[int]:
        """
        Return the configured number of parallel workers, or None if unconstrained.

        Returns:
            Optional[int]: The max_workers value stored in process kwargs, or None.
        """
        return self._process_kwargs.get("max_workers")

    @max_workers.setter
    def max_workers(self, max_workers: int):
        """
        Setting max_workers after construction is not supported by the base scheduler.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("The max_workers setter is not implemented.")

    @property
    def info(self) -> Optional[dict]:
        """
        Get the information about the executor.

        Returns:
            Optional[dict]: Information about the executor.
        """
        meta_data_dict = self._process_kwargs.copy()
        if "future_queue" in meta_data_dict:
            del meta_data_dict["future_queue"]
        if self._process is not None and isinstance(self._process, list):
            meta_data_dict["max_workers"] = len(self._process)
            return meta_data_dict
        elif self._process is not None:
            return meta_data_dict
        else:
            return None

    @property
    def future_queue(self) -> Optional[queue.Queue]:
        """
        Get the future queue.

        Returns:
            queue.Queue: The future queue.
        """
        return self._future_queue

    def batched(
        self,
        iterable: list[Future],
        n: int,
    ) -> list[Future]:
        """
        Batch futures from the iterable into tuples of length n. The last batch may be shorter than n.

        Args:
            iterable (list): list of future objects to batch based on which future objects finish first
            n (int): badge size

        Returns:
            list[Future]: list of future objects one for each batch
        """
        raise NotImplementedError("The batched method is not implemented.")

    def submit(  # type: ignore
        self,
        fn: Callable,
        /,
        *args,
        resource_dict: Optional[dict] = None,
        **kwargs,
    ) -> Future:
        """
        Submits a callable to be executed with the given arguments.

        Schedules the callable to be executed as fn(*args, **kwargs) and returns
        a Future instance representing the execution of the callable.

        Args:
            fn (callable): function to submit for execution
            args: arguments for the submitted function
            kwargs: keyword arguments for the submitted function
            resource_dict (dict): A dictionary of resources required by the task. With the following keys:
                              - cores (int): number of MPI cores to be used for each function call
                              - threads_per_core (int): number of OpenMP threads to be used for each function call
                              - gpus_per_core (int): number of GPUs per worker - defaults to 0
                              - cwd (str/None): current working directory where the parallel python task is executed
                              - openmpi_oversubscribe (bool): adds the `--oversubscribe` command line flag (OpenMPI and
                                                              SLURM only) - default False
                              - slurm_cmd_args (list): Additional command line arguments for the srun call (SLURM only)
                              - error_log_file (str): Name of the error log file to use for storing exceptions raised
                                                      by the Python functions submitted to the Executor.

        Returns:
            Future: A Future representing the given call.
        """
        if resource_dict is None:
            resource_dict = {}
        self._validator(resource_dict=resource_dict)
        cores = resource_dict.get("cores")
        if (
            cores is not None
            and self._max_cores is not None
            and cores > self._max_cores
        ):
            raise ValueError(
                "The specified number of cores is larger than the available number of cores."
            )
        check_resource_dict(function=fn)
        f: Future = Future()
        if self._future_queue is not None:
            self._future_queue.put(
                {
                    "fn": fn,
                    "args": args,
                    "kwargs": kwargs,
                    "future": f,
                    "resource_dict": resource_dict,
                }
            )
        return f

    def map(
        self,
        fn: Callable,
        *iterables,
        timeout: Optional[float] = None,
        chunksize: int = 1,
    ):
        """Returns an iterator equivalent to map(fn, iter).

        Args:
            fn: A callable that will take as many arguments as there are
                passed iterables.
            timeout: The maximum number of seconds to wait. If None, then there
                is no limit on the wait time.
            chunksize: The size of the chunks the iterable will be broken into
                before being passed to a child process. This argument is only
                used by ProcessPoolExecutor; it is ignored by
                ThreadPoolExecutor.

        Returns:
            An iterator equivalent to: map(func, *iterables) but the calls may
            be evaluated out-of-order.

        Raises:
            TimeoutError: If the entire result iterator could not be generated
                before the given timeout.
            Exception: If fn(*args) raises for any values.
        """
        if isinstance(iterables, (list, tuple)) and any(
            isinstance(i, Future) for i in iterables
        ):
            iterables = tuple(
                i.result() if isinstance(i, Future) else i for i in iterables
            )

        return super().map(fn, *iterables, timeout=timeout, chunksize=chunksize)

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False):
        """
        Clean-up the resources associated with the Executor.

        It is safe to call this method several times. Otherwise, no other
        methods can be called after this one.

        Args:
            wait (bool): If True then shutdown will not return until all running
                futures have finished executing and the resources used by the
                parallel_executors have been reclaimed.
            cancel_futures (bool): If True then shutdown will cancel all pending
                futures. Futures that are completed or running will not be
                cancelled.
        """
        if cancel_futures and self._future_queue is not None:
            cancel_items_in_queue(que=self._future_queue)
        if self._process is not None and self._future_queue is not None:
            self._future_queue.put(
                {"shutdown": True, "wait": wait, "cancel_futures": cancel_futures}
            )
            if isinstance(self._process, Thread):
                self._process.join()
                self._future_queue.join()
        self._process = None
        self._future_queue = None

    def _set_process(self, process: Thread):
        """
        Set the process for the executor.

        Args:
            process (RaisingThread): The process for the executor.
        """
        self._process = process
        self._process.start()

    def __len__(self) -> int:
        """
        Get the length of the executor.

        Returns:
            int: The length of the executor.
        """
        queue_size = 0
        if self._future_queue is not None:
            queue_size = self._future_queue.qsize()
        return queue_size

    def _shutdown_at_interpreter_exit(
        self, timeout: float = _SHUTDOWN_AT_EXIT_TIMEOUT
    ) -> None:
        """Best-effort shutdown from the interpreter-exit hook.

        Deliberately not ``shutdown()``: that joins unconditionally, and an unbounded join
        while the interpreter is tearing down would hang the process. Here the workers are
        *told* to stop and then given a bounded grace period; if they do not manage it the
        daemon threads are abandoned, which is no worse than not trying.

        Args:
            timeout (float): total seconds to wait for all task threads to finish.
        """
        future_queue = self._future_queue
        process = self._process
        if future_queue is None or process is None:
            return
        process_lst = process if isinstance(process, list) else [process]
        # One message per task thread -- each consumes exactly one and stops.
        for _ in process_lst:
            with contextlib.suppress(Exception):
                future_queue.put({"shutdown": True, "wait": False})
        deadline = time.monotonic() + timeout
        for process_instance in process_lst:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            with contextlib.suppress(Exception):
                process_instance.join(timeout=remaining)
        # Mark the scheduler shut down, exactly as ``shutdown()`` does. Without this a
        # later ``shutdown()`` -- in particular the one ``__del__`` runs when the
        # interpreter clears module globals -- would enqueue a second shutdown message
        # that no task thread is left to acknowledge, and its ``queue.join()`` would then
        # block forever. That reintroduces the original hang by a new route.
        self._process = None
        self._future_queue = None

    def __del__(self):
        """
        Clean-up the resources associated with the Executor.
        """
        with contextlib.suppress(AttributeError, RuntimeError):
            self.shutdown(wait=False)
