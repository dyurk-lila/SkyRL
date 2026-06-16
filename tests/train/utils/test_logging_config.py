"""Regression tests for Ray worker logging configuration.

Guard against a re-entrant deadlock between stdlib ``logging`` and ``loguru``.
``configure_ray_worker_logging`` installs an ``_InterceptHandler`` as the stdlib
logging root handler that forwards every stdlib record into loguru.

The production deadlock (observed via py-spy on a frozen RL run) required the
loguru sink to use ``enqueue=True``:

  1. The ``loguru-writer`` thread drains a record and calls ``sink.write`` to
     stderr.
  2. Code on that write path emits a stdlib log (e.g. an ``info`` from a wandb
     artifact download during reward verification).
  3. The stdlib record is forwarded by ``_InterceptHandler`` back into
     ``logger.log`` -> ``Handler.emit`` -> ``self._queue.put(record)`` -- now run
     *from the writer thread itself*.
  4. ``put`` blocks on the queue pipe, so the writer never drains again while
     stranding the stdlib logging lock; every other thread that logs hangs.

The fix is to configure the sink with ``enqueue=False``: records are written
inline on the calling thread, so there is no background writer thread to re-enter
via the intercept handler and no cross-thread ``queue.put`` to block on. A nested
stdlib log from inside a sink is handled by loguru's own non-reentrant
``_protected_lock`` (it raises/skips) rather than deadlocking.
"""

import logging
import threading

from loguru import logger

from skyrl.train.utils.utils import configure_ray_worker_logging


def _reset_logging():
    logger.remove()
    for handler in list(logging.root.handlers):
        logging.root.removeHandler(handler)


def test_configured_loguru_sink_does_not_use_a_background_writer_thread():
    """The configured stderr sink must write inline (no enqueue writer thread).

    A background ``loguru-writer`` thread is the necessary condition for the
    production deadlock: it is the thread that re-enters the intercept handler
    and blocks on ``queue.put``. Asserting no such thread is spawned pins the
    fix to its root cause (``enqueue=False``) without coupling to internals.
    """
    try:
        before = {t.name for t in threading.enumerate() if t.name.startswith("loguru-writer")}

        configure_ray_worker_logging()
        # Emit through both stdlib and loguru so any lazily-started writer would appear.
        logging.getLogger("probe").info("via stdlib")
        logger.info("via loguru")

        after = {t.name for t in threading.enumerate() if t.name.startswith("loguru-writer")}
        new_writers = after - before
        assert not new_writers, f"enqueue writer thread present (deadlock-prone): {new_writers}"
    finally:
        _reset_logging()


def test_nested_stdlib_log_from_sink_does_not_deadlock():
    """A stdlib log emitted from inside a sink completes promptly (no hang).

    Mirrors the production trigger -- a library logging during the sink write --
    on the same thread. With ``enqueue=False`` this resolves inline (loguru's
    non-reentrant lock drops the nested record) instead of deadlocking on a
    cross-thread queue. Bounded by a timeout so a regression hangs the test
    rather than silently passing.
    """
    try:
        configure_ray_worker_logging()

        sink_calls: list[str] = []
        in_sink = {"active": False}

        def sink(message) -> None:
            sink_calls.append(message.record["message"])
            if not in_sink["active"]:
                in_sink["active"] = True
                try:
                    logging.getLogger("nested-probe").info("nested from sink")
                finally:
                    in_sink["active"] = False

        logger.remove()
        logger.add(sink, level="INFO", enqueue=False)

        done = threading.Event()

        def emit():
            logging.getLogger("outer-probe").info("outer message")
            done.set()

        t = threading.Thread(target=emit)
        t.start()
        t.join(timeout=5.0)

        assert done.is_set(), "logging from inside a sink deadlocked"
        assert "outer message" in sink_calls
    finally:
        _reset_logging()


def test_stdlib_logging_is_routed_through_loguru():
    """The intercept handler still forwards ordinary stdlib logs to loguru."""
    try:
        configure_ray_worker_logging()

        sink_calls: list[str] = []
        logger.remove()
        logger.add(lambda m: sink_calls.append(m.record["message"]), level="INFO", enqueue=False)

        logging.getLogger("some.library").info("hello from stdlib")

        assert "hello from stdlib" in sink_calls
    finally:
        _reset_logging()
