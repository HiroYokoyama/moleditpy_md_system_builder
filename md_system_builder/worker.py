"""Run a long numpy job in a QThread with progress and cancel."""

from __future__ import annotations

import logging
import threading

from PySide6.QtCore import QObject, QThread, Signal

from .packing import Cancelled

logger = logging.getLogger(__name__)


class _Runner(QObject):
    progress = Signal(int, int)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, job):
        super().__init__()
        self.job = job
        self.stop = threading.Event()

    def run(self):
        try:
            result = self.job(self._progress, self.stop.is_set)
        except Cancelled:
            self.cancelled.emit()
            return
        except Exception as exc:
            logger.debug("Job failed", exc_info=True)
            self.failed.emit(str(exc) or type(exc).__name__)
            return
        self.finished.emit(result)

    def _progress(self, done, total):
        self.progress.emit(int(done), int(total))


class Job(QObject):
    """``job(progress, cancelled) -> result`` on a worker thread.

    Signals arrive on the thread that created the Job (the GUI thread).
    """

    progress = Signal(int, int)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, job, parent=None):
        super().__init__(parent)
        self._thread = QThread()
        self._runner = _Runner(job)
        self._runner.moveToThread(self._thread)
        self._thread.started.connect(self._runner.run)
        self._runner.progress.connect(self.progress)
        for signal, target in (
            (self._runner.finished, self.finished),
            (self._runner.failed, self.failed),
            (self._runner.cancelled, self.cancelled),
        ):
            signal.connect(target)
            signal.connect(self._thread.quit)

    def start(self):
        self._thread.start()

    def cancel(self):
        self._runner.stop.set()

    def is_running(self) -> bool:
        return self._thread.isRunning()

    def wait(self, ms: int = 5000) -> bool:
        return self._thread.wait(ms)
