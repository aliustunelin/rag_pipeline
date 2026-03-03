import logging
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".csv", ".json"}


class DataFileHandler(FileSystemEventHandler):
    """Watches the data directory for file changes and triggers re-indexing."""

    def __init__(self, on_change_callback):
        super().__init__()
        self.on_change_callback = on_change_callback

    def on_modified(self, event):
        if isinstance(event, FileModifiedEvent):
            self._handle(event.src_path)

    def on_created(self, event):
        if isinstance(event, FileCreatedEvent):
            self._handle(event.src_path)

    def _handle(self, file_path: str):
        path = Path(file_path)
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            logger.info(f"File change detected: {path.name}")
            self.on_change_callback()


class DataWatcher:
    """Manages the watchdog observer for the data directory."""

    def __init__(self, data_dir: str, on_change_callback):
        self.data_dir = data_dir
        self.handler = DataFileHandler(on_change_callback)
        self.observer = Observer()

    def start(self):
        self.observer.schedule(self.handler, self.data_dir, recursive=False)
        self.observer.start()
        logger.info(f"Watching data directory: {self.data_dir}")

    def stop(self):
        self.observer.stop()
        self.observer.join()
