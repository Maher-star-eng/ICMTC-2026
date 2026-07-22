# Author: Maher Ahmed Maher Shola
# Email:  maher.shola.officer@gmail.com
#         M.shoala61308@student.aast.edu
# Date:   04/06/2026

"""logging_cfg.py - centralized diagnostic logging setup (ICMTC 2026, Mission 2).

Configures Python's standard `logging` for the whole in-process vision
pipeline (capture / detect / track / geo-locate / mission decisions), which
runs across several threads on the Raspberry Pi 5.  Call `configure_logging`
exactly once at startup (from mission.py); every other module then does
`log = get_logger(__name__)` and inherits the root configuration.

SCOPE BOUNDARY
--------------
This is operational/diagnostic logging only: "what is the software doing,
what warned, what broke" - the log a developer reads over SSH or after a
flight.  It is NOT the judge log and NOT the mission-results/coordinate
output.  The structured per-flag GPS record submitted to judges belongs to a
separate `output.py` module and is out of scope here.  This module never
serializes flag coordinates or mission results.

================================================================================
ASSUMPTIONS (collected here; also marked inline with # ASSUMPTION: <tag>)
================================================================================

L1  DEFAULT LEVEL
    `configure_logging` defaults to level="INFO" when the caller passes
    nothing.  DEBUG is reserved for verbose per-frame tracing during bench
    testing; INFO is the right signal-to-noise ratio for a live flight.

L2  FORMAT STRING
    "%(asctime)s.%(msecs)03d %(levelname)-8s [%(threadName)s] %(name)s: "
    "%(message)s"
    Millisecond-precision, human-readable wall-clock timestamp; level;
    thread name (capture/inference/telemetry run on separate threads and
    must be untangleable in a single interleaved log); logger name; message.

L3  ROTATION SIZE
    max_bytes=5_000_000 (~5 MB) and backup_count=3 (bounded total disk use,
    ~20 MB including the active file) are sane defaults for an SD-card-
    constrained Raspberry Pi 5 running for the duration of a mission flight.

L4  WALL-CLOCK VS MONOTONIC TIMESTAMP
    This module's timestamp is `time.time()`-based wall-clock (via the
    standard `logging.Formatter`/`asctime`), for human readability over SSH
    or in a post-flight log review.  It is not the monotonic pose clock used
    by telemetry.py.  If a module ever needs to correlate a log line with a
    specific monotonic pose sample, it should include that numeric monotonic
    timestamp in the message payload itself - not this module's job.

L5  QueueHandler / QueueListener (DEFERRED)
    Standard `logging` level-gating (an early integer comparison before any
    formatting or I/O happens) is cheap enough that nothing here should
    force expensive work onto the capture/inference hot path at INFO level.
    `logging.handlers.QueueHandler` + `QueueListener` could move file I/O off
    the calling thread entirely; documented here as a future option, not
    implemented now, unless profiling on real hardware demands it.

================================================================================
"""

from __future__ import annotations

import logging
import logging.handlers
import warnings
from pathlib import Path

_LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d %(levelname)-8s [%(threadName)s] %(name)s: %(message)s"
)  # ASSUMPTION L2
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

_DEFAULT_LEVEL = "INFO"  # ASSUMPTION L1
_DEFAULT_MAX_BYTES = 5_000_000  # ASSUMPTION L3
_DEFAULT_BACKUP_COUNT = 3  # ASSUMPTION L3

_LEVELS: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}

# Marker attribute set on handlers this module attaches, so a re-invocation
# of configure_logging can find and remove exactly its own handlers without
# disturbing handlers some other part of the process may have added.
_HANDLER_MARKER = "_icmtc_logging_cfg"


def _resolve_level(level: str) -> int:
    """Map a level name to a `logging` level constant, falling back to INFO."""
    resolved = _LEVELS.get(level.upper())
    if resolved is None:
        warnings.warn(
            f"Unknown logging level {level!r}; falling back to {_DEFAULT_LEVEL}.",
            RuntimeWarning,
            stacklevel=3,
        )
        return _LEVELS[_DEFAULT_LEVEL]
    return resolved


def _remove_own_handlers(logger: logging.Logger) -> None:
    """Remove and close only the handlers this module previously attached."""
    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            logger.removeHandler(handler)
            handler.close()


def configure_logging(
    level: str = _DEFAULT_LEVEL,
    log_dir: str | Path | None = None,
    log_filename: str = "mission.log",
    max_bytes: int = _DEFAULT_MAX_BYTES,
    backup_count: int = _DEFAULT_BACKUP_COUNT,
    console: bool = True,
) -> logging.Logger:
    """Configure the root logger for the whole application.

    Idempotent: calling this more than once (e.g. if mission.py's startup
    path runs twice) replaces this module's own handlers rather than
    stacking duplicates, so log lines are never doubled.

    Never raises: if `log_dir` cannot be created or is not writable, this
    falls back to console-only logging and emits a `logging.warning` on the
    configured logger, consistent with the system-wide rule that a
    non-flight-critical failure degrades gracefully instead of taking the
    process down.

    Args:
        level: One of "DEBUG", "INFO", "WARNING", "ERROR" (case-insensitive).
            An unrecognized value falls back to "INFO" with a `RuntimeWarning`
            rather than raising.  # ASSUMPTION L1
        log_dir: Directory to hold the rotating log file.  `None` (the
            default) means console-only; no file handler is attached.
        log_filename: File name created inside `log_dir`.
        max_bytes: Rotation threshold per file, bytes.  # ASSUMPTION L3
        backup_count: Number of rotated backups to retain.  # ASSUMPTION L3
        console: Whether to attach a console (stream) handler.

    Returns:
        The configured root `logging.Logger`.
    """
    resolved_level = _resolve_level(level)

    root = logging.getLogger()
    _remove_own_handlers(root)
    root.setLevel(resolved_level)

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)  # ASSUMPTION L4

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(resolved_level)
        console_handler.setFormatter(formatter)
        setattr(console_handler, _HANDLER_MARKER, True)
        root.addHandler(console_handler)

    if log_dir is not None:
        try:
            log_dir_path = Path(log_dir)
            log_dir_path.mkdir(parents=True, exist_ok=True)
            file_path = log_dir_path / log_filename
            file_handler = logging.handlers.RotatingFileHandler(
                file_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
            )
        except OSError as exc:
            root.warning(
                "Cannot create/write log file in %r (%s); falling back to "
                "console-only logging.",
                str(log_dir),
                exc,
            )
        else:
            file_handler.setLevel(resolved_level)
            file_handler.setFormatter(formatter)
            setattr(file_handler, _HANDLER_MARKER, True)
            root.addHandler(file_handler)

    return root


def get_logger(name: str) -> logging.Logger:
    """Return a named logger that inherits the root configuration.

    Attaches no handlers of its own; relies on propagation to the root
    logger configured by `configure_logging`, so output is never duplicated.

    Usage: every module does `log = get_logger(__name__)` at import time;
    `configure_logging` must have been called once by mission.py before any
    log record is expected to reach console or file.

    Args:
        name: Logger name, conventionally `__name__` of the calling module.

    Returns:
        The named `logging.Logger`.
    """
    return logging.getLogger(name)
