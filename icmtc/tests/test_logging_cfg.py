# Author: Maher Ahmed Maher Shola
# Email:  maher.shola.officer@gmail.com
#         M.shoala61308@student.aast.edu
# Date:   04/06/2026

"""test_logging_cfg.py — hardware-free pytest suite for logging_cfg.py.

All tests are pure-stdlib and touch only the process-global `logging`
tree plus a pytest `tmp_path`; no Raspberry Pi, camera, or MAVLink link is
ever involved.  Handler introspection keys off the module's private
`_HANDLER_MARKER`, the same attribute `configure_logging` uses to recognise
(and cleanly replace) its own handlers.

Run:
    pytest test_logging_cfg.py -v
"""

from __future__ import annotations

import logging
import logging.handlers
import re
from pathlib import Path

import pytest

from ioutils import logging_cfg
from ioutils.logging_cfg import configure_logging, get_logger

# ISO-8601 wall-clock timestamp with millisecond precision, e.g.
# 2026-07-03T22:41:07.123 — the format L2/L4 promise in the module header.
_TS_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}")


# ---------------------------------------------------------------------------
# Fixtures / helpers

@pytest.fixture(autouse=True)
def _restore_root_logger():
    """Undo configure_logging's global mutations after every test.

    Removes and closes only the handlers this module attached (identified by
    `_HANDLER_MARKER`) and restores the root logger's level, so tests do not
    leak open file handles pointing at deleted tmp dirs or bleed level state
    into each other or into pytest's own logging plugin.
    """
    root = logging.getLogger()
    saved_level = root.level
    yield
    for handler in list(root.handlers):
        if getattr(handler, logging_cfg._HANDLER_MARKER, False):
            root.removeHandler(handler)
            handler.close()
    root.setLevel(saved_level)


def _own_handlers() -> list[logging.Handler]:
    """Return only the handlers configure_logging attached to the root."""
    root = logging.getLogger()
    return [h for h in root.handlers if getattr(h, logging_cfg._HANDLER_MARKER, False)]


def _console_handlers() -> list[logging.Handler]:
    """Our stream handlers that are NOT file handlers (i.e. the console)."""
    return [
        h
        for h in _own_handlers()
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    ]


def _file_handlers() -> list[logging.handlers.RotatingFileHandler]:
    """Our rotating file handlers."""
    return [h for h in _own_handlers() if isinstance(h, logging.handlers.RotatingFileHandler)]


# ---------------------------------------------------------------------------
# Test 1 — both handlers created at the requested level

def test_handlers_created(tmp_path: Path):
    """With a log_dir, the root gains one console + one RotatingFileHandler."""
    configure_logging(level="INFO", log_dir=tmp_path)

    console = _console_handlers()
    files = _file_handlers()

    assert len(console) == 1, "expected exactly one console handler"
    assert len(files) == 1, "expected exactly one rotating file handler"
    assert console[0].level == logging.INFO
    assert files[0].level == logging.INFO
    assert logging.getLogger().level == logging.INFO


# ---------------------------------------------------------------------------
# Test 2 — idempotent: repeated calls neither stack handlers nor double lines

def test_idempotent(tmp_path: Path):
    """A second configure_logging replaces our handlers instead of adding."""
    configure_logging(level="INFO", log_dir=tmp_path)
    first_count = len(_own_handlers())
    assert first_count == 2  # one console + one file

    configure_logging(level="INFO", log_dir=tmp_path)
    assert len(_own_handlers()) == first_count, "handlers stacked on re-invocation"

    # And a single log call must produce exactly one line on disk, not two.
    token = "idempotency-unique-marker-42"
    get_logger("test.idempotent").info(token)

    content = (tmp_path / "mission.log").read_text()
    assert content.count(token) == 1, "log line was duplicated"


# ---------------------------------------------------------------------------
# Test 3 — the requested level is honoured (DEBUG dropped, INFO kept)

def test_level_respected(tmp_path: Path):
    """At level=INFO a DEBUG record is suppressed but an INFO record is not."""
    configure_logging(level="INFO", log_dir=tmp_path, console=False)
    log = get_logger("test.levels")

    log.debug("debug-must-not-appear")
    log.info("info-must-appear")

    content = (tmp_path / "mission.log").read_text()
    assert "debug-must-not-appear" not in content
    assert "info-must-appear" in content


# ---------------------------------------------------------------------------
# Test 4 — a logged message actually reaches the file on disk

def test_file_written(tmp_path: Path):
    """A message logged through get_logger lands in the rotating log file."""
    configure_logging(level="INFO", log_dir=tmp_path, console=False)
    get_logger("test.file").info("hello-from-the-flight-log")

    log_file = tmp_path / "mission.log"
    assert log_file.exists()
    assert "hello-from-the-flight-log" in log_file.read_text()


# ---------------------------------------------------------------------------
# Test 5 — rotation is configured and actually rolls over

def test_rotation_configured(tmp_path: Path):
    """maxBytes/backupCount match the args and writing past the cap rolls over."""
    configure_logging(
        level="INFO",
        log_dir=tmp_path,
        max_bytes=512,
        backup_count=3,
        console=False,
    )

    handler = _file_handlers()[0]
    assert handler.maxBytes == 512
    assert handler.backupCount == 3

    # Write well past the 512-byte cap so at least one rollover occurs.
    log = get_logger("test.rotation")
    for i in range(60):
        log.info("rotation filler line number %03d — padding padding padding", i)

    assert (tmp_path / "mission.log.1").exists(), "expected a rolled-over backup file"


# ---------------------------------------------------------------------------
# Test 6 — graceful fallback when the log dir can't be created

def test_graceful_fallback_unwritable_dir(tmp_path: Path, caplog):
    """An uncreatable log_dir degrades to console-only with a warning, no raise."""
    # Make a regular file, then ask for a log dir *underneath* it: mkdir must
    # fail with NotADirectoryError (an OSError), exercising the fallback path.
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("i am a file, not a directory")
    bad_dir = blocker / "logs"

    with caplog.at_level(logging.WARNING):
        # Must not raise.
        returned = configure_logging(level="INFO", log_dir=bad_dir, console=True)

    assert returned is logging.getLogger()
    # No file handler was attached; console survives.
    assert _file_handlers() == []
    assert len(_console_handlers()) == 1
    # A warning was emitted explaining the degradation.
    assert any(
        rec.levelno == logging.WARNING and "console-only" in rec.getMessage()
        for rec in caplog.records
    ), "expected a console-only fallback warning"


# ---------------------------------------------------------------------------
# Test 7 — an unknown level string falls back to INFO without raising

def test_unknown_level_falls_back(tmp_path: Path):
    """level='BOGUS' warns and defaults to INFO rather than crashing."""
    with pytest.warns(RuntimeWarning):
        configure_logging(level="BOGUS", log_dir=tmp_path, console=False)

    assert logging.getLogger().level == logging.INFO
    assert _file_handlers()[0].level == logging.INFO


# ---------------------------------------------------------------------------
# Test 8 — get_logger attaches no handlers and logs exactly once

def test_get_logger_no_duplication(tmp_path: Path):
    """get_logger returns a correctly named, handler-free, propagating logger."""
    configure_logging(level="INFO", log_dir=tmp_path, console=False)

    log = get_logger("test.some.module")
    assert log.name == "test.some.module"
    assert log.handlers == [], "get_logger must not attach its own handlers"
    assert log.propagate is True

    token = "single-emission-token-99"
    log.info(token)

    content = (tmp_path / "mission.log").read_text()
    assert content.count(token) == 1, "message emitted more than once"


# ---------------------------------------------------------------------------
# Test 9 — the formatter carries timestamp, level, name, and thread name

def test_format_fields(tmp_path: Path):
    """The configured formatter includes ms timestamp, level, name, thread."""
    configure_logging(level="DEBUG", log_dir=tmp_path, console=False)
    formatter = _file_handlers()[0].formatter
    assert formatter is not None

    record = logging.LogRecord(
        name="capture.detect",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="thermal flag candidate at %d",
        args=(7,),
        exc_info=None,
    )
    record.threadName = "capture-thread"

    formatted = formatter.format(record)

    assert _TS_PATTERN.search(formatted), f"no ms ISO timestamp in {formatted!r}"
    assert "WARNING" in formatted
    assert "capture.detect" in formatted
    assert "capture-thread" in formatted
    assert "thermal flag candidate at 7" in formatted
