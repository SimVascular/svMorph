"""Tests for svmorph.logging – custom log level and logger helpers."""

import logging

import svmorph.logging as svlog


def test_timing_level_between_debug_and_info():
    assert logging.DEBUG < svlog.TIMING < logging.INFO


def test_timing_level_name_registered():
    assert logging.getLevelName(svlog.TIMING) == "TIMING"


def test_get_logger_returns_logger():
    logger = svlog.get_logger("svmorph.test_module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "svmorph.test_module"


def test_get_logger_prefixes_non_svmorph_name():
    logger = svlog.get_logger("mymodule")
    assert logger.name == "svmorph.mymodule"


def test_get_logger_does_not_double_prefix():
    logger = svlog.get_logger("svmorph.core.units")
    assert logger.name == "svmorph.core.units"


def test_set_level_changes_root_logger():
    original = logging.getLogger("svmorph").level
    svlog.set_level(logging.WARNING)
    assert logging.getLogger("svmorph").level == logging.WARNING
    # Restore
    svlog.set_level(original if original != 0 else logging.INFO)
