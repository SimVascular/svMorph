"""Structured logging configuration for svMorph.

Provides a custom TIMING log level (between DEBUG and INFO) for performance
instrumentation, a pre-configured ``svmorph`` logger hierarchy, and small
helpers used across the package.

Typical usage inside a module::

    from svmorph.logging import get_logger, TIMING

    logger = get_logger(__name__)
    logger.log(TIMING, "Displacement computation: %.4f s", elapsed)
    logger.info("Saved mesh to %s", path)
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

# ---------------------------------------------------------------------------
# Custom TIMING level (15) – sits between DEBUG (10) and INFO (20)
# ---------------------------------------------------------------------------
TIMING: int = 15
logging.addLevelName(TIMING, "TIMING")


def _timing(self: logging.Logger, message: str, *args, **kwargs) -> None:
    if self.isEnabledFor(TIMING):
        self._log(TIMING, message, args, **kwargs)


logging.Logger.timing = _timing  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------
_DEFAULT_FMT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
_DEFAULT_DATEFMT = "%H:%M:%S"


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``svmorph`` hierarchy.

    If *name* already starts with ``svmorph`` it is used as-is; otherwise it is
    prefixed so that the root ``svmorph`` logger controls all verbosity.
    """
    if not name.startswith("svmorph"):
        name = f"svmorph.{name}"
    return logging.getLogger(name)


def setup_logging(level: int | str = logging.INFO, *, stream: Optional[object] = None) -> None:
    """Configure the root ``svmorph`` logger.

    Call once at application startup (e.g. in ``main.py``).  Subsequent calls
    are safe — duplicate handlers are avoided.

    Parameters
    ----------
    level:
        Minimum level to display.  Pass ``svmorph.logging.TIMING`` (15) to
        include timing instrumentation, or ``logging.DEBUG`` (10) for full
        diagnostics.
    stream:
        Output stream; defaults to ``sys.stderr``.
    """
    root = logging.getLogger("svmorph")

    if root.handlers:
        return

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter(_DEFAULT_FMT, datefmt=_DEFAULT_DATEFMT))
    root.addHandler(handler)
    root.setLevel(level)


def set_level(level: int | str) -> None:
    """Change the verbosity of the ``svmorph`` logger at runtime."""
    logging.getLogger("svmorph").setLevel(level)
