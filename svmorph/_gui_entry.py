"""GUI entry point for the svMorph interactive application.

This module is the console-script target for ``svmorph-gui``.  It mirrors
the argument parsing that lives in the top-level ``main.py`` so that the
package works correctly whether invoked via the installed script or directly
with ``python main.py``.
"""

from __future__ import annotations

import argparse
import logging
import sys

_UNIT_SCALES = {"cm": 1.0, "mm": 10.0}


def main() -> None:
    """Parse CLI arguments, configure logging, and launch the Qt application."""
    try:
        from PyQt6.QtWidgets import QApplication
    except ModuleNotFoundError:
        print(
            "Error: the svMorph GUI requires PyQt6.\n"
            "Install it with:  pip install svmorph[gui]",
            file=sys.stderr,
        )
        sys.exit(1)

    from svmorph.core.units import set_unit_scale
    from svmorph.logging import TIMING, setup_logging
    from svmorph.gui.main_window import MainWindow

    parser = argparse.ArgumentParser(description="svMorph - interactive vascular morphing")
    parser.add_argument(
        "--units", choices=list(_UNIT_SCALES), default="cm",
        help="Coordinate unit system of input geometry (default: cm)",
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show timing instrumentation (TIMING level)",
    )
    verbosity.add_argument(
        "--debug", action="store_true",
        help="Show all diagnostic output (DEBUG level)",
    )
    args, remaining = parser.parse_known_args()

    set_unit_scale(_UNIT_SCALES[args.units])

    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = TIMING
    else:
        log_level = logging.INFO

    setup_logging(log_level)

    sys.argv = [sys.argv[0]] + remaining
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
