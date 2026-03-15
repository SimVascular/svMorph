"""Entry point for the svMorph interactive vascular morphing application."""

import argparse
import logging
import sys

from PyQt6.QtWidgets import QApplication

from svmorph.logging import TIMING, setup_logging
from svmorph.gui.main_window import MainWindow


def main():
    """Parse CLI arguments, configure logging, and launch the Qt application."""
    parser = argparse.ArgumentParser(description="svMorph – interactive vascular morphing")
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
