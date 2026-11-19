"""Entry point for the svMorph interactive vascular morphing application."""

import argparse
import logging
import sys

from PyQt6.QtWidgets import QApplication

from svmorph.core.backend import set_backend
from svmorph.core.units import set_unit_scale
from svmorph.logging import TIMING, setup_logging
from svmorph.gui.main_window import MainWindow

_UNIT_SCALES = {"cm": 1.0, "mm": 10.0}


def main():
    """Parse CLI arguments, configure logging, and launch the Qt application."""
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
    parser.add_argument(
        "--backend", choices=("auto", "jax", "warp"), default=None,
        help="What computes stent deployment steps: 'jax' (CPU), 'warp' (NVIDIA Warp on a CUDA GPU; "
             "needs the svmorph[warp] extra) or 'auto' (Warp if it is installed and finds a CUDA GPU, "
             "else JAX). Default: $SVMORPH_BACKEND if set, else auto",
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

    if args.backend is not None:
        try:
            set_backend(args.backend)
        except ImportError as error:  # --backend warp without Warp installed
            parser.error(str(error))

    sys.argv = [sys.argv[0]] + remaining
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
