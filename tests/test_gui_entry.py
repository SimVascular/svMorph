"""Tests for the GUI entry points – that both start up as far as parsing their arguments.

Nothing here opens a window: ``--help`` makes each entry point exit right after
its imports and its argument parser are in place, which is where the two
copies (``main.py`` for source runs, ``svmorph/_gui_entry.py`` for the
``svmorph-gui`` console script) can go wrong or drift apart.
"""

import os
import pathlib
import subprocess
import sys

import pytest

pytest.importorskip("PyQt6", reason="the GUI needs PyQt6 (pip install 'svmorph[gui]')")
pytest.importorskip("vtk")

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
ENTRY_POINTS = {
    "main.py": [str(REPOSITORY / "main.py")],
    "svmorph-gui": ["-c", "from svmorph._gui_entry import main; main()"],
}


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_entry_point_imports_and_offers_its_options(entry_point):
    # A failed import shows up here as a non-zero exit.  On Windows that includes importing Qt
    # before JAX: PyQt6 bundles an older C++ runtime than recent jaxlib builds can load after it.
    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    result = subprocess.run(
        [sys.executable, *ENTRY_POINTS[entry_point], "--help"],
        capture_output=True, text=True, cwd=REPOSITORY, env=environment)
    assert result.returncode == 0, result.stderr
    for option in ("--units", "--verbose", "--debug", "--backend"):
        assert option in result.stdout
    assert "{auto,jax,warp}" in result.stdout
