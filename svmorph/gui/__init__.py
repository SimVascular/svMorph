"""PyQt6 widgets and layout.

Requires the ``gui`` optional extra::

    pip install svmorph[gui]
"""

try:
    from svmorph.gui.main_window import MainWindow
except ModuleNotFoundError as exc:
    if "PyQt6" in str(exc):
        raise ModuleNotFoundError(
            "The svMorph GUI requires PyQt6. "
            "Install it with:  pip install svmorph[gui]"
        ) from exc
    raise
