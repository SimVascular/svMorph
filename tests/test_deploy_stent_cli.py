"""Tests for svmorph.scripts.deploy_stent – command-line argument handling."""

import pytest

pytest.importorskip("vtk")  # the script imports the VTK I/O layer; the rest of the suite does not need VTK

from svmorph.scripts import deploy_stent  # noqa: E402

_REQUIRED = ["--mesh", "surface.vtp", "--cline", "centerline.vtp", "--start", "0"]


def test_flare_end_accepts_both():
    args = deploy_stent.build_parser().parse_args(_REQUIRED + ["--flare-R", "0.6", "--flare-end", "both"])
    assert args.flare_end == "both"
    assert args.cap_height_fraction == 1.0


@pytest.mark.parametrize("extra_args", [
    ["--cap-height-fraction", "0"],
    ["--cap-height-fraction", "1.5"],
    ["--flare-R", "0.6", "--flare-length", "0"],
])
def test_invalid_stent_shape_arguments_are_rejected(extra_args):
    # Rejected while parsing, before any input file is read
    with pytest.raises(SystemExit):
        deploy_stent.main(_REQUIRED + extra_args)
