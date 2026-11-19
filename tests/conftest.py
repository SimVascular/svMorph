"""Fixtures shared by the test modules."""

from __future__ import annotations

import pytest

from svmorph.core import backend


@pytest.fixture(autouse=True)
def _jax_backend_unless_asked_otherwise():
    """Compute with the JAX reference unless a test asks for Warp through ``warp_device``.

    Without this, ``"auto"`` would pick Warp on a machine that has it and a
    GPU, and the tests of the reference implementation would not be testing it.
    """
    backend.set_backend("jax")
    yield
    backend.set_backend(None)


@pytest.fixture(params=["cpu", "cuda:0"])
def warp_device(request):
    """Select the Warp backend on each device this machine has, and return the device alias.

    The kernels are the same on both, so a CI runner without a GPU exercises
    on the CPU the code that runs on CUDA elsewhere.
    """
    wp = pytest.importorskip("warp", reason="the Warp backend needs warp-lang (pip install 'svmorph[warp]')")
    if request.param.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("Warp found no CUDA device here")
    backend.set_backend("warp", device=request.param)
    return request.param
