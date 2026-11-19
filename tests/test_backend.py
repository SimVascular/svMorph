"""Tests for svmorph.core.backend – choosing what computes the stent deployment step."""

from __future__ import annotations

import subprocess
import sys
import types

import numpy as np
import pytest

from svmorph.core import backend, deformation
from svmorph.core.units import set_unit_scale
from tests.helpers import make_stent_axis, make_tube


@pytest.fixture(autouse=True)
def _unit_scale():
    set_unit_scale(1.0)


@pytest.fixture
def no_request(monkeypatch):
    """Start from a clean slate: no set_backend() call and no environment variable."""
    monkeypatch.delenv(backend.ENVIRONMENT_VARIABLE, raising=False)
    backend.set_backend(None)


def _fake_warp(cuda: bool) -> types.ModuleType:
    """A stand-in for the warp module that reports a CUDA device, or none."""
    module = types.ModuleType("warp")
    module.is_cuda_available = lambda: cuda
    module.get_device = lambda alias: types.SimpleNamespace(alias=alias, name=f"fake {alias}")
    return module


# ── The request ──────────────────────────────────────────────────────────────

def test_request_defaults_to_auto(no_request):
    assert backend.requested_backend() == "auto"


def test_environment_variable_names_the_backend(no_request, monkeypatch):
    monkeypatch.setenv(backend.ENVIRONMENT_VARIABLE, " JAX ")
    assert backend.requested_backend() == "jax"
    assert backend.get_backend() == "jax"


def test_set_backend_takes_precedence_over_the_environment(no_request, monkeypatch):
    monkeypatch.setenv(backend.ENVIRONMENT_VARIABLE, "warp")
    backend.set_backend("jax")
    assert backend.requested_backend() == "jax"
    assert backend.get_backend() == "jax"


def test_set_backend_none_hands_the_choice_back_to_the_environment(no_request, monkeypatch):
    backend.set_backend("jax")
    monkeypatch.setenv(backend.ENVIRONMENT_VARIABLE, "auto")
    backend.set_backend(None)
    assert backend.requested_backend() == "auto"


def test_unknown_backend_is_refused_and_changes_nothing(no_request):
    backend.set_backend("jax")
    with pytest.raises(ValueError, match="torch"):
        backend.set_backend("torch")
    assert backend.requested_backend() == "jax"


def test_unknown_backend_in_the_environment_names_the_variable(no_request, monkeypatch):
    monkeypatch.setenv(backend.ENVIRONMENT_VARIABLE, "torch")
    with pytest.raises(ValueError, match=backend.ENVIRONMENT_VARIABLE):
        backend.get_backend()


@pytest.mark.parametrize("name", ["jax", "auto", None])
def test_a_device_needs_the_warp_backend(no_request, name):
    with pytest.raises(ValueError, match="device"):
        backend.set_backend(name, device="cpu")


def test_jax_is_always_available():
    assert "jax" in backend.available_backends()
    assert set(backend.available_backends()) <= set(backend.BACKENDS)


# ── Resolving it ─────────────────────────────────────────────────────────────

def test_jax_backend_has_no_warp_device(no_request):
    backend.set_backend("jax")
    assert backend.get_backend() == "jax"
    assert backend.warp_device() is None


def test_auto_is_jax_where_warp_is_not_installed(no_request, monkeypatch):
    monkeypatch.setattr(backend, "available_backends", lambda: ("jax",))
    assert backend.get_backend() == "jax"


def test_auto_is_jax_where_warp_finds_no_gpu(no_request, monkeypatch):
    monkeypatch.setattr(backend, "available_backends", lambda: backend.BACKENDS)
    monkeypatch.setitem(sys.modules, "warp", _fake_warp(cuda=False))
    assert backend.get_backend() == "jax"
    assert backend.warp_device() is None


def test_auto_is_warp_where_it_finds_a_gpu(no_request, monkeypatch):
    monkeypatch.setattr(backend, "available_backends", lambda: backend.BACKENDS)
    monkeypatch.setitem(sys.modules, "warp", _fake_warp(cuda=True))
    assert backend.get_backend() == "warp"
    assert backend.warp_device() == "cuda:0"


def test_auto_survives_a_warp_that_cannot_be_imported(no_request, monkeypatch):
    # Installed as far as the package index is concerned, broken when imported: None in
    # sys.modules is how Python says "this cannot be imported"
    monkeypatch.setattr(backend, "available_backends", lambda: backend.BACKENDS)
    monkeypatch.setitem(sys.modules, "warp", None)
    assert backend.get_backend() == "jax"


def test_named_warp_backend_that_cannot_be_imported_raises_and_changes_nothing(no_request, monkeypatch):
    # A named backend is never quietly replaced by another
    backend.set_backend("jax")
    monkeypatch.setitem(sys.modules, "warp", None)
    with pytest.raises(ImportError, match=r"svmorph\[warp\]"):
        backend.set_backend("warp")
    assert backend.get_backend() == "jax"


def test_named_warp_backend_in_the_environment_raises_when_first_needed(no_request, monkeypatch):
    monkeypatch.setenv(backend.ENVIRONMENT_VARIABLE, "warp")
    monkeypatch.setitem(sys.modules, "warp", None)
    with pytest.raises(ImportError, match=r"svmorph\[warp\]"):
        backend.get_backend()


def test_named_warp_backend_runs_on_the_cpu_without_a_gpu(no_request, monkeypatch):
    monkeypatch.setitem(sys.modules, "warp", _fake_warp(cuda=False))
    backend.set_backend("warp")
    assert (backend.get_backend(), backend.warp_device()) == ("warp", "cpu")


def test_named_warp_device_is_used(no_request, monkeypatch):
    monkeypatch.setitem(sys.modules, "warp", _fake_warp(cuda=True))
    backend.set_backend("warp", device="cuda:1")
    assert (backend.get_backend(), backend.warp_device()) == ("warp", "cuda:1")


def test_warp_device_that_does_not_exist_is_refused(no_request, monkeypatch):
    fake = _fake_warp(cuda=False)

    def get_device(alias):
        raise RuntimeError(f"no device {alias}")

    fake.get_device = get_device
    monkeypatch.setitem(sys.modules, "warp", fake)
    backend.set_backend("jax")
    with pytest.raises(ValueError, match="cuda:7"):
        backend.set_backend("warp", device="cuda:7")
    assert backend.get_backend() == "jax"


# ── What the choice touches ──────────────────────────────────────────────────

def test_importing_svmorph_does_not_import_warp():
    # The request is resolved at the first deployment step, never at import
    code = "import sys, svmorph, svmorph.core; assert 'warp' not in sys.modules, 'warp was imported'"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_jax_backend_never_reaches_for_the_warp_kernels(monkeypatch):
    # With the JAX backend selected (tests/conftest.py), a deployment step must not need Warp
    monkeypatch.setitem(sys.modules, "warp", None)
    monkeypatch.setitem(sys.modules, "svmorph.core.sdf_warp", None)
    surface, _, step = deformation.compute_sdf_contact_displacements(
        make_tube(), make_stent_axis(), s=-1.0, target_stent_radius=0.8, current_stent_radius=0.52)
    assert step == pytest.approx(0.01)
    assert np.abs(surface).max() > 0.0


def test_warp_backend_computes_the_deployment_step_with_the_warp_kernels(warp_device, monkeypatch):
    from svmorph.core import sdf_warp

    calls = []
    kernels = sdf_warp.sdf_contact_displacements
    monkeypatch.setattr(sdf_warp, "sdf_contact_displacements",
                        lambda *args, **kwargs: calls.append(kwargs) or kernels(*args, **kwargs))
    # ... and not with the reference kernels
    monkeypatch.setattr(deformation, "smin_sdf_capsule_contact_sculpt", None)
    surface, centerline, step = deformation.compute_sdf_contact_displacements(
        make_tube(), make_stent_axis(), s=-1.0, target_stent_radius=0.8, current_stent_radius=0.52)
    assert len(calls) == 1
    assert step == pytest.approx(0.01)
    assert np.abs(surface).max() > 0.0
