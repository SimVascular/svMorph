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


def _fake_warp(cuda: bool, version: str = "1.19.0") -> types.ModuleType:
    """A stand-in for the warp module that reports a CUDA device, or none."""
    module = types.ModuleType("warp")
    module.__version__ = version
    module.is_cuda_available = lambda: cuda
    module.get_device = lambda alias: types.SimpleNamespace(alias=alias, name=f"fake {alias}")
    return module


@pytest.fixture
def kernels_load(monkeypatch):
    """Let "auto" believe svMorph's kernels built on the device, without a real Warp to build them."""
    loaded = []
    monkeypatch.setattr(backend, "_load_warp_kernels", loaded.append)
    return loaded


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


def test_auto_is_warp_where_it_finds_a_gpu(no_request, monkeypatch, kernels_load):
    monkeypatch.setattr(backend, "available_backends", lambda: backend.BACKENDS)
    monkeypatch.setitem(sys.modules, "warp", _fake_warp(cuda=True))
    assert backend.get_backend() == "warp"
    assert backend.warp_device() == "cuda:0"
    assert kernels_load == ["cuda:0"], "auto should have built the kernels before committing to Warp"


def test_auto_is_jax_where_the_kernels_do_not_build(no_request, monkeypatch):
    # "auto" is the default: a Warp that is there but cannot run svMorph's kernels must not
    # turn a deployment that works on JAX into one that fails at its first step
    def fail(device):
        raise RuntimeError("kernel build failed")

    monkeypatch.setattr(backend, "available_backends", lambda: backend.BACKENDS)
    monkeypatch.setitem(sys.modules, "warp", _fake_warp(cuda=True))
    monkeypatch.setattr(backend, "_load_warp_kernels", fail)
    assert backend.get_backend() == "jax"


def test_auto_is_jax_where_warp_is_too_old(no_request, monkeypatch, kernels_load):
    # warp-lang installed for another project, older than the floor of the "warp" extra
    monkeypatch.setattr(backend, "available_backends", lambda: backend.BACKENDS)
    monkeypatch.setitem(sys.modules, "warp", _fake_warp(cuda=True, version="1.4.2"))
    assert backend.get_backend() == "jax"
    assert kernels_load == []


def test_auto_is_jax_where_warp_cannot_be_initialised(no_request, monkeypatch, kernels_load):
    fake = _fake_warp(cuda=True)

    def is_cuda_available():
        raise OSError("could not load the Warp native library")

    fake.is_cuda_available = is_cuda_available
    monkeypatch.setattr(backend, "available_backends", lambda: backend.BACKENDS)
    monkeypatch.setitem(sys.modules, "warp", fake)
    assert backend.get_backend() == "jax"


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


@pytest.mark.parametrize("version", ["1.4.2", "0.15.1", "1.5.0.dev1+cu12", "2.0.0"])
def test_named_warp_backend_needs_a_supported_version(no_request, monkeypatch, version):
    monkeypatch.setitem(sys.modules, "warp", _fake_warp(cuda=True, version=version))
    backend.set_backend("jax")
    if backend._parsed_version(version)[:2] >= backend.MINIMUM_WARP_VERSION:
        backend.set_backend("warp")
        assert backend.get_backend() == "warp"
    else:
        with pytest.raises(ImportError, match="warp-lang 1.5 or newer"):
            backend.set_backend("warp")
        assert backend.get_backend() == "jax"


def test_named_warp_backend_that_cannot_be_initialised_raises_import_error(no_request, monkeypatch):
    # Not whatever Warp raised: the front ends report ImportError and ValueError as one-line errors
    fake = _fake_warp(cuda=True)

    def is_cuda_available():
        raise OSError("could not load the Warp native library")

    fake.is_cuda_available = is_cuda_available
    monkeypatch.setitem(sys.modules, "warp", fake)
    backend.set_backend("jax")
    with pytest.raises(ImportError, match="could not be initialised"):
        backend.set_backend("warp")
    assert backend.get_backend() == "jax"


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


# ── Front ends ───────────────────────────────────────────────────────────────

def test_configure_selects_the_backend_named_on_the_command_line(no_request, monkeypatch):
    monkeypatch.setenv(backend.ENVIRONMENT_VARIABLE, "warp")
    backend.configure("jax")
    assert backend.requested_backend() == "jax"


def test_configure_leaves_auto_to_be_resolved_when_needed(no_request, monkeypatch):
    # Resolving "auto" imports Warp where it is installed; a front end that may never deploy a
    # stent should not pay for that at startup
    monkeypatch.setattr(backend, "_auto_warp_device", lambda: pytest.fail("auto was resolved"))
    backend.configure(None)
    assert backend.requested_backend() == "auto"
    assert backend._resolved is None


@pytest.mark.parametrize("value, error, message", [
    ("wrap", ValueError, "SVMORPH_BACKEND"),
    ("warp", ImportError, r"svmorph\[warp\]"),
], ids=["a typo", "warp without Warp"])
def test_configure_checks_a_backend_named_in_the_environment(no_request, monkeypatch, value, error, message):
    # Reported at startup, not at the first deployment step
    monkeypatch.setenv(backend.ENVIRONMENT_VARIABLE, value)
    monkeypatch.setitem(sys.modules, "warp", None)
    with pytest.raises(error, match=message):
        backend.configure(None)


@pytest.mark.parametrize("name, environment", [("warp", ""), (None, "warp"), (None, "wrap")],
                         ids=["flag", "environment", "typo in the environment"])
def test_configure_from_command_line_exits_with_one_line(no_request, monkeypatch, name, environment):
    monkeypatch.setenv(backend.ENVIRONMENT_VARIABLE, environment)
    monkeypatch.setitem(sys.modules, "warp", None)
    with pytest.raises(SystemExit) as exit_info:
        backend.configure_from_command_line(name)
    assert str(exit_info.value.code).startswith("error: ")
    assert "\n" not in str(exit_info.value.code)


def test_add_backend_argument_offers_every_backend(no_request):
    import argparse

    parser = argparse.ArgumentParser()
    backend.add_backend_argument(parser)
    assert parser.parse_args([]).backend is None
    for name in ("auto",) + backend.BACKENDS:
        assert parser.parse_args(["--backend", name]).backend == name


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
