"""Which implementation computes the SDFStent deployment step.

``jax``
    The reference implementation in :mod:`svmorph.core.deformation`: the stent
    SDF as JIT-compiled JAX kernels, the contact search with SciPy's k-d tree.
    Always available, and what every other workflow (aneurysm, stenosis) uses
    regardless of the choice made here.
``warp``
    The same deployment step as `NVIDIA Warp <https://github.com/NVIDIA/warp>`_
    kernels (:mod:`svmorph.core.sdf_warp`), which run on a CUDA GPU.  Optional:
    it needs the ``warp-lang`` package (``pip install "svmorph[warp]"``).

This module is the single place where the choice is made.  The request comes
from, in order of precedence, :func:`set_backend` (which is what the
``--backend`` command-line flag calls), the ``SVMORPH_BACKEND`` environment
variable, and otherwise ``"auto"``:

* ``"auto"`` picks Warp when ``warp-lang`` is installed *and* Warp finds a CUDA
  device, and JAX in every other case.  It never fails.
* ``"jax"`` and ``"warp"`` are taken at their word.  Asking for Warp where it
  cannot be imported raises instead of computing with JAX: a named backend is
  the backend that runs.  Without a CUDA device Warp runs its kernels on a
  single CPU core, which works but is not the speed-up a GPU gives.

The request is resolved the first time a deployment step asks for it, not when
svMorph is imported, so importing svMorph never imports or initialises Warp.
"""

from __future__ import annotations

import importlib.util
import os

from svmorph.logging import get_logger

logger = get_logger(__name__)

#: The backends that can be named, besides ``"auto"``.
BACKENDS = ("jax", "warp")

#: Environment variable that names the backend when :func:`set_backend` was not called.
ENVIRONMENT_VARIABLE = "SVMORPH_BACKEND"

# What set_backend() asked for; None while it has not been called (or was reset)
_requested: str | None = None
_requested_device: str | None = None
# (backend, Warp device alias or None) once the request has been resolved
_resolved: tuple[str, str | None] | None = None


def _validated(name: str, source: str) -> str:
    """Return *name* in canonical form, or raise naming where the bad value came from."""
    canonical = str(name).strip().lower()
    if canonical != "auto" and canonical not in BACKENDS:
        raise ValueError(
            f"Unknown svMorph backend {name!r} ({source}): expected one of "
            f"{', '.join(('auto',) + BACKENDS)}")
    return canonical


def available_backends() -> tuple[str, ...]:
    """Return the backends whose library is installed.

    Nothing is imported to find out, so a Warp installation that fails to
    load is still listed; :func:`set_backend` is what reports that.

    Returns
    -------
    tuple[str, ...]
        ``("jax",)`` or ``("jax", "warp")``.
    """
    try:
        warp_installed = importlib.util.find_spec("warp") is not None
    except (ImportError, ValueError):  # a half-removed module: None in sys.modules
        warp_installed = False
    return BACKENDS if warp_installed else ("jax",)


def requested_backend() -> str:
    """Return the backend that was asked for: ``"auto"``, ``"jax"`` or ``"warp"``.

    This is the request, not its outcome; :func:`get_backend` says what
    ``"auto"`` turned into.
    """
    if _requested is not None:
        return _requested
    from_environment = os.environ.get(ENVIRONMENT_VARIABLE, "").strip()
    if from_environment:
        return _validated(from_environment, f"from the {ENVIRONMENT_VARIABLE} environment variable")
    return "auto"


def set_backend(name: str | None = "auto", *, device: str | None = None) -> None:
    """Choose the backend that computes stent deployment steps from here on.

    Parameters
    ----------
    name : {"auto", "jax", "warp"} or None
        The backend to use.  ``None`` withdraws an earlier call, so that the
        ``SVMORPH_BACKEND`` environment variable (or ``"auto"``) decides again.
    device : str or None
        Warp device alias to compute on, e.g. ``"cuda:0"``, ``"cuda:1"`` or
        ``"cpu"``.  Only meaningful with ``name="warp"``; by default Warp's
        first CUDA device is used, or the CPU where there is none.

    Raises
    ------
    ValueError
        If *name* is not a backend, or *device* is given without ``"warp"``.
    ImportError
        If ``"warp"`` is asked for and Warp cannot be imported.  The backend
        in use is then left as it was.
    """
    global _requested, _requested_device, _resolved

    if name is None:
        if device is not None:
            raise ValueError("A device can only be chosen together with the 'warp' backend")
        _requested = _requested_device = _resolved = None
        return

    canonical = _validated(name, "passed to set_backend")
    if device is not None and canonical != "warp":
        raise ValueError("A device can only be chosen together with the 'warp' backend")
    resolved = None
    if canonical == "warp":
        # Resolved here and not at the first step, so that a missing Warp or a
        # device that does not exist is reported where it was asked for
        resolved = ("warp", _warp_device_alias(device))
    _requested, _requested_device, _resolved = canonical, device, resolved
    if resolved is not None:
        _report(resolved)


def get_backend() -> str:
    """Return the backend that computes stent deployment steps: ``"jax"`` or ``"warp"``.

    Resolves an ``"auto"`` request on the first call, which under ``"auto"``
    imports Warp where it is installed.
    """
    return _resolve()[0]


def warp_device() -> str | None:
    """Return the alias of the Warp device in use, or ``None`` under the JAX backend."""
    return _resolve()[1]


def _import_warp():
    """Import Warp, turning a failure into an ImportError that says how to get it."""
    try:
        import warp
    except ImportError as error:
        raise ImportError(
            "The svMorph 'warp' backend needs NVIDIA Warp, which could not be imported "
            f"({error}). Install it with: pip install \"svmorph[warp]\"") from error
    return warp


def _warp_device_alias(device: str | None) -> str:
    """Return the alias of the Warp device to compute on, importing Warp to find out."""
    wp = _import_warp()
    if device is None:
        return wp.get_device("cuda:0").alias if wp.is_cuda_available() else "cpu"
    try:
        return wp.get_device(device).alias
    except Exception as error:  # Warp raises several types for an unknown or unusable device
        raise ValueError(f"Warp has no usable device {device!r}: {error}") from error


def _auto_warp_device() -> str | None:
    """Return Warp's CUDA device alias if ``"auto"`` should use Warp, otherwise ``None``."""
    if "warp" not in available_backends():
        return None
    try:
        import warp as wp
        if wp.is_cuda_available():
            return wp.get_device("cuda:0").alias
    except Exception as error:  # "auto" never fails: a broken Warp install means JAX
        logger.warning(f"Warp is installed but could not be used ({error}); computing with JAX")
        return None
    logger.debug("Warp is installed but found no CUDA device; computing with JAX")
    return None


def _report(resolved: tuple[str, str | None]) -> None:
    """Log the backend a request resolved to."""
    name, device = resolved
    if name == "jax":
        logger.info("Stent deployment backend: JAX")
        return
    import warp as wp
    logger.info(f"Stent deployment backend: Warp on {device} ({wp.get_device(device).name})")
    if device == "cpu":
        logger.info(
            "Warp is running its kernels on one CPU core; the speed-up it is meant for "
            "needs a CUDA-capable NVIDIA GPU")


def _resolve() -> tuple[str, str | None]:
    """Turn the request into a backend and, for Warp, a device; cached until the next request."""
    global _resolved

    if _resolved is None:
        requested = requested_backend()
        if requested == "warp":  # named in the environment; set_backend() resolves its own
            resolved = ("warp", _warp_device_alias(_requested_device))
        elif requested == "jax":
            resolved = ("jax", None)
        else:
            device = _auto_warp_device()
            resolved = ("jax", None) if device is None else ("warp", device)
        _resolved = resolved
        _report(resolved)
    return _resolved
