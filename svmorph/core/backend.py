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
from :func:`set_backend` (which is what the ``--backend`` command-line flag
calls) or, where that was not called, from the ``SVMORPH_BACKEND`` environment
variable; with neither it is ``"auto"``:

* ``"auto"`` uses Warp when a supported ``warp-lang`` is installed, Warp finds
  a CUDA device *and* the kernels load on it, and JAX in every other case.
  Whatever goes wrong with Warp on the way, it settles for JAX and says so.
* ``"jax"`` and ``"warp"`` are taken at their word.  Asking for Warp where it
  cannot be used raises instead of computing with JAX: a named backend is the
  backend that runs.  Without a CUDA device Warp runs its kernels on a single
  CPU core, which works but is not the speed-up a GPU gives.

The request is resolved the first time a deployment step asks for it, not when
svMorph is imported, so importing svMorph never imports or initialises Warp.
Front ends call :func:`configure` at startup instead, so that a backend named
by the user that cannot be used is reported before any work is done.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys

from svmorph.logging import get_logger

logger = get_logger(__name__)

#: The backends that can be named, besides ``"auto"``.
BACKENDS = ("jax", "warp")

#: Environment variable that names the backend when :func:`set_backend` was not called.
ENVIRONMENT_VARIABLE = "SVMORPH_BACKEND"

#: Oldest ``warp-lang`` release the Warp backend is tested with; the floor of the ``warp`` extra.
MINIMUM_WARP_VERSION = (1, 5)

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
        If *name* is not a backend, *device* is given without ``"warp"``, or
        Warp has no such device.
    ImportError
        If ``"warp"`` is asked for and Warp cannot be imported or initialised,
        or is older than :data:`MINIMUM_WARP_VERSION`.  The backend in use is
        then left as it was.
    """
    global _requested, _requested_device, _resolved

    if name is None:
        if device is not None:
            raise ValueError("A device can only be chosen together with the 'warp' backend")
        resolved = canonical = None
    else:
        canonical = _validated(name, "passed to set_backend")
        if device is not None and canonical != "warp":
            raise ValueError("A device can only be chosen together with the 'warp' backend")
        resolved = None
        if canonical == "warp":
            # Resolved here and not at the first step, so that a missing Warp or a device that
            # does not exist is reported where it was asked for
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


def configure(name: str | None = None) -> None:
    """Apply a front end's choice of backend, and check now that it can be honoured.

    For the entry points (the stent scripts, the GUI) to call at startup.
    *name* is what the user passed as ``--backend``, or ``None`` if they
    passed nothing, in which case a backend named in the environment is
    resolved straight away.  Either way a named backend that cannot be used
    raises here, before any work is done, rather than at the first step.
    ``"auto"`` is left to be resolved when it is first needed: it cannot fail.

    Raises
    ------
    ValueError, ImportError
        As :func:`set_backend`; also for a bad value of ``SVMORPH_BACKEND``.
    """
    if name is not None:
        set_backend(name)
    elif requested_backend() != "auto":
        _resolve()


def add_backend_argument(parser) -> None:
    """Add the ``--backend`` option to a command-line *parser* (an ``argparse.ArgumentParser``)."""
    parser.add_argument(
        "--backend", choices=("auto",) + BACKENDS, default=None,
        help="What computes stent deployment steps: 'jax' (CPU, the reference), 'warp' (NVIDIA Warp "
             "kernels on a CUDA GPU; needs the svmorph[warp] extra) or 'auto' (Warp if it is "
             f"installed and finds a CUDA GPU, else JAX). Default: ${ENVIRONMENT_VARIABLE} if set, "
             "else auto",
    )


def configure_from_command_line(name: str | None) -> None:
    """:func:`configure` for a command-line program: exit with a one-line error instead of a traceback."""
    try:
        configure(name)
    except (ImportError, ValueError) as error:
        sys.exit(f"error: {error}")


def _parsed_version(version: str) -> tuple[int, ...]:
    """The leading numeric components of a version string: ``"1.12.0.dev1"`` gives ``(1, 12, 0)``."""
    match = re.match(r"\d+(\.\d+)*", version)
    return tuple(int(part) for part in match.group().split(".")) if match else ()


def _import_warp():
    """Import Warp, turning a failure or an unsupported version into an ImportError that says what to do."""
    try:
        import warp
    except ImportError as error:
        raise ImportError(
            "The svMorph 'warp' backend needs NVIDIA Warp, which could not be imported "
            f"({error}). Install it with: pip install \"svmorph[warp]\"") from error
    minimum = ".".join(str(part) for part in MINIMUM_WARP_VERSION)
    version = str(getattr(warp, "__version__", ""))
    if _parsed_version(version)[:len(MINIMUM_WARP_VERSION)] < MINIMUM_WARP_VERSION:
        raise ImportError(
            f"The svMorph 'warp' backend needs warp-lang {minimum} or newer, and found {version or 'an unknown version'}. "
            "Upgrade it with: pip install --upgrade \"svmorph[warp]\"")
    return warp


def _warp_device_alias(device: str | None) -> str:
    """Return the alias of the Warp device to compute on, importing and initialising Warp to find out."""
    wp = _import_warp()
    try:
        cuda_available = wp.is_cuda_available()  # initialises Warp, which loads its native library
    except Exception as error:
        raise ImportError(f"NVIDIA Warp is installed but could not be initialised: {error}") from error
    if device is None:
        device = "cuda:0" if cuda_available else "cpu"
    try:
        return wp.get_device(device).alias
    except Exception as error:  # Warp raises several types for an unknown or unusable device
        raise ValueError(f"Warp has no usable device {device!r}: {error}") from error


def _load_warp_kernels(device: str) -> None:
    """Build svMorph's Warp kernels for *device* now, rather than at the first step that needs them."""
    import warp as wp

    from svmorph.core import sdf_warp

    wp.load_module(sdf_warp, device=device)


def _auto_warp_device() -> str | None:
    """Return Warp's CUDA device alias if ``"auto"`` should use Warp, otherwise ``None``.

    Never raises: ``"auto"`` is the default, so nothing about a Warp
    installation - too old, broken, unable to build the kernels - may turn a
    deployment that works on JAX into one that fails.
    """
    if "warp" not in available_backends():
        return None
    try:
        device = _warp_device_alias(None)
        if not device.startswith("cuda"):
            logger.debug("Warp is installed but found no CUDA device; computing with JAX")
            return None
        _load_warp_kernels(device)
    except Exception as error:
        logger.warning(f"Warp is installed but cannot be used ({error}); computing with JAX")
        return None
    return device


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
