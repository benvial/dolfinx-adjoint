import pathlib

import dolfinx.jit
import numpy as np
import pyadjoint
import pytest


def _repair_jit_cache() -> None:
    """Undo the FFCx cache damage left by a form compilation that raised.

    {py:func}`ffcx.codegeneration.jit.compile_forms` claims a form hash by creating an
    empty ``<hash>.c`` and writes the ``<hash>.c.cached`` marker beside it only once
    compilation has succeeded. A later compile of the same hash that finds the ``.c``
    without the marker assumes another process is mid-compile, waits ``timeout``
    seconds for the marker to appear, and then raises ``TimeoutError``. FFCx guards
    against exactly this by renaming the placeholder to ``.c.failed`` when compilation
    raises -- but it does so under ``except Exception``, and UFL raises both of its
    complex-mode rejections (``ArityMismatch`` and ``ComplexComparisonError``) from
    ``BaseException``, which that clause does not catch. Under a complex build the
    placeholder therefore survives, poisoning its hash for the rest of the session:
    every later test needing the same form fails with a ``TimeoutError`` that has
    nothing to do with what that test exercises, and since which tests those are
    depends on execution order, two identical runs disagree about which tests failed.

    Clearing the placeholders around every test keeps a failed compile inside the test
    that caused it -- the next test needing that form recompiles and sees the real error.

    Only *empty* placeholders are removed. That is exactly the set FFCx's own cleanup
    misses: UFL rejects a form during code generation, before CFFI has written any
    source, so a poisoned ``.c`` is always zero bytes, whereas a non-empty one without a
    marker is a compile still in flight -- possibly in another process sharing the same
    cache directory -- and must be left alone. A marker with no ``.c`` beside it is
    removed too: FFCx cannot make progress from that state (it re-claims the hash,
    recompiles, then dies writing a marker that already exists), and it is the one state
    this pruning could itself produce if it raced a foreign process.

    Deliberately not synchronised across MPI ranks. A barrier here would be the natural
    way to guarantee no rank is mid-compile, but ``dolfinx.jit.mpi_jit_decorator``
    catches only ``Exception`` as well, so the very failure this repairs escapes rank 0
    before its ``bcast`` and leaves the other ranks blocked inside it -- a barrier would
    then deadlock the run rather than let the remaining tests report. Every rank prunes
    independently instead, and the worst a lost race costs is a redundant recompile.
    """
    cache_dir = pathlib.Path(dolfinx.jit.get_options()["cache_dir"])
    for c_file in cache_dir.glob("*.c"):
        marker = c_file.with_suffix(".c.cached")
        if not marker.exists() and c_file.stat().st_size == 0:
            c_file.unlink(missing_ok=True)
    for marker in cache_dir.glob("*.c.cached"):
        if not marker.with_suffix("").exists():
            marker.unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def repair_jit_cache():
    """Stop a form that fails to compile from taking later, unrelated tests down with it.

    See {py:func}`_repair_jit_cache` for what is repaired and why. Runs before as well as
    after each test, so a cache left broken by an earlier (possibly interrupted) run does
    not leak into this one.
    """
    _repair_jit_cache()
    yield
    _repair_jit_cache()


@pytest.fixture
def assert_hessian_matches_finite_difference():
    """A Hessian-accuracy checker, as a more numerically robust
    alternative to {py:class}`pyadjoint.taylor_test`'s standard rate-3 Hessian-corrected check.

    That check needs cancelling several O(1) quantities down to an O(eps**3) remainder
    at eps <= 0.01, which the direct (MUMPS) LU factorization behind the
    adjoint/TLM/second-order-adjoint solves cannot always resolve to the precision it
    requires -- observed for saddle-point (e.g. Taylor-Hood velocity/pressure) and other
    blocked/nonlinear ``NonlinearProblem``/``LinearProblem`` systems in this suite, where
    ``mat_mumps_icntl_24`` alone does not fully resolve it and PETSc's ``SNESSolve`` can
    even intermittently fail to converge (error code 91) under repeated nearby re-solves.
    The returned checker instead only needs the *gradient*'s own precision (already
    validated wherever a rate-2 ``taylor_test`` passes), comparing
    ``Jhat.hessian(h)._ad_dot(h)`` directly against a central difference of
    ``Jhat.derivative()._ad_dot(h)``.

    Returns:
        A callable ``check(Jhat, m, h, *, fd_eps=1e-3, rtol=1e-2, atol=1e-2)`` -- see
        its own docstring for details. Exposed as a fixture (rather than a plain
        module-level function) so every test can use it with no import of its own,
        matching this project's ``--import-mode=importlib`` pytest configuration.
    """

    def _check(
        Jhat: pyadjoint.ReducedFunctional,
        m: pyadjoint.OverloadedType,
        h: pyadjoint.OverloadedType,
        *,
        fd_eps: float = 1e-3,
        rtol: float = 1e-2,
        atol: float = 1e-2,
    ) -> None:
        """Verify ``Jhat``'s Hessian-vector product against a central difference of its own gradient.

        Uses ``m``/``h``'s own ``_ad_add``/``_ad_mul`` (the same primitives
        ``pyadjoint.taylor_test`` perturbs its own evaluation points with) rather than
        type-specific perturbation code, so this works unchanged for a
        {py:class}`dolfinx_adjoint.Function`,
        {py:class}`dolfinx_adjoint.Constant``, or any other
        {py:class}`pyadjoint.OverloadedType` control.

        Leaves ``Jhat`` evaluated at ``m`` on return.

        ``Hm`` and ``Hm_fd`` are two independent estimates of the same mathematical
        quantity, so they should agree up to two, unrelated, and much smaller error
        sources: (1) central-difference truncation, ``O(fd_eps**2)`` relative --
        `<1e-5` relative at the default ``fd_eps=1e-3``, negligible here; and (2)
        whatever precision the adjoint/TLM/second-order-adjoint linear solves and the
        forward (possibly SNES) solve actually achieve at the two perturbed evaluation
        points. If a particular problem's own solves are markedly less precise (an
        iterative KSP/SNES rather than a direct LU factorization, say), loosen
        ``rtol``/``atol`` explicitly for that call rather than lowering the default.

        Args:
            Jhat: The reduced functional to check.
            m: The control value to evaluate the Hessian at.
            h: The direction to evaluate the Hessian-vector product/gradient in.
            fd_eps: Finite-difference step size, in units of ``h``.
            rtol: Relative tolerance passed to ``numpy.isclose`` -- see above for why
                ``1e-2`` is the default.
            atol: Absolute tolerance passed to ``numpy.isclose`` -- see above for why
                ``1e-2`` is the default.
        """

        def dJdm_at(scale: float) -> float:
            Jhat(m._ad_add(h._ad_mul(scale)))
            return Jhat.derivative()._ad_dot(h)

        Jhat(m)
        Jhat.derivative()
        Hm = Jhat.hessian(h)._ad_dot(h)
        Hm_fd = (dJdm_at(fd_eps) - dJdm_at(-fd_eps)) / (2 * fd_eps)
        Jhat(m)
        assert np.isclose(Hm, Hm_fd, rtol=rtol, atol=atol), (
            f"Hessian-vector product {Hm} did not match central-difference-of-gradient "
            f"estimate {Hm_fd} (fd_eps={fd_eps})"
        )

    return _check
