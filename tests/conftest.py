import gc

from mpi4py import MPI

import numpy as np
import pyadjoint
import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Take the cyclic garbage collector out of the loop for the duration of a parallel run.

    A {py:class}`dolfinx.fem.petsc.LinearProblem`/``NonlinearProblem`` releases its PETSc
    ``Mat``/``Vec``/``KSP``/``SNES`` objects from ``__del__``, and destroying a PETSc object is
    collective over the communicator it was built on. That is safe as long as the object dies by
    reference counting, which happens at the same point in the program on every rank -- the
    invariant ``test_linear_problem_released_by_refcounting_not_gc`` exists to protect. It is not
    safe when the object dies in a cyclic collection, because *when* the cyclic collector runs is
    decided by each rank's own allocation counters, and those diverge.

    They diverge reliably, in one specific window. ``dolfinx.jit.mpi_jit_decorator`` has rank 0
    compile a form while every other rank waits for the result in ``comm.bcast``: rank 0 allocates
    heavily inside FFCx, the others allocate nothing. So the collector fires on rank 0, deep inside
    the compile, and if any Problem has become cyclic garbage by then its ``__del__`` runs there
    and enters a collective that the waiting ranks -- sitting in a ``bcast`` on the same
    communicator -- will never join. Both processes then spin at full CPU forever. Observed as a
    stalled ``mpirun -n 2`` complex-scalar run whose two ranks were in exactly those two places.

    Problems become cyclic garbage more easily than the refcounting invariant suggests, because a
    cycle does not have to be one the object itself takes part in: the traceback of any exception
    raised inside a test holds that test's frame, which holds its local Problem, and a traceback is
    itself cyclic. A complex-scalar run raises far more of them than a real one -- every
    second-order-adjoint refusal handled below arrives as an exception -- which is why the stall is
    specific to the complex build even though nothing about it is.

    Disabling automatic collection removes the nondeterminism rather than trying to chase the
    cycles: nothing is finalised until ``collect_cycles_between_tests`` collects, at a point every
    rank reaches having run the same code. Serial runs keep the collector, since with one rank
    there is no divergence to protect against and no collective to deadlock.
    """
    if MPI.COMM_WORLD.size > 1:
        gc.disable()


@pytest.fixture(autouse=True)
def collect_cycles_between_tests():
    """Collect cyclic garbage at a point every rank reaches together.

    The counterpart to the collector ``pytest_configure`` turns off: with automatic collection
    disabled, cycles accumulate until something collects them, and a test boundary is the coarsest
    place where every rank is provably at the same point in the program. Anything finalised here --
    including the collective PETSc destructors that motivate the whole arrangement -- therefore runs
    in the same order on every rank.

    A test that needs a collection earlier than this (``test_linear_problem_rebuilt_after_garbage_collection``
    and friends) calls {py:func}`gc.collect` itself, which works whether or not automatic collection
    is enabled.
    """
    yield
    if MPI.COMM_WORLD.size > 1:
        gc.collect()


_SECOND_ORDER_ADJOINT_UNDERIVED = "second-order adjoint"


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    """Report an unsupported complex-scalar Hessian as a skip rather than a failure.

    Under a complex-scalar build the second-order adjoint has not been derived, and the
    Hessian entry points say so by raising rather than returning a number from a path nobody
    has checked. That is the intended behaviour, but it leaves a complex run unreadable: a
    test that asks for a Hessian is indistinguishable, in the summary, from a test of
    something that is actually broken.

    The tests that reach a Hessian are not confined to one file -- they span the Hessian
    suite, the TLM update suite, the linear and blocked solver suites, solver reuse,
    interpolation and assembly -- and most reach one incidentally, through
    {py:func}`pyadjoint.taylor_test`'s rate-3 Hessian-corrected check rather than by asking
    for a Hessian in so many words. Recognising the refusal as it propagates keeps that list
    from having to be maintained by hand, and means a test stops being skipped the moment it
    no longer needs the unimplemented path. A test that asserts the refusal
    (``pytest.raises``) never reaches here, since its exception does not propagate.

    Real-scalar builds are unaffected: nothing raises this there.
    """
    try:
        return (yield)
    except NotImplementedError as refusal:
        if _SECOND_ORDER_ADJOINT_UNDERIVED not in str(refusal):
            raise
        pytest.skip(f"Unsupported under a complex-scalar build: {refusal}")


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
