"""Sesquilinear-phase prototype: gradient-check the Load-bearing block's Real-lifted
control fix (a linear-solve-plus-assembly block driven by a real-valued control lifted
into a zero-imaginary complex Function) against a finite-difference check.
"""

from mpi4py import MPI

import dolfinx
import numpy as np
import pyadjoint
import pytest
import ufl

from dolfinx_adjoint import Function, RealLifted, assemble_scalar
from dolfinx_adjoint.blocks._vector import _vector
from dolfinx_adjoint.solvers import LinearProblem

pytestmark = pytest.mark.skipif(
    not np.issubdtype(dolfinx.default_scalar_type, np.complexfloating),
    reason="RealLifted only applies to complex-PETSc builds (Sesquilinear phase).",
)


@pytest.fixture
def mesh():
    return dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 20)


@pytest.fixture
def V(mesh):
    return dolfinx.fem.functionspace(mesh, ("Lagrange", 1))


def _boundary_dofs(mesh, V):
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    return dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)


def _solve_helmholtz(mesh, V, f: Function) -> Function:
    """The Load-bearing block: a complex sesquilinear Helmholtz solve driven by a
    Real-lifted forcing ``f``, exercising ``LinearProblemBlock.evaluate_adj_component``.
    """
    uh = Function(V, name="state")
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    k = 4.0
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - k**2 * ufl.inner(u, v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    bc_val = dolfinx.fem.Constant(mesh, np.dtype(dolfinx.default_scalar_type).type(0.0))
    bc = dolfinx.fem.dirichletbc(bc_val, _boundary_dofs(mesh, V), V)

    options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ksp_error_if_not_converged": True,
        "pc_factor_mat_solver_type": "mumps",
    }
    problem = LinearProblem(a, L, u=uh, bcs=[bc], petsc_options=options, adjoint_petsc_options=options)
    problem.solve()
    return uh


def test_real_lifted_rejects_non_complex_dtype(mesh, V):
    with pytest.raises(TypeError):
        RealLifted(V, dtype=np.float64)


def test_real_lifted_rejects_nonzero_imaginary_part(mesh, V):
    x = _vector(
        V.dofmap.index_map, V.dofmap.index_map_bs, dtype=dolfinx.default_scalar_type, function_space=V
    )
    x.array[:] = 1.0 + 1.0j
    with pytest.raises(ValueError):
        RealLifted(V, x=x)


def test_real_lifted_rejected_as_problem_unknown(mesh, V):
    g = RealLifted(V, name="not_a_control")
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(u, v) * ufl.dx
    forcing = dolfinx.fem.Constant(mesh, np.dtype(dolfinx.default_scalar_type).type(1.0))
    L = ufl.inner(forcing, v) * ufl.dx
    # Rejected at construction time (not just inside solve()), so that even a
    # solve(annotate=False) forward-only use can't silently corrupt a RealLifted's
    # leaf/zero-imaginary invariant by overwriting it with a solution.
    with pytest.raises(RuntimeError):
        LinearProblem(a, L, u=g)


def test_load_bearing_block_gradient_matches_finite_difference(mesh, V):
    pyadjoint.get_working_tape().clear_tape()
    # Random (not symmetric-about-midpoint) profiles: this domain/BC/operator is
    # reflection-symmetric about x=0.5, so a symmetric base point paired with an
    # antisymmetric perturbation (e.g. sin(pi x) and cos(pi x)) makes the true
    # first-order term vanish by parity -- accidentally validating a wrong gradient.
    rng = np.random.default_rng(0)

    f = RealLifted(V, name="control")
    f.x.array[:] = rng.uniform(-1.0, 1.0, size=f.x.array.shape)
    f.x.scatter_forward()

    uh = _solve_helmholtz(mesh, V, f)

    # J = Re(inner(uh, target)): a *holomorphic-in-uh* misfit against a fixed,
    # untracked target -- not |uh|**2 (= inner(uh, uh)), which differentiates
    # Conj(uh) as well as uh and trips UFL's holomorphic-derivative restriction
    # (ArityMismatch: 'v_0' vs 'conj(v_0)'). That non-holomorphic-in-state case is
    # explicitly out of scope for now (a future Wirtinger-calculus-aware adjoint).
    target = dolfinx.fem.Function(V)
    target.interpolate(lambda x: np.exp(x[0]))
    J = assemble_scalar(ufl.inner(uh, target) * ufl.dx)
    assert isinstance(J, float)

    control = pyadjoint.Control(f)
    Jhat = pyadjoint.ReducedFunctional(J, control)

    perturbation = RealLifted(V)
    perturbation.x.array[:] = rng.uniform(-1.0, 1.0, size=perturbation.x.array.shape)
    perturbation.x.scatter_forward()

    min_rate = pyadjoint.taylor_test(Jhat, f, perturbation, dJdm=0)
    assert np.isclose(min_rate, 1.0, rtol=1e-1, atol=1e-1), (
        f"Expected first-order (unconverged-gradient) rate close to 1.0, got {min_rate}"
    )

    # The Load-bearing block here is *exactly* affine in the control (a linear PDE with
    # the control only in the RHS, paired with a Functional that is linear in the state):
    # J(f + eps*h) is exactly linear in eps, with no curvature for a rate-2 Taylor
    # remainder to converge against -- pyadjoint.taylor_test's second-order check
    # degenerates to floating-point noise here ("taylor remainder is close to machine
    # precision"). A direct finite-difference comparison is the right check for an
    # affine problem: forward difference is exact but for roundoff, at any reasonable eps.
    dJdm_dot_h = complex(Jhat.derivative()._ad_dot(perturbation))
    assert np.isclose(dJdm_dot_h.imag, 0.0, atol=1e-10), (
        f"Gradient of a real Functional w.r.t. a RealLifted control should be real, got {dJdm_dot_h}"
    )
    eps = 1.0e-3
    f_plus = f._ad_copy()
    f_plus.x.array[:] = f.x.array[:] + eps * perturbation.x.array[:]
    finite_difference = (float(Jhat(f_plus)) - float(J)) / eps
    assert np.isclose(dJdm_dot_h.real, finite_difference, rtol=1e-6, atol=1e-8), (
        f"Analytic gradient {dJdm_dot_h.real} does not match finite difference {finite_difference}"
    )
