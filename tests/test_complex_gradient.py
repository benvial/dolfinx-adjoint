"""Gradients of a real-valued Functional through a complex-valued (sesquilinear) PDE solve.

Under a complex-PETSc build a Control is real-valued -- complex-valued controls are out of
scope -- but is represented as a complex Function whose imaginary part is identically zero,
which is simply what constructing one with the default dtype gives. The gradient of a real
Functional with respect to such a control is ``2*Re[lambda^H . dR/dm]``, and that extraction
happens in exactly one place: ``Function._ad_convert_riesz``, which pyadjoint calls only on a
Control's own object. These tests pin that down against finite differences.
"""

from mpi4py import MPI

import dolfinx
import numpy as np
import pyadjoint
import pytest
import ufl

from dolfinx_adjoint import Constant, Function, assemble_scalar, interpolate
from dolfinx_adjoint.solvers import LinearProblem

complex_only = pytest.mark.skipif(
    not np.issubdtype(dolfinx.default_scalar_type, np.complexfloating),
    reason="Exercises the complex-scalar adjoint path.",
)

pytestmark = complex_only

_PETSC_LU = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "ksp_error_if_not_converged": True,
    "pc_factor_mat_solver_type": "mumps",
}


@pytest.fixture
def mesh():
    return dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 20)


@pytest.fixture
def V(mesh):
    return dolfinx.fem.functionspace(mesh, ("Lagrange", 1))


@pytest.fixture
def target(V):
    """A fixed, untracked profile to build a misfit against.

    Deliberately complex-valued: a real-valued target equals its own conjugate, which hides
    every error that consists of conjugating (or failing to conjugate) the adjoint seed.
    """
    t = dolfinx.fem.Function(V)
    t.interpolate(lambda x: np.exp(x[0]) + 1j * np.sin(x[0]))
    return t


def _solve_helmholtz(mesh, V, source) -> tuple[LinearProblem, Function]:
    """A complex sesquilinear Helmholtz solve driven by ``source``.

    ``k`` is given a small imaginary part so the operator is genuinely complex (damped)
    rather than a real operator that merely happens to be stored in a complex dtype.
    """
    uh = Function(V, name="state")
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    k = dolfinx.default_scalar_type(4.0 + 0.5j)  # type: ignore[arg-type]
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - k**2 * ufl.inner(u, v)) * ufl.dx
    L = ufl.inner(source, v) * ufl.dx

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    bc = dolfinx.fem.dirichletbc(
        dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0)),
        dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, facets),
        V,
    )

    problem = LinearProblem(a, L, u=uh, bcs=[bc], petsc_options=_PETSC_LU, adjoint_petsc_options=_PETSC_LU)
    problem.solve()
    # Returned, not dropped: pyadjoint replays this block during the adjoint sweep, and a
    # garbage-collected LinearProblem forces it to rebuild an equivalent one.
    return problem, uh


def _central_difference(Jhat, m, perturbation, eps=1.0e-3):
    """Central difference of ``Jhat`` at ``m`` along ``perturbation``.

    Used where the Functional is quadratic in the control rather than affine -- a misfit that
    is not linear in the state -- for which a central difference is still exact but for
    roundoff, while the forward difference below carries an O(eps) error.
    """
    values = []
    for step in (eps, -eps):
        m_step = m._ad_copy()
        m_step.x.array[:] = m.x.array[:] + step * perturbation.x.array[:]
        values.append(float(Jhat(m_step)))
    return (values[0] - values[1]) / (2 * eps)


def _finite_difference(Jhat, m, perturbation, J0, eps=1.0e-3):
    """Forward difference of ``Jhat`` at ``m`` along ``perturbation``.

    The problems here are exactly affine in the control (a linear PDE with the control in the
    right-hand side only, paired with a Functional linear in the state), so ``J(m + eps*h)``
    is exactly linear in ``eps``. That leaves no curvature for ``taylor_test``'s second-order
    remainder to converge against -- it degenerates into floating-point noise -- while a
    forward difference is exact but for roundoff at any reasonable step size.
    """
    m_plus = m._ad_copy()
    m_plus.x.array[:] = m.x.array[:] + eps * perturbation.x.array[:]
    return (float(Jhat(m_plus)) - float(J0)) / eps


def test_function_control_gradient_matches_finite_difference(mesh, V, target):
    pyadjoint.get_working_tape().clear_tape()
    # A random (not reflection-symmetric) profile: this domain, BC and operator are symmetric
    # about x=0.5, so a symmetric base point with an antisymmetric perturbation would make the
    # true first-order term vanish by parity and accidentally validate a wrong gradient.
    rng = np.random.default_rng(0)

    f = Function(V, name="control")
    f.x.array[:] = rng.uniform(-1.0, 1.0, size=f.x.array.shape)
    f.x.scatter_forward()

    problem, uh = _solve_helmholtz(mesh, V, f)

    # A holomorphic-in-state misfit against a fixed target: ufl.inner conjugates its *second*
    # argument, so inner(uh, target) is holomorphic in uh. The anti-holomorphic and the
    # genuinely non-holomorphic shapes are covered below.
    J = assemble_scalar(ufl.inner(uh, target) * ufl.dx)
    assert isinstance(J, float)

    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(f))

    h = Function(V)
    h.x.array[:] = rng.uniform(-1.0, 1.0, size=h.x.array.shape)
    h.x.scatter_forward()

    dJdm = Jhat.derivative()._ad_dot(h)
    assert isinstance(dJdm, float), f"_ad_dot should return a real scalar, got {type(dJdm)}"
    assert np.isclose(dJdm, _finite_difference(Jhat, f, h, J), rtol=1e-6, atol=1e-8)
    del problem


def test_gradient_of_real_functional_is_real(mesh, V, target):
    pyadjoint.get_working_tape().clear_tape()
    rng = np.random.default_rng(1)

    f = Function(V, name="control")
    f.x.array[:] = rng.uniform(-1.0, 1.0, size=f.x.array.shape)
    f.x.scatter_forward()

    problem, uh = _solve_helmholtz(mesh, V, f)
    J = assemble_scalar(ufl.inner(uh, target) * ufl.dx)
    grad = pyadjoint.ReducedFunctional(J, pyadjoint.Control(f)).derivative()

    # The gradient is mathematically real, but stays in the control's complex dtype so that
    # `control + step * gradient` in an optimisation loop does not mix dtypes.
    assert np.issubdtype(grad.x.array.dtype, np.complexfloating)
    assert np.allclose(grad.x.array.imag, 0.0, atol=1e-12), f"max |imag| = {np.abs(grad.x.array.imag).max()}"
    del problem


def test_constant_control_gradient_matches_finite_difference(mesh, V, target):
    """A `Constant` control, which the previous Block-level gating never reached.

    `Constant` inherits `_ad_convert_riesz` from `Function` unchanged, so putting the
    extraction there covers it with no `Constant`-specific code.
    """
    pyadjoint.get_working_tape().clear_tape()

    alpha = Constant(mesh, 2.0)
    f = Function(V, name="source")
    f.interpolate(lambda x: np.sin(np.pi * x[0]) + 0.3 * x[0])

    problem, uh = _solve_helmholtz(mesh, V, alpha * f)
    J = assemble_scalar(ufl.inner(uh, target) * ufl.dx)

    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(alpha))

    h = alpha._ad_copy()
    h.x.array[:] = 1.0
    dJdm = Jhat.derivative()._ad_dot(h)
    assert np.isclose(dJdm, _finite_difference(Jhat, alpha, h, J), rtol=1e-6, atol=1e-8)
    del problem


def test_anti_holomorphic_functional_gradient_matches_finite_difference(mesh, V, target):
    """`inner(target, uh)` is anti-holomorphic in `uh`, since `ufl.inner` conjugates its
    *second* argument -- so argument order alone decides whether a misfit is holomorphic in
    the state, a slip a user cannot see. Both orders have to give the right gradient.
    """
    pyadjoint.get_working_tape().clear_tape()
    rng = np.random.default_rng(2)

    f = Function(V, name="control")
    f.x.array[:] = rng.uniform(-1.0, 1.0, size=f.x.array.shape)
    f.x.scatter_forward()

    problem, uh = _solve_helmholtz(mesh, V, f)
    J = assemble_scalar(ufl.inner(target, uh) * ufl.dx)
    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(f))

    h = Function(V)
    h.x.array[:] = rng.uniform(-1.0, 1.0, size=h.x.array.shape)
    h.x.scatter_forward()

    dJdm = Jhat.derivative()._ad_dot(h)
    assert np.isclose(dJdm, _finite_difference(Jhat, f, h, J), rtol=1e-6, atol=1e-8)
    del problem


def test_non_holomorphic_misfit_gradient_matches_finite_difference(mesh, V, target):
    """The standard misfit `|uh - target|**2`, which is neither holomorphic nor
    anti-holomorphic in the state: `uh` appears conjugated in some terms and not in others.

    Its derivative is not determined by the holomorphic part alone, so the adjoint seed has to
    carry both Wirtinger derivatives -- which is why the seed is assembled from a derivative in
    the real direction *and* one in the imaginary direction.
    """
    pyadjoint.get_working_tape().clear_tape()
    rng = np.random.default_rng(3)

    f = Function(V, name="control")
    f.x.array[:] = rng.uniform(-1.0, 1.0, size=f.x.array.shape)
    f.x.scatter_forward()

    problem, uh = _solve_helmholtz(mesh, V, f)
    J = assemble_scalar(ufl.inner(uh - target, uh - target) * ufl.dx)
    assert isinstance(J, float)

    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(f))

    h = Function(V)
    h.x.array[:] = rng.uniform(-1.0, 1.0, size=h.x.array.shape)
    h.x.scatter_forward()

    dJdm = Jhat.derivative()._ad_dot(h)
    assert np.isclose(dJdm, _central_difference(Jhat, f, h), rtol=1e-6, atol=1e-10)
    del problem


def test_gradient_is_unchanged_by_the_order_of_a_squared_misfit(mesh, V, target):
    """`inner(a, b)` and `inner(b, a)` are conjugates of one another, so a misfit built from
    either order has the same real part -- and therefore the same gradient. A seed that
    conjugates one Wirtinger part but not the other would split them apart.
    """
    pyadjoint.get_working_tape().clear_tape()
    rng = np.random.default_rng(4)

    gradients = []
    for order in (lambda a, b: ufl.inner(a, b), lambda a, b: ufl.inner(b, a)):
        pyadjoint.get_working_tape().clear_tape()
        f = Function(V, name="control")
        f.x.array[:] = rng.uniform(-1.0, 1.0, size=f.x.array.shape)
        f.x.scatter_forward()
        problem, uh = _solve_helmholtz(mesh, V, f)
        J = assemble_scalar(order(uh, target) * ufl.dx)
        gradients.append(pyadjoint.ReducedFunctional(J, pyadjoint.Control(f)).derivative().x.array.copy())
        del problem
        rng = np.random.default_rng(4)

    assert np.allclose(gradients[0], gradients[1])


def test_gradient_through_an_interpolation_step_matches_finite_difference(mesh, V, target):
    """A Control reaching the Functional through an interpolation into another space.

    Interpolation between two spaces is a real linear map, but the field it is applied to is
    complex, and the matrix DOLFINx builds for it is real-valued even under a complex build --
    so the adjoint of this step has to apply a real operator to a complex vector without
    dropping half of it.
    """
    pyadjoint.get_working_tape().clear_tape()
    rng = np.random.default_rng(5)

    W = dolfinx.fem.functionspace(mesh, ("Lagrange", 2))

    f = Function(V, name="control")
    f.x.array[:] = rng.uniform(-1.0, 1.0, size=f.x.array.shape)
    f.x.scatter_forward()

    problem, uh = _solve_helmholtz(mesh, V, f)
    # The state is genuinely complex here (a damped Helmholtz solve), so an interpolation
    # that dropped its imaginary part would still produce a plausible number.
    target_W = dolfinx.fem.Function(W)
    target_W.interpolate(target)

    interpolated = interpolate(uh, W)
    J = assemble_scalar(ufl.inner(interpolated, target_W) * ufl.dx)

    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(f))

    h = Function(V)
    h.x.array[:] = rng.uniform(-1.0, 1.0, size=h.x.array.shape)
    h.x.scatter_forward()

    dJdm = Jhat.derivative()._ad_dot(h)
    assert np.isclose(dJdm, _finite_difference(Jhat, f, h, J), rtol=1e-6, atol=1e-8)
    del problem


def test_hessian_is_refused_under_complex_scalars(mesh, V, target):
    pyadjoint.get_working_tape().clear_tape()

    f = Function(V, name="control")
    f.x.array[:] = 1.0
    problem, uh = _solve_helmholtz(mesh, V, f)
    J = assemble_scalar(ufl.inner(uh, target) * ufl.dx)
    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(f))

    h = Function(V)
    h.x.array[:] = 1.0
    Jhat.derivative()
    with pytest.raises(NotImplementedError, match="second-order adjoint"):
        Jhat.hessian(h)
    del problem


def test_assemble_scalar_accepts_a_compiled_form(mesh, V):
    """`assemble_scalar` is the single rank-0 path, so it must take a compiled form too."""
    u = dolfinx.fem.Function(V)
    u.x.array[:] = 2.0
    compiled = dolfinx.fem.form(ufl.inner(u, u) * ufl.dx)

    value = assemble_scalar(compiled, annotate=False)
    assert isinstance(value, float)
    assert np.isclose(value, 4.0)

    with pytest.raises(ValueError, match="already-compiled"):
        assemble_scalar(compiled)


def test_mixed_scalar_type_form_names_the_offending_coefficient(mesh, V):
    """A real/complex dtype mix must say so, not fail deep in the nanobind bindings."""
    complex_fn = dolfinx.fem.Function(V, name="complex_one")
    complex_fn.x.array[:] = 1.0
    real_fn = dolfinx.fem.Function(V, name="real_one", dtype=np.float64)
    real_fn.x.array[:] = 1.0

    with pytest.raises((TypeError, RuntimeError), match="mixes real- and complex-dtype"):
        assemble_scalar(ufl.inner(complex_fn, real_fn) * ufl.dx, annotate=False)
