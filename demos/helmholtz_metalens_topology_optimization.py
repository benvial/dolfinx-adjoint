# # Topology optimization of a dielectric metalens (complex Helmholtz)
#
# This demo is the validation case for complex-valued PDE support in
# {py:mod}`dolfinx_adjoint` (see [issue #27][issue27]). It differentiates a **real-valued**
# objective through a **complex-valued**, sesquilinear PDE solve with respect to a
# **real-valued** control, and then runs a full topology optimization on top of that gradient.
#
# ```{admonition} This demo requires a complex PETSc build
# :class: warning
# Run it inside a DOLFINx container after sourcing `/usr/local/bin/dolfinx-complex-mode`.
# Under a real build the wave problem below has no imaginary part to speak of and the demo
# stops immediately. It is deliberately *not* listed in `_toc.yml` for that reason: the
# documentation is built in real mode.
# ```
#
# [issue27]: https://github.com/scientificcomputing/dolfinx-adjoint/issues/27
#
# ## Problem definition
#
# We design a flat dielectric lens that focuses a normally incident plane wave onto a small
# spot behind it. In two dimensions with TM polarization the only field component is the
# out-of-plane electric field $E_z$, and the time-harmonic Maxwell equations collapse to the
# scalar Helmholtz equation
#
# $$
# \nabla^2 E + k_0^2\,\varepsilon_r(\mathbf{x})\,E = 0,
# \qquad k_0 = \frac{2\pi}{\lambda},
# $$
#
# with an $e^{-i\omega t}$ time convention, so that outgoing waves behave like $e^{+ik_0 r}$.
#
# ### Scattered-field formulation
#
# Rather than solving for the total field we split it as $E = E_\mathrm{inc} + E_s$, where
# $E_\mathrm{inc} = e^{ik_0 x}$ is the incident plane wave, which already solves the equation
# for the background $\varepsilon_r = 1$. Subtracting the background equation leaves a problem
# for the scattered field alone,
#
# $$
# \nabla^2 E_s + k_0^2 \varepsilon_r E_s
# = -k_0^2\,(\varepsilon_r - 1)\,E_\mathrm{inc},
# $$
#
# whose source is supported only where the material differs from vacuum, i.e. inside the
# design region. This is what makes a plane-wave excitation cheap: no incident field has to be
# launched through a boundary, and $E_s$ is purely outgoing everywhere, so the same absorbing
# treatment works on all four sides.
#
# ### Perfectly matched layers
#
# The computational box is surrounded by a perfectly matched layer (PML), implemented as a
# complex coordinate stretch $\partial_x \mapsto s_x^{-1}\partial_x$ with
#
# $$
# s_x(x) = 1 + i\,\sigma\left(\frac{\max(|x| - x_\mathrm{phys},\,0)}{d_\mathrm{PML}}\right)^2,
# $$
#
# and likewise for $s_y$. Under the $e^{-i\omega t}$ convention a positive imaginary part makes
# $e^{ik_0\tilde{x}}$ decay, so an outgoing wave is absorbed instead of reflected. Folding the
# stretch into the weak form gives the sesquilinear problem: find $E_s \in V$ such that
#
# $$
# \int_\Omega (\mathbf{A}\nabla E_s)\cdot\overline{\nabla v}~\mathrm{d}x
# - k_0^2 \int_\Omega c\, E_s \overline{v}~\mathrm{d}x
# - k_0^2 \int_{\Omega_d} (\varepsilon_r - 1) E_s \overline{v}~\mathrm{d}x
# = k_0^2 \int_{\Omega_d} (\varepsilon_r - 1) E_\mathrm{inc} \overline{v}~\mathrm{d}x
# $$
#
# for all $v \in V$, with
#
# $$
# \mathbf{A} = \operatorname{diag}\left(\frac{s_y}{s_x}, \frac{s_x}{s_y}\right),
# \qquad c = s_x s_y,
# $$
#
# and $E_s = 0$ on the outer boundary, which the PML has already made unreachable. Note that
# $\mathbf{A}$ and $c$ are genuinely complex: this is not a real operator that merely happens
# to be stored in a complex dtype.
#
# ### Design parametrization
#
# The design region $\Omega_d$ carries a density $\rho \in [0,1]$, turned into a permittivity
# by the three standard topology-optimization steps:
#
# 1. **Filtering.** A Helmholtz filter of radius $r$ imposes a minimum length scale,
#    $-r^2\nabla^2\tilde\rho + \tilde\rho = \rho$. This is a *second* PDE solve, recorded on
#    the same tape, so the adjoint has to propagate back through it into the control. The radius
#    only buys a length scale if it is comfortably larger than the mesh width: at $r \approx h$
#    the filter is a one-element smoother and the design ends up resolution-limited instead.
# 2. **Projection.** A smoothed Heaviside pushes $\tilde\rho$ towards 0 or 1,
#
#    $$
#    \bar\rho = \frac{\tanh(\beta\eta) + \tanh(\beta(\tilde\rho - \eta))}
#                    {\tanh(\beta\eta) + \tanh(\beta(1 - \eta))},
#    $$
#
#    with the sharpness $\beta$ raised in stages (continuation) and $\eta = 0.5$.
# 3. **SIMP.** The permittivity interpolates between vacuum and the lens material,
#    $\varepsilon_r(\bar\rho) = 1 + \bar\rho^{\,p}(\varepsilon_\mathrm{mat} - 1)$, with the
#    penalization exponent $p$ biasing the permittivity of an intermediate density downwards.
#    Unlike the compliance problem this exponent was invented for, SIMP on a permittivity does
#    not on its own make grey unprofitable in a wave problem; here it is the $\beta$
#    continuation that drives the design binary, and $M_\mathrm{nd}$ below measures whether
#    it did.
#
# ### Objective
#
# We maximize the mean intensity over a small circular focal spot $\Omega_f$ behind the lens,
# normalized by the incident intensity (which is 1), i.e. we minimize
#
# $$
# J(\rho) = -\frac{1}{|\Omega_f|}\int_{\Omega_f} |E_\mathrm{inc} + E_s|^2~\mathrm{d}x.
# $$
#
# $|E|^2$ is neither holomorphic nor anti-holomorphic in the state, so the adjoint seed for
# this Functional carries both Wirtinger derivatives; $\rho$, by contrast, is a Real control,
# whose gradient is the doubled real part $2\,\mathrm{Re}[\lambda^H \partial R/\partial\rho]$.
#
# ## Implementation

# +
import time

from mpi4py import MPI

import dolfinx
import matplotlib.pyplot as plt
import matplotlib.tri
import numpy as np
import pyadjoint
import scipy.optimize
import ufl

import dolfinx_adjoint

# -

# The whole point of the demo is the complex scalar type, so we refuse to pretend otherwise.

if not np.issubdtype(dolfinx.default_scalar_type, np.complexfloating):
    raise RuntimeError(
        "This demo needs a complex-scalar DOLFINx build; source /usr/local/bin/dolfinx-complex-mode before running it."
    )

# ## Parameters
#
# Lengths are in units of the free-space wavelength $\lambda$, so $k_0 = 2\pi$.

# +
WAVELENGTH = 1.0
K0 = 2.0 * np.pi / WAVELENGTH

PML_WIDTH = 0.6  # absorbing-layer thickness on every side
PML_STRENGTH = 4.0  # peak of the quadratic conductivity profile
X_PHYS, Y_PHYS = 2.0, 1.6  # half-extent of the physical (non-PML) region
LX, LY = X_PHYS + PML_WIDTH, Y_PHYS + PML_WIDTH  # half-extent of the whole box

DESIGN_BOX = (-0.3, 0.3, -1.2, 1.2)  # (xmin, xmax, ymin, ymax) of the lens slab
FOCUS_CENTRE, FOCUS_RADIUS = (1.2, 0.0), 0.15  # the spot whose intensity we maximize

EPS_MATERIAL = 4.0  # relative permittivity of the solid phase (refractive index 2)
PENALIZATION = 3.0  # SIMP exponent
FILTER_RADIUS = 0.10  # Helmholtz-filter length scale; see the note on mesh resolution below
ETA = 0.5  # projection threshold
BETA_STAGES = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)  # projection-sharpness continuation
ITERATIONS_PER_STAGE = 30

CELLS_PER_WAVELENGTH = 24  # cell width h is then about 0.042, so FILTER_RADIUS is ~2.4 h

DESIGN_TAG, FOCUS_TAG, BULK_TAG = 1, 2, 3

PETSC_LU = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "ksp_error_if_not_converged": True,
}
# -

# ## Mesh and subdomains
#
# The design slab and the focal spot are marked by cell, which is why the circle is only
# resolved to the accuracy of the mesh. Both are used purely as integration subdomains.

# +
mesh = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([-LX, -LY]), np.array([LX, LY])],
    (int(2 * LX * CELLS_PER_WAVELENGTH), int(2 * LY * CELLS_PER_WAVELENGTH)),
    dolfinx.mesh.CellType.triangle,
)
tdim = mesh.topology.dim

design_cells = dolfinx.mesh.locate_entities(
    mesh,
    tdim,
    lambda x: (x[0] > DESIGN_BOX[0]) & (x[0] < DESIGN_BOX[1]) & (x[1] > DESIGN_BOX[2]) & (x[1] < DESIGN_BOX[3]),
)
focus_cells = dolfinx.mesh.locate_entities(
    mesh,
    tdim,
    lambda x: (x[0] - FOCUS_CENTRE[0]) ** 2 + (x[1] - FOCUS_CENTRE[1]) ** 2 < FOCUS_RADIUS**2,
)
index_map = mesh.topology.index_map(tdim)
markers = np.full(index_map.size_local + index_map.num_ghosts, BULK_TAG, dtype=np.int32)
markers[design_cells] = DESIGN_TAG
markers[focus_cells] = FOCUS_TAG
cell_tags = dolfinx.mesh.meshtags(mesh, tdim, np.arange(markers.size, dtype=np.int32), markers)
dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)

print(f"{mesh.topology.index_map(tdim).size_global} cells, {len(design_cells)} in the design region")
# -

# ## PML and incident field
#
# The stretch factors are 1 in the physical region and grow quadratically into the layer, so
# `ufl.max_value` does the clamping symbolically rather than through a marked subdomain.

# +
x = ufl.SpatialCoordinate(mesh)


def stretch(coordinate, half_width):
    """Complex coordinate-stretch factor for a PML of thickness `PML_WIDTH`."""
    depth = ufl.max_value(abs(coordinate) - half_width, 0.0) / PML_WIDTH
    return 1.0 + 1j * PML_STRENGTH * depth**2


s_x, s_y = stretch(x[0], X_PHYS), stretch(x[1], Y_PHYS)
pml_tensor = ufl.as_matrix([[s_y / s_x, 0], [0, s_x / s_y]])
pml_scale = s_x * s_y

incident = ufl.exp(1j * K0 * x[0])
# -

# ## Control, filter and projection
#
# `beta` is a plain {py:class}`dolfinx.fem.Constant` rather than a control: the recorded blocks
# read its value at replay time, so the continuation loop below can raise it without
# re-recording the tape.

# +
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 2))  # scattered field
Q = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))  # density

rho = dolfinx_adjoint.Function(Q, name="density")
# Grey inside the slab, vacuum outside it, which is where the optimization starts as well: the
# reported baseline then describes the same design the optimizer is handed, and the filter sees
# a clean vacuum boundary around the lens.
mesh.topology.create_connectivity(tdim, tdim)
design_dofs = dolfinx.fem.locate_dofs_topological(Q, tdim, design_cells)
rho.x.array[:] = 0.0
rho.x.array[design_dofs] = 0.5
rho.x.scatter_forward()

trial_rho, w = ufl.TrialFunction(Q), ufl.TestFunction(Q)
rho_filtered = dolfinx_adjoint.Function(Q, name="filtered_density")
filter_problem = dolfinx_adjoint.LinearProblem(
    (FILTER_RADIUS**2 * ufl.inner(ufl.grad(trial_rho), ufl.grad(w)) + ufl.inner(trial_rho, w)) * ufl.dx,
    ufl.inner(rho, w) * ufl.dx,
    u=rho_filtered,
    petsc_options=PETSC_LU,
    adjoint_petsc_options=PETSC_LU,
)
filter_problem.solve()

beta = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(BETA_STAGES[0]))
projected = (ufl.tanh(beta * ETA) + ufl.tanh(beta * (rho_filtered - ETA))) / (
    ufl.tanh(beta * ETA) + ufl.tanh(beta * (1 - ETA))
)
permittivity = 1.0 + projected**PENALIZATION * (EPS_MATERIAL - 1.0)
# -

# ## Forward solve
#
# The design-region terms are integrated over `dx(DESIGN_TAG)` alone, so the density can only
# ever place material inside the slab, whatever it does elsewhere.

# +
scattered = dolfinx_adjoint.Function(V, name="scattered_field")
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

a = (ufl.inner(pml_tensor * ufl.grad(u), ufl.grad(v)) - K0**2 * pml_scale * ufl.inner(u, v)) * dx - K0**2 * ufl.inner(
    (permittivity - 1.0) * u, v
) * dx(DESIGN_TAG)
L = K0**2 * ufl.inner((permittivity - 1.0) * incident, v) * dx(DESIGN_TAG)

mesh.topology.create_connectivity(tdim - 1, tdim)
outer_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
bc = dolfinx.fem.dirichletbc(
    dolfinx.default_scalar_type(0.0),
    dolfinx.fem.locate_dofs_topological(V, tdim - 1, outer_facets),
    V,
)

problem = dolfinx_adjoint.LinearProblem(
    a, L, u=scattered, bcs=[bc], petsc_options=PETSC_LU, adjoint_petsc_options=PETSC_LU
)
problem.solve()
# -

# ## Objective
#
# {py:func}`dolfinx_adjoint.assemble_scalar` takes the real part of a rank-0 form, so `J` is a
# real `AdjFloat` even though every field in it is complex.

# +
focus_area = dolfinx_adjoint.assemble_scalar(1.0 * dx(FOCUS_TAG), annotate=False)
design_area = dolfinx_adjoint.assemble_scalar(1.0 * dx(DESIGN_TAG), annotate=False)
J = -dolfinx_adjoint.assemble_scalar(ufl.inner(scattered + incident, scattered + incident) * dx(FOCUS_TAG)) / focus_area

Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(rho))
print(f"intensity enhancement of the grey starting design: {-float(J):.4f}")
# -

# ## Gradient verification
#
# The Taylor remainder $|J(\rho + \epsilon h) - J(\rho)|$ is $\mathcal{O}(\epsilon)$, and
# subtracting the adjoint gradient's contribution leaves $\mathcal{O}(\epsilon^2)$. Rates of 1
# and 2 are what make this demo a validation case rather than a picture.
#
# Second-order Taylor tests are deliberately absent: complex-mode Hessians raise
# {py:class}`NotImplementedError`, since nobody has derived that path yet.

# +
with pyadjoint.stop_annotating():
    direction = dolfinx_adjoint.Function(Q)
    direction.x.array[:] = np.random.default_rng(42).standard_normal(direction.x.array.size)
    direction.x.scatter_forward()

    rate_0 = pyadjoint.taylor_test(Jhat, rho, direction, dJdm=0)
    rate_1 = pyadjoint.taylor_test(Jhat, rho, direction)
    print(f"0th-order Taylor rate (expect ~1): {rate_0:.4f}")
    print(f"1st-order Taylor rate (expect ~2): {rate_1:.4f}")

    gradient = Jhat.derivative()
    print(f"max |Im(dJ/drho)| (expect 0): {np.abs(gradient.x.array.imag).max():.3e}")
# -

# ## Optimization with projection continuation
#
# `scipy.optimize.minimize` drives a {py:class}`pyadjoint.reduced_functional_numpy.ReducedFunctionalNumPy`.
# L-BFGS-B rather than the `trust-constr` of the
# [elastic topology optimization demo](./topology_optimization): that method wants Hessian-vector
# products, which the complex path does not provide.
#
# The density is bounded to $[0,1]$ inside the design slab and pinned to 0 outside it, so the
# filter sees a clean vacuum boundary condition around the lens.
#
# ```{note}
# The NumPy optimizer interface flattens the control into one array, so this loop is written
# for serial execution, as the elastic topology optimization demo is.
# ```

# +
num_owned = Q.dofmap.index_map.size_local * Q.dofmap.index_map_bs

lower = np.zeros(num_owned)
upper = np.zeros(num_owned)
upper[design_dofs[design_dofs < num_owned]] = 1.0

rf_np = pyadjoint.reduced_functional_numpy.ReducedFunctionalNumPy(Jhat)
history: list[float] = []

start = time.perf_counter()
for stage_beta in BETA_STAGES:
    beta.value = dolfinx.default_scalar_type(stage_beta)
    result = scipy.optimize.minimize(
        rf_np.__call__,
        rf_np.get_controls(),
        jac=lambda m: rf_np.derivative(),
        method="L-BFGS-B",
        bounds=scipy.optimize.Bounds(lower, upper),
        options={"maxiter": ITERATIONS_PER_STAGE, "maxcor": 20},
    )
    rho.x.array[:num_owned] = result.x
    rho.x.scatter_forward()
    history.append(-result.fun)
    print(f"beta = {stage_beta:5.1f}: enhancement {-result.fun:8.4f} after {result.nit} iterations")

print(f"optimization took {time.perf_counter() - start:.1f} s")
# -

# ## Binarization
#
# Continuation drives the *projected* density towards 0/1, which the measure of
# non-discreteness $M_\mathrm{nd} = \frac{4}{|\Omega_d|}\int_{\Omega_d}\bar\rho(1-\bar\rho)$
# quantifies: 1 for an all-grey design, 0 for a fully binary one. Thresholding the design at
# $\rho = 0.5$ and re-evaluating shows what that grey was still buying.

# +
with pyadjoint.stop_annotating():
    continuous = -Jhat(rho)
    non_discreteness = (
        dolfinx_adjoint.assemble_scalar(4.0 * projected * (1.0 - projected) * dx(DESIGN_TAG), annotate=False)
        / design_area
    )

    binary = dolfinx_adjoint.Function(Q, name="binarized_density")
    binary.x.array[:] = (rho.x.array.real > 0.5).astype(float)
    binarized = -Jhat(binary)

print(f"measure of non-discreteness: {non_discreteness:.4f}")
print(f"enhancement, continuous design: {continuous:.4f}")
print(f"enhancement, binarized design:  {binarized:.4f}")
# -

# ## Results
#
# We plot the final design and the intensity it produces. The scattered field is quadratic, so
# it is interpolated into the linear density space for plotting.

# +
topology, _, geometry = dolfinx.plot.vtk_mesh(Q)
triangulation = matplotlib.tri.Triangulation(geometry[:, 0], geometry[:, 1], topology.reshape(-1, 4)[:, 1:])


def nodal_values(expression):
    """Interpolate a UFL expression into `Q` and return its real nodal values."""
    out = dolfinx.fem.Function(Q)
    out.interpolate(dolfinx.fem.Expression(expression, Q.element.interpolation_points))
    return out.x.array.real


def draw_geometry(axis):
    axis.add_patch(
        plt.Rectangle(
            (DESIGN_BOX[0], DESIGN_BOX[2]),
            DESIGN_BOX[1] - DESIGN_BOX[0],
            DESIGN_BOX[3] - DESIGN_BOX[2],
            fill=False,
            edgecolor="white",
            linewidth=0.8,
        )
    )
    axis.add_patch(plt.Circle(FOCUS_CENTRE, FOCUS_RADIUS, fill=False, edgecolor="white", linewidth=0.8))
    axis.set_xlim(-X_PHYS, X_PHYS)
    axis.set_ylim(-Y_PHYS, Y_PHYS)
    axis.set_aspect("equal")


Jhat(rho)  # leave the tape holding the optimized design
intensity = nodal_values(ufl.inner(scattered + incident, scattered + incident))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
design_plot = axes[0].tripcolor(triangulation, nodal_values(projected), cmap="binary", vmin=0.0, vmax=1.0)
axes[0].set_title(r"optimized design $\bar\rho$")
fig.colorbar(design_plot, ax=axes[0])
field_plot = axes[1].tripcolor(triangulation, intensity, cmap="inferno")
axes[1].set_title(r"$|E_\mathrm{inc} + E_s|^2$")
fig.colorbar(field_plot, ax=axes[1])
for axis in axes:
    draw_geometry(axis)
fig.savefig("metalens_design_and_field.png", dpi=150, bbox_inches="tight")
plt.show()
# -

# The enhancement per continuation stage, with the stage boundaries marked.

# +
fig, axis = plt.subplots(figsize=(7, 4))
axis.plot(history, marker="o", markersize=3)
axis.set_xticks(range(len(BETA_STAGES)), [f"{b:g}" for b in BETA_STAGES])
axis.set_xlabel(r"projection sharpness $\beta$")
axis.set_ylabel("intensity enhancement at the focus")
axis.set_title("Metalens topology optimization")
axis.grid(alpha=0.3)
fig.savefig("metalens_convergence.png", dpi=150, bbox_inches="tight")
plt.show()
# -

# Both `LinearProblem`s are kept alive until here: pyadjoint replays their blocks during every
# adjoint sweep, and a garbage-collected one has to be rebuilt.

del problem, filter_problem
