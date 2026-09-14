import contextlib
import typing
from functools import singledispatchmethod

import dolfinx
import numpy
import numpy.typing as npt
import ufl
from ufl.algorithms.check_arities import ArityChecker, ArityMismatch
from ufl.algorithms.map_integrands import map_integrands
from ufl.corealg.dag_traverser import DAGTraverser
from ufl.corealg.map_dag import map_expr_dag

from .compat import extract_linear_combination


def function_from_vector(
    V: dolfinx.fem.FunctionSpace,
    vector: typing.Union[
        dolfinx.la.Vector,
        dolfinx.cpp.la.Vector_float32,
        dolfinx.cpp.la.Vector_float64,
        dolfinx.cpp.la.Vector_complex64,
        dolfinx.cpp.la.Vector_complex128,
        dolfinx.cpp.la.Vector_int8,
        dolfinx.cpp.la.Vector_int32,
        dolfinx.cpp.la.Vector_int64,
    ],
) -> dolfinx.fem.Function:
    """Create a new Function from a vector.

    Arguments:
        V: The function space
        vector: The vector data.
    Returns:
        A new {py:class}`dolfinx.fem.Function` instance that has been assigned the
        values from the vector (deep-copy)
    """
    ret = dolfinx.fem.Function(V, dtype=vector.array.dtype)
    ret.x.array[:] = vector.array[:]
    return ret


def gather(vector: dolfinx.la.Vector) -> npt.NDArray[numpy.number]:
    """Gather a vector on all processes.

    Args:
        vector: The vector to gather.
    Returns:
        A numpy array containing the gathered vector.
    """
    local_size = vector.index_map.size_local * vector.block_size
    comm = vector.index_map.comm
    data = comm.allgather(vector.array[:local_size])
    return numpy.hstack(data)


class ad_kwargs(typing.TypedDict):
    ad_block_tag: typing.NotRequired[str]
    """Tag for the block in the adjoint tape."""
    annotate: typing.NotRequired[bool]
    """Whether to annotate the assignment in the adjoint tape."""


def assign_linear_combination(value: ufl.core.expr.Expr, function: dolfinx.fem.Function) -> None:
    """Assign a linear combination of functions to a function.

    Arguments:
        value: A linear combination of functions, e.g. `2*u + 3*v`.
        function: The function to assign the linear combination to.
    """
    pairs = extract_linear_combination(value)
    function.x.array[:] = 0.0
    floatifier = Floatify()
    for weight, func in pairs:
        # extract_linear_combination is typed against UFL, which knows nothing of degrees of
        # freedom; assigning a linear combination needs the DOLFINx Function that carries them.
        if not isinstance(func, dolfinx.fem.Function):
            raise TypeError(f"Expected the linear combination to be over dolfinx Functions, got {type(func)}.")
        if not func.function_space == function.function_space:
            raise ValueError("Function spaces of all functions in the linear combination must match for assignment.")
        function.x.array[:] += floatifier.process(weight) * func.x.array[:]
    function.x.scatter_forward()


class Floatify(DAGTraverser):
    """Traverser to convert a UFL expression into a float."""

    def __init__(self, **kwargs):
        """Convert a ufl expression into a float"""
        super().__init__(**kwargs)

    @singledispatchmethod
    def process(self, o: ufl.classes.Expr, **kwargs):
        return float(o)

    @process.register(dolfinx.fem.Function)
    def _(self, o, **kwargs):
        if ufl.checks.is_scalar_constant_expression(o):
            return o.x.array[0]
        raise NotImplementedError(f"Unsupported UFL node type for floatification: {type(o)}")

    @process.register(ufl.classes.Sum)
    @DAGTraverser.postorder
    def _(self, o, *operands, **kwargs):
        # operands is a tuple of the already-floatified children
        return sum(operands)

    @process.register(ufl.classes.Division)
    @DAGTraverser.postorder
    def _(self, o, *operands, **kwargs):
        # Division always has exactly two operands: numerator and denominator
        return operands[0] / operands[1]

    @process.register(ufl.classes.Power)
    @DAGTraverser.postorder
    def _(self, o, *operands, **kwargs):
        # Power has exactly two operands: base and exponent
        return operands[0] ** operands[1]

    @process.register(ufl.classes.Product)
    @DAGTraverser.postorder
    def _(self, o, *operands, **kwargs):
        # Product has exactly two operands: left and right
        return operands[0] * operands[1]


def scalar_type_mismatch_message(form: typing.Any) -> str | None:
    """Describe a real/complex dtype mismatch among a form's coefficients, if there is one.

    Accepts a single form or any nesting of forms -- a blocked problem's ``a`` is a list of
    lists, whose entries may be ``None`` -- and considers every coefficient across them, since
    it is their coexistence in one assembly that fails.

    DOLFINx cannot assemble a form whose coefficients do not share one scalar dtype: its
    nanobind bindings are templated on a fixed set of ``(scalar, geometry)`` dtype pairs, so
    a mismatch surfaces several layers down as a template-resolution failure naming neither
    the dtype nor the coefficient responsible.

    Under a complex-scalar build every {py:class}`~dolfinx_adjoint.Function` and
    {py:class}`~dolfinx_adjoint.Constant` built with the default dtype is already complex, so
    a mismatch is reached only when a caller explicitly asked for a real dtype, or passed a
    raw {py:class}`dolfinx.fem.Constant` built from a Python float. This is called only from
    the failure path, so walking the coefficients costs nothing in the normal case.

    Returns:
        A message naming each dtype and the coefficients carrying it, or ``None`` if the
        coefficients agree (in which case the original error was about something else).
    """
    by_dtype: dict[numpy.dtype, list[str]] = {}

    def record(obj) -> None:
        values = getattr(getattr(obj, "x", None), "array", None)
        if values is None:
            values = getattr(obj, "value", None)
        if values is None:
            return
        dtype = numpy.asarray(values).dtype
        if not numpy.issubdtype(dtype, numpy.number):
            return
        name = getattr(obj, "name", None) or repr(obj)
        by_dtype.setdefault(dtype, []).append(str(name))

    def record_form(candidate) -> None:
        if candidate is None:
            return
        if not isinstance(candidate, ufl.Form):
            for nested in candidate:
                record_form(nested)
            return
        for coefficient in ufl.algorithms.extract_coefficients(candidate):
            record(coefficient)
        try:
            constants = ufl.algorithms.analysis.extract_constants(candidate)
        except AttributeError:  # pragma: no cover - depends on the installed UFL version
            constants = []
        for constant in constants:
            record(constant)

    record_form(form)

    kinds = {numpy.issubdtype(dtype, numpy.complexfloating) for dtype in by_dtype}
    if len(kinds) < 2:
        return None

    listing = "; ".join(f"{dtype}: {', '.join(sorted(names))}" for dtype, names in sorted(by_dtype.items(), key=str))
    return (
        f"This form mixes real- and complex-dtype coefficients, which DOLFINx cannot assemble "
        f"({listing}). Under a complex-scalar build, construct every coefficient with the default "
        f"dtype so they are all complex; a coefficient given an explicit real dtype, or a raw "
        f"dolfinx.fem.Constant built from a Python float, is the usual cause."
    )


def unroll_dofmap(dofs: npt.NDArray[numpy.int32], bs: int) -> npt.NDArray[numpy.int32]:
    """
    Given a two-dimensional dofmap of size `(num_cells, num_dofs_per_cell)`
    Expand the dofmap by its block size such that the resulting array
    is of size `(num_cells, bs*num_dofs_per_cell)`
    """
    num_cells, num_dofs_per_cell = dofs.shape
    unrolled_dofmap = numpy.repeat(dofs, bs).reshape(num_cells, num_dofs_per_cell * bs) * bs
    unrolled_dofmap += numpy.tile(numpy.arange(bs), num_dofs_per_cell)
    return unrolled_dofmap


def _is_complex_build() -> bool:
    """Whether DOLFINx was built against a complex-scalar PETSc."""
    return bool(numpy.issubdtype(dolfinx.default_scalar_type, numpy.complexfloating))


def _argument_conjugation(expression: ufl.core.expr.Expr, argument: ufl.Argument) -> bool | None:
    """Report how ``argument`` enters ``expression``: conjugated, bare, or not at all.

    Raises:
        ufl.algorithms.check_arities.ArityMismatch: If a sum somewhere inside the
            expression adds a conjugated occurrence to a bare one, so that no single answer
            describes the expression.
    """
    arities = map_expr_dag(ArityChecker((argument,)), expression, compress=False)
    conjugations = {conjugated for arg, conjugated in arities if arg.number() == argument.number()}
    if not conjugations:
        return None
    (conjugation,) = conjugations
    return conjugation


def _conjugate_argument(expression: ufl.core.expr.Expr, argument: ufl.Argument) -> ufl.core.expr.Expr:
    """Flip whether ``argument`` counts as conjugated inside ``expression``.

    The value is unchanged: the argument ranges over a real-valued (Lagrange) basis, for
    which ``conj(phi) == phi`` pointwise. Only the conjugation UFL *records* moves, and that
    is what the complex-mode arity rules are about.
    """
    return ufl.replace(expression, {argument: ufl.conj(argument)})


def _agree_on_conjugation(expression: ufl.core.expr.Expr, argument: ufl.Argument) -> ufl.core.expr.Expr:
    """Rewrite ``expression`` so that no sum inside it mixes conjugation states.

    Descends only where the arity checker reports a mixture -- an expression it already
    accepts is returned untouched -- and repairs each offending sum by conjugating the
    argument in the terms that lack it.
    """
    try:
        _argument_conjugation(expression, argument)
    except ArityMismatch:
        pass
    else:
        return expression

    operands = [_agree_on_conjugation(operand, argument) for operand in expression.ufl_operands]
    if isinstance(expression, ufl.classes.Sum):
        conjugations = [_argument_conjugation(operand, argument) for operand in operands]
        if True in conjugations and False in conjugations:
            operands = [
                _conjugate_argument(operand, argument) if conjugation is False else operand
                for operand, conjugation in zip(operands, conjugations)
            ]
    return expression._ufl_expr_reconstruct_(*operands)


def _conjugate_for_complex_mode(integrand: ufl.core.expr.Expr, argument: ufl.Argument) -> ufl.core.expr.Expr:
    """Rewrite an integrand so that ``argument`` is conjugated, as complex mode requires."""
    integrand = _agree_on_conjugation(integrand, argument)
    if _argument_conjugation(integrand, argument) is False:
        integrand = _conjugate_argument(integrand, argument)
    return integrand


def _derivative_along(form: ufl.Form, coefficient: ufl.core.expr.Expr, direction, argument) -> ufl.Form:
    """Differentiate ``form`` along ``direction``, repaired for complex-mode arity rules."""
    dform = ufl.algorithms.expand_derivatives(ufl.derivative(form, coefficient, direction))
    return map_integrands(lambda integrand: _conjugate_for_complex_mode(integrand, argument), dform)


def wirtinger_derivative_forms(
    form: ufl.Form, coefficient: ufl.core.expr.Expr, argument: ufl.Argument
) -> tuple[ufl.Form, ufl.Form | None]:
    r"""Forms whose assembly gives the adjoint seed of ``form`` with respect to ``coefficient``.

    Under a real-scalar build this is just ``ufl.derivative(form, coefficient, argument)``,
    returned alone.

    Under a complex-scalar build one derivative is not enough. ``form`` is a real-differentiable
    function of a complex coefficient, so its differential splits into a holomorphic and an
    anti-holomorphic part,

    .. math::

        dF = \frac{\partial F}{\partial c}\,dc + \frac{\partial F}{\partial \bar c}\,\overline{dc},

    and the adjoint seed this codebase carries is :math:`\overline{\partial F/\partial c} +
    \partial F/\partial \bar c` -- conjugated on the holomorphic part because every pairing
    downstream of it is Hermitian ({py:func}`ufl.adjoint`), and carrying the anti-holomorphic
    part is what lets a Functional such as :math:`|u - u_d|^2` be differentiated at all.

    {py:func}`ufl.derivative` returns neither part on its own: differentiating along
    ``argument`` gives their sum, :math:`v_1 = \partial F/\partial c + \partial F/\partial
    \bar c` (the derivative along a *real* perturbation, since the argument ranges over a
    real-valued basis). Differentiating along ``1j * argument`` gives
    :math:`v_2 = i(\partial F/\partial c - \partial F/\partial \bar c)`, and the two
    together separate the parts. The seed is then, vector by vector,

    .. math::

        \mathrm{Re}(v_1) + i\,\mathrm{Re}(v_2),

    which callers must form themselves, having assembled both forms.

    Each form is also repaired for UFL's complex-mode arity rules, which require argument
    number 0 to appear conjugated in every term. {py:func}`ufl.derivative` leaves the direction
    it is handed exactly as given, so the raw derivative generally violates that; conjugating
    the direction up front instead only moves the violation to the form shapes where the
    differentiated occurrence of ``coefficient`` sits in the second slot of an
    {py:func}`ufl.inner`, which conjugates it a second time. The conjugation therefore has to be
    decided per term, after differentiating. Doing so is free of numerical consequence: the
    argument ranges over a real-valued (Lagrange) basis, so only the conjugation UFL records
    changes, never a value.

    Args:
        form: The rank-0 form to differentiate.
        coefficient: The coefficient to differentiate with respect to.
        argument: The direction to differentiate in, an argument over a real-valued basis.
    Returns:
        The derivative along ``argument``, and -- under a complex-scalar build -- the
        derivative along ``1j * argument``, which is ``None`` otherwise.
    """
    if not _is_complex_build():
        return ufl.derivative(form, coefficient, argument), None
    return (
        _derivative_along(form, coefficient, argument, argument),
        _derivative_along(form, coefficient, 1j * argument, argument),
    )


@contextlib.contextmanager
def _explaining_scalar_type_mismatch(form: typing.Any) -> typing.Iterator[None]:
    """Re-raise a dtype-mix failure from DOLFINx with a message naming the coefficients.

    See {py:func}`scalar_type_mismatch_message` for why the original error names neither.
    Anything else raised inside the block is left alone.
    """
    try:
        yield
    except (TypeError, RuntimeError) as error:
        hint = scalar_type_mismatch_message(form)
        if hint is None:
            raise
        raise type(error)(f"{hint}\n\nOriginal error: {error}") from error
