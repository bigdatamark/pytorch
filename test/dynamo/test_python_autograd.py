# Owner(s): ["module: dynamo"]
# [MODULE DOC] A pure-python tape-based reverse-mode autograd (Variable, tape entries, per-
# operator backward closures) used as a hard tracing case for dynamo, plus tests comparing
# dynamo-optimized and eager results. Import-time state: a global name counter and a global
# gradient tape list.
from typing import NamedTuple, TYPE_CHECKING

import torch
import torch._dynamo
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.testing import CompileCounter, same


if TYPE_CHECKING:
    from collections.abc import Callable


"""
This is an example of a pure-python version of autograd implemented by
@zdevito.  It represents a rather challenging test case for TorchDynamo
to push the limits of what it can do.
"""


_name: int = 0


def fresh_name() -> str:
    # [FUNCTION DOC] HELPER: Returns a fresh unique variable name of the form v followed by a
    # counter, and increments the module-level counter (global side effect).
    """create a new unique name for a variable: v0, v1, v2"""
    global _name
    r = f"v{_name}"
    _name += 1
    return r


class Variable:
    # [CLASS DOC] Tensor wrapper with a name; the name is auto-generated when not given.
    # Overloaded multiplication and addition delegate to the module-level operators which record
    # entries on the gradient tape. Also offers sum and expand returning Variables.
    def __init__(self, value: torch.Tensor, name: str | None = None):
        # [FUNCTION DOC] Stores the wrapped tensor and a name, using a freshly generated name if
        # none is given (or an empty one).
        self.value = value
        self.name = name or fresh_name()

    # We need to start with some tensors whose values were not computed
    # inside the autograd. This function constructs leaf nodes.
    @staticmethod
    def constant(value: torch.Tensor, name: str | None = None):
        # [FUNCTION DOC] Static constructor for leaf variables whose values were not computed by
        # this autograd; returns a Variable with the given tensor and optional name.
        return Variable(value, name)

    def __repr__(self):
        # [FUNCTION DOC] Returns the repr of the wrapped tensor value.
        return repr(self.value)

    # This performs a pointwise multiplication of a Variable, tracking gradients
    def __mul__(self, rhs: "Variable") -> "Variable":
        # [FUNCTION DOC] Pointwise multiplication with another Variable (or the float one);
        # delegates to operator_mul, which records the operation on the tape.
        # defined later in the notebook
        return operator_mul(self, rhs)

    def __add__(self, rhs: "Variable") -> "Variable":
        # [FUNCTION DOC] Pointwise addition; delegates to operator_add, which records the
        # operation on the tape.
        return operator_add(self, rhs)

    def sum(self, name: str | None = None) -> "Variable":
        # [FUNCTION DOC] Sums all elements into a scalar Variable with the optional name;
        # delegates to operator_sum.
        return operator_sum(self, name)

    def expand(self, sizes: list[int]) -> "Variable":
        # [FUNCTION DOC] Expands a scalar Variable to the given sizes; delegates to
        # operator_expand.
        return operator_expand(self, sizes)


class TapeEntry(NamedTuple):
    # [CLASS DOC] Named tuple recording one taped computation: input variable names, output
    # variable names, and a propagate callable mapping gradients of outputs to gradients of
    # inputs (chain rule step).
    # names of the inputs to the original computation
    inputs: list[str]
    # names of the outputs of the original computation
    outputs: list[str]
    # apply chain rule
    propagate: "Callable[list[Variable], list[Variable]]"


gradient_tape: list[TapeEntry] = []


def reset_tape():
    # [FUNCTION DOC] Clears the global gradient tape and resets the global name counter to zero.
    gradient_tape.clear()
    global _name
    _name = 0


def grad(L, desired_results: list[Variable]) -> list[Variable]:
    # [FUNCTION DOC] Computes gradients of the loss variable L with respect to the desired
    # variables by walking the global gradient tape in reverse. Seeds the loss gradient with one,
    # skips entries whose output gradients are all None, calls each entry's propagate, and
    # accumulates gradients per input name (adding values in place for repeated inputs). Returns
    # a list of gradients aligned with the desired variables, None for unused ones.
    # this map holds dL/dX for all values X
    dL_d: dict[str, Variable] = {}
    # It starts by initializing the 'seed' dL/dL, which is 1
    dL_d[L.name] = Variable(torch.ones(()))
    # print(f'd{L.name} ------------------------')

    # look up dL_dentries. If a variable is never used to compute the loss,
    # we consider its gradient None, see the note below about zeros for more information.
    def gather_grad(entries: list[str]):
        # [FUNCTION DOC] HELPER: Returns the gradient mapping value (or None when absent) for
        # each name in the list.
        return [dL_d.get(entry) for entry in entries]

    # propagate the gradient information backward
    for entry in reversed(gradient_tape):
        dL_doutputs = gather_grad(entry.outputs)
        if all(dL_doutput is None for dL_doutput in dL_doutputs):
            # optimize for the case where some gradient pathways are zero. See
            # The note below for more details.
            continue

        # perform chain rule propagation specific to each compute
        dL_dinputs = entry.propagate(dL_doutputs)

        # Accumulate the gradient produced for each input.
        # Each use of a variable produces some gradient dL_dinput for that
        # use. The multivariate chain rule tells us it is safe to sum
        # all the contributions together.
        for input, dL_dinput in zip(entry.inputs, dL_dinputs):
            if input not in dL_d:
                dL_d[input] = dL_dinput
            else:
                dL_d[input].value += dL_dinput.value

    # print some information to understand the values of each intermediate
    # for name, value in dL_d.items():
    #    print(f'd{L.name}_d{name} = {value.name}')
    # print(f'------------------------')

    return gather_grad(desired.name for desired in desired_results)


def operator_mul(self: Variable, rhs: Variable) -> Variable:
    # [FUNCTION DOC] Multiplies two Variables and appends a tape entry whose backward closure
    # scales the output gradient by each other operand. If rhs is the float one, returns self
    # unchanged without taping. Returns the new Variable.
    if isinstance(rhs, float) and rhs == 1.0:
        # peephole optimization
        return self

    # define forward
    r = Variable(self.value * rhs.value)
    # print(f'{r.name} = {self.name} * {rhs.name}')

    # record what the inputs and outputs of the op were
    inputs = [self.name, rhs.name]
    outputs = [r.name]

    # define backprop
    def propagate(dL_doutputs: list[Variable]):
        # [FUNCTION DOC] HELPER: Backward closure for multiplication; takes the single output
        # gradient and returns gradients for self and rhs by multiplying with the opposite
        # operand.
        (dL_dr,) = dL_doutputs

        dr_dself = rhs  # partial derivative of r = self*rhs
        dr_drhs = self  # partial derivative of r = self*rhs

        # chain rule propagation from outputs to inputs of multiply
        dL_dself = dL_dr * dr_dself
        dL_drhs = dL_dr * dr_drhs
        dL_dinputs = [dL_dself, dL_drhs]
        return dL_dinputs

    # finally, we record the compute we did on the tape
    gradient_tape.append(TapeEntry(inputs=inputs, outputs=outputs, propagate=propagate))
    return r


def operator_add(self: Variable, rhs: Variable) -> Variable:
    # [FUNCTION DOC] Adds two Variables and records a tape entry; returns the new Variable.
    # Add follows a similar pattern to Mul, but it doesn't end up
    # capturing any variables.
    r = Variable(self.value + rhs.value)
    # print(f'{r.name} = {self.name} + {rhs.name}')

    def propagate(dL_doutputs: list[Variable]):
        # [FUNCTION DOC] HELPER: Backward closure for addition; returns the output gradient
        # multiplied by one for each of the two inputs.
        (dL_dr,) = dL_doutputs
        dr_dself = 1.0
        dr_drhs = 1.0
        dL_dself = dL_dr * dr_dself
        dL_drhs = dL_dr * dr_drhs
        return [dL_dself, dL_drhs]

    gradient_tape.append(
        TapeEntry(inputs=[self.name, rhs.name], outputs=[r.name], propagate=propagate)
    )
    return r


def operator_sum(self: Variable, name: str | None) -> "Variable":
    # [FUNCTION DOC] Sums a Variable into a scalar Variable with the given name and records a
    # tape entry; returns it.
    r = Variable(torch.sum(self.value), name=name)
    # print(f'{r.name} = {self.name}.sum()')

    def propagate(dL_doutputs: list[Variable]):
        # [FUNCTION DOC] HELPER: Backward closure for sum; expands the output gradient to the
        # size of the input value.
        (dL_dr,) = dL_doutputs
        size = self.value.size()
        return [dL_dr.expand(*size)]

    gradient_tape.append(
        TapeEntry(inputs=[self.name], outputs=[r.name], propagate=propagate)
    )
    return r


def operator_expand(self: Variable, sizes: list[int]) -> "Variable":
    # [FUNCTION DOC] Expands a scalar Variable to the given sizes and records a tape entry.
    # Asserts the value is zero-dimensional. Returns the new Variable.
    assert self.value.dim() == 0  # noqa: S101 - only works for scalars
    r = Variable(self.value.expand(sizes))
    # print(f'{r.name} = {self.name}.expand({sizes})')

    def propagate(dL_doutputs: list[Variable]):
        # [FUNCTION DOC] HELPER: Backward closure for expand; returns the sum of the output
        # gradient.
        (dL_dr,) = dL_doutputs
        return [dL_dr.sum()]

    gradient_tape.append(
        TapeEntry(inputs=[self.name], outputs=[r.name], propagate=propagate)
    )
    return r


def simple(a, b):
    # [FUNCTION DOC] HELPER: Test function: adds the two inputs and multiplies the result by the
    # second.
    t = a + b
    return t * b


class TestPythonAutograd(TestCase):
    # [CLASS DOC] Dynamo test case that compiles forward and backward passes of the pure-python
    # autograd with a compile counter and compares them to eager results.
    def _common(self, fn, expected_ops):
        # [FUNCTION DOC] HELPER: Runs fn with dynamo (assert mode, compile counter) on two random
        # pairs of vectors, resetting the tape between every call, and checks results equal eager
        # outputs, a single compiled frame, and the expected operator count.
        args1 = [torch.randn(10), torch.randn(10)]
        args2 = [torch.randn(10), torch.randn(10)]
        cnt = CompileCounter()
        fn_dynamo = torch._dynamo.optimize_assert(cnt)(fn)
        reset_tape()
        res1 = fn_dynamo(*args1)
        reset_tape()
        res2 = fn_dynamo(*args2)
        reset_tape()
        self.assertTrue(same(res1, fn(*args1)))
        reset_tape()
        self.assertTrue(same(res2, fn(*args2)))
        reset_tape()
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, expected_ops)

    def test_forwards1(self):
        # [FUNCTION DOC] Forward-only test; the compiled function builds variables and returns
        # the summed loss; expects 3 ops.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Wraps inputs as constants and returns the sum of the simple
            # function.
            a = Variable.constant(a, name="a")
            b = Variable.constant(b, name="b")
            loss = simple(a, b).sum()
            return loss

        self._common(fn, 3)

    def test_forwards2(self):
        # [FUNCTION DOC] Forward-only test where the function resets the tape before and after;
        # expects 3 ops.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Resets the tape, wraps inputs, returns the summed simple
            # function, resets the tape again.
            reset_tape()
            a = Variable.constant(a, name="a")
            b = Variable.constant(b, name="b")
            loss = simple(a, b).sum()
            reset_tape()
            return loss

        self._common(fn, 3)

    def test_backwards1(self):
        # [FUNCTION DOC] Forward and backward in one compiled function returning gradients;
        # expects 8 ops.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Wraps inputs, computes the loss and returns the gradients
            # with respect to both inputs.
            a = Variable.constant(a, name="a")
            b = Variable.constant(b, name="b")
            loss = simple(a, b).sum()
            return grad(loss, [a, b])

        self._common(fn, 8)

    def test_backwards2(self):
        # [FUNCTION DOC] Same as the previous backward test but the function resets the tape at
        # the start and end; expects 8 ops.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Resets the tape, computes loss and gradients, resets the
            # tape, returns the gradients.
            reset_tape()
            a = Variable.constant(a, name="a")
            b = Variable.constant(b, name="b")
            loss = simple(a, b).sum()
            res = grad(loss, [a, b])
            reset_tape()
            return res

        self._common(fn, 8)

    def test_split(self):
        # [FUNCTION DOC] Compiles the forward and the grad function separately with one shared
        # counter. Checks the loss and gradients equal eager values, two frames compiled and
        # eight ops in total.
        v1 = Variable.constant(torch.randn(10), name="a")
        v2 = Variable.constant(torch.randn(10), name="b")
        cnt = CompileCounter()

        def forward(a, b):
            # [FUNCTION DOC] HELPER: Returns the sum of the simple function over its two
            # arguments.
            return simple(a, b).sum()

        reset_tape()
        loss1 = forward(v1, v2)
        grad1 = grad(loss1, [v1, v2])

        reset_tape()
        opt_forward = torch._dynamo.optimize_assert(cnt)(forward)
        opt_grad = torch._dynamo.optimize_assert(cnt)(grad)
        loss2 = opt_forward(v1, v2)
        # force two frames
        grad2 = opt_grad(loss2, [v1, v2])

        self.assertTrue(same(loss1, loss2))
        self.assertTrue(same(grad1, grad2))
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(cnt.op_count, 8)


if __name__ == "__main__":
    run_tests()
