# Owner(s): ["module: autograd"]
"""Tests for autograd Node lifecycle management.

Verifies that autograd nodes, their Python wrappers, and associated objects
are properly freed — both by reference counting and by the GC cycle collector.

The test matrix covers:
  - Node type: C++ (MulBackward0 etc.) vs Python (custom Function)
  - Hook type: none, tensor hook, grad_fn hook, post_accumulate_grad hook
  - Hook captures grad_fn: creates a cycle requiring GC
  - Backward state: none, backward(), backward(retain_graph=True)
  - Shared grad_fn: one grad_fn with multiple output tensors
"""

# [MODULE DOC] Tests that autograd graph nodes, saved tensors and hook closures are released
# promptly, either by reference counting or by the cycle collector. Defines enums describing the
# test matrix, several custom Function classes, a Marker sentinel tracked through weak references,
# and TestCase classes for single outputs, leaves, shared grad_fn, chains and saved variables.
# No import-time side effects beyond defining classes.

import gc
import weakref
from enum import auto, Enum

import torch
from torch.autograd import Function
from torch.testing._internal.common_utils import run_tests, TestCase


class NodeType(Enum):
    # [CLASS DOC] HELPER: Enumeration of the kind of autograd node under test: built-in C++ node
    # or node created by a Python custom Function.
    CPP = auto()
    PYTHON = auto()


class HookType(Enum):
    # [CLASS DOC] HELPER: Enumeration of the hook registered on the output: none, tensor hook,
    # grad_fn hook (each optionally with a closure capturing grad_fn, forming a reference cycle),
    # or a post-accumulate-grad hook.
    NONE = auto()
    TENSOR_HOOK = auto()
    TENSOR_HOOK_CAPTURES_GRAD_FN = auto()
    GRAD_FN_HOOK = auto()
    GRAD_FN_HOOK_CAPTURES_GRAD_FN = auto()
    POST_ACCUMULATE_GRAD = auto()


class BackwardState(Enum):
    # [CLASS DOC] HELPER: Enumeration of whether backward was run before checking: not run, run
    # normally (graph freed), or run with retain_graph set.
    NONE = auto()
    BACKWARD = auto()
    RETAIN_GRAPH = auto()


class PythonId(Function):
    # [CLASS DOC] HELPER: Custom autograd Function acting as identity; forward clones the input
    # and backward passes the gradient through unchanged. Saves no tensors.
    """Identity custom function."""

    @staticmethod
    def forward(ctx, x):
        # [FUNCTION DOC] HELPER: Returns a clone of the input tensor.
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        # [FUNCTION DOC] HELPER: Returns the incoming gradient unchanged.
        return grad


class PythonMul(Function):
    # [CLASS DOC] HELPER: Custom autograd Function that doubles its input and saves the input
    # for backward, used to test saved-tensor lifetime for Python nodes.
    """Custom function that saves a tensor."""

    @staticmethod
    def forward(ctx, x):
        # [FUNCTION DOC] HELPER: Saves the input in the context and returns it times two.
        ctx.save_for_backward(x)
        return x * 2

    @staticmethod
    def backward(ctx, grad):
        # [FUNCTION DOC] HELPER: Unpacks the saved input (unused) and returns grad times two.
        (x,) = ctx.saved_tensors
        return grad * 2


class PythonExp(Function):
    # [CLASS DOC] HELPER: Custom autograd Function computing the exponential and saving its own
    # output for backward, used to test saved-output lifetime.
    """Custom function that saves its output."""

    @staticmethod
    def forward(ctx, x):
        # [FUNCTION DOC] HELPER: Computes exp of the input, saves that output, and returns it.
        y = x.exp()
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad):
        # [FUNCTION DOC] HELPER: Returns the gradient multiplied by the saved output.
        (y,) = ctx.saved_tensors
        return grad * y


class PythonSplit(Function):
    # [CLASS DOC] HELPER: Custom autograd Function with two outputs that share one grad_fn;
    # splits a tensor into a first part of two elements and the remainder.
    """Custom function with two outputs."""

    @staticmethod
    def forward(ctx, x):
        # [FUNCTION DOC] HELPER: Returns clones of the first two elements and of the rest.
        return x[:2].clone(), x[2:].clone()

    @staticmethod
    def backward(ctx, grad0, grad1):
        # [FUNCTION DOC] HELPER: Concatenates the two output gradients into the input gradient.
        return torch.cat([grad0, grad1])


def make_output(node_type, a):
    # [FUNCTION DOC] HELPER: Builds an output from tensor a: for the C++ node type multiplies by
    # two with a built-in op, otherwise applies the PythonMul custom function. Returns the tensor.
    """Create an output tensor with the given node type."""
    if node_type == NodeType.CPP:
        return a * 2
    else:
        return PythonMul.apply(a)


class Marker:
    # [CLASS DOC] HELPER: Empty plain class whose instances support weak references; captured by
    # hooks so a dead weak reference proves the hook closure was freed.
    """Weak-ref-able sentinel to track object lifetime."""


class TestNodeLifecycle(TestCase):
    # [CLASS DOC] Parameter-matrix tests for a single output tensor: node type crossed with hook
    # type and backward state. Helpers build the graph and assert the marker is freed by
    # reference counting (no cycle) or only after the cycle collector (hook captures grad_fn).
    """Single output, various node types / hooks / backward states."""

    def _run(self, node_type, hook_type, backward_state):
        # [FUNCTION DOC] HELPER: Builds a graph from a random leaf, optionally registers a hook
        # of the requested type (capturing a Marker and, for the captures variants, grad_fn), and
        # optionally runs backward on the sum (with retain_graph per backward_state). Returns the
        # leaf, output, marker, weak reference to marker, and grad_fn. Hooks only touch captures.
        a = torch.randn(4, requires_grad=True)
        b = make_output(node_type, a)
        marker = Marker()
        marker_ref = weakref.ref(marker)

        grad_fn = b.grad_fn

        if hook_type == HookType.TENSOR_HOOK:

            def hook(grad):
                # [FUNCTION DOC] HELPER: Tensor hook that only references the marker closure.
                marker

            b.register_hook(hook)
        elif hook_type == HookType.TENSOR_HOOK_CAPTURES_GRAD_FN:

            def hook(grad):
                # [FUNCTION DOC] HELPER: Tensor hook referencing grad_fn and marker, creating a
                # reference cycle through the graph.
                grad_fn
                marker

            b.register_hook(hook)
        elif hook_type == HookType.GRAD_FN_HOOK:

            def hook(*args):
                # [FUNCTION DOC] HELPER: grad_fn hook that only references the marker closure.
                marker

            b.grad_fn.register_hook(hook)
        elif hook_type == HookType.GRAD_FN_HOOK_CAPTURES_GRAD_FN:

            def hook(*args):
                # [FUNCTION DOC] HELPER: grad_fn hook referencing grad_fn and marker, creating a
                # reference cycle through the node.
                grad_fn
                marker

            b.grad_fn.register_hook(hook)

        if backward_state == BackwardState.BACKWARD:
            b.sum().backward()
        elif backward_state == BackwardState.RETAIN_GRAPH:
            b.sum().backward(retain_graph=True)

        return a, b, marker, marker_ref, grad_fn

    def _check_freed_by_refcount(self, node_type, hook_type, backward_state):
        # [FUNCTION DOC] HELPER: Runs the scenario, drops all local references without running the
        # cycle collector, and asserts the marker weak reference is dead. Fails otherwise.
        a, b, marker, marker_ref, grad_fn = self._run(
            node_type, hook_type, backward_state
        )
        del a, b, marker, grad_fn
        self.assertIsNone(
            marker_ref(),
            lambda msg: f"{msg}\nnot freed by refcount: {node_type} {hook_type} {backward_state}",
        )

    def _check_freed_by_gc(self, node_type, hook_type, backward_state):
        # [FUNCTION DOC] HELPER: Same as the refcount check but runs a full garbage collection
        # after dropping references, then asserts the marker weak reference is dead.
        a, b, marker, marker_ref, grad_fn = self._run(
            node_type, hook_type, backward_state
        )
        del a, b, marker, grad_fn
        gc.collect()
        self.assertIsNone(
            marker_ref(),
            lambda msg: f"{msg}\nnot freed by GC: {node_type} {hook_type} {backward_state}",
        )

    # === No hooks ===

    def test_cpp_no_hook_no_backward_refcount(self):
        # [FUNCTION DOC] C++ node, no hook, no backward: marker freed by refcount.
        self._check_freed_by_refcount(NodeType.CPP, HookType.NONE, BackwardState.NONE)

    def test_cpp_no_hook_backward_refcount(self):
        # [FUNCTION DOC] C++ node, no hook, after backward: marker freed by refcount.
        self._check_freed_by_refcount(
            NodeType.CPP, HookType.NONE, BackwardState.BACKWARD
        )

    def test_cpp_no_hook_retain_refcount(self):
        # [FUNCTION DOC] C++ node, no hook, backward with retain_graph: freed by refcount.
        self._check_freed_by_refcount(
            NodeType.CPP, HookType.NONE, BackwardState.RETAIN_GRAPH
        )

    def test_python_no_hook_no_backward_refcount(self):
        # [FUNCTION DOC] Python node, no hook, no backward: marker freed by refcount.
        self._check_freed_by_refcount(
            NodeType.PYTHON, HookType.NONE, BackwardState.NONE
        )

    def test_python_no_hook_backward_refcount(self):
        # [FUNCTION DOC] Python node, no hook, after backward: marker freed by refcount.
        self._check_freed_by_refcount(
            NodeType.PYTHON, HookType.NONE, BackwardState.BACKWARD
        )

    def test_python_no_hook_retain_refcount(self):
        # [FUNCTION DOC] Python node, no hook, backward with retain_graph: freed by refcount.
        self._check_freed_by_refcount(
            NodeType.PYTHON, HookType.NONE, BackwardState.RETAIN_GRAPH
        )

    # === Tensor hooks (no cycle) ===

    def test_cpp_tensor_hook_no_backward_refcount(self):
        # [FUNCTION DOC] C++ node with a non-capturing tensor hook, no backward: refcount free.
        self._check_freed_by_refcount(
            NodeType.CPP, HookType.TENSOR_HOOK, BackwardState.NONE
        )

    def test_cpp_tensor_hook_backward_refcount(self):
        # [FUNCTION DOC] C++ node with a non-capturing tensor hook, after backward: refcount free.
        self._check_freed_by_refcount(
            NodeType.CPP, HookType.TENSOR_HOOK, BackwardState.BACKWARD
        )

    def test_python_tensor_hook_no_backward_refcount(self):
        # [FUNCTION DOC] Python node with a non-capturing tensor hook, no backward: refcount free.
        self._check_freed_by_refcount(
            NodeType.PYTHON, HookType.TENSOR_HOOK, BackwardState.NONE
        )

    def test_python_tensor_hook_backward_refcount(self):
        # [FUNCTION DOC] Python node with a non-capturing tensor hook, after backward: refcount.
        self._check_freed_by_refcount(
            NodeType.PYTHON, HookType.TENSOR_HOOK, BackwardState.BACKWARD
        )

    # === Tensor hooks capturing grad_fn (cycle, needs GC) ===

    def test_cpp_tensor_hook_captures_no_backward_gc(self):
        # [FUNCTION DOC] C++ node, tensor hook capturing grad_fn (cycle), no backward: GC frees.
        self._check_freed_by_gc(
            NodeType.CPP, HookType.TENSOR_HOOK_CAPTURES_GRAD_FN, BackwardState.NONE
        )

    def test_cpp_tensor_hook_captures_backward_gc(self):
        # [FUNCTION DOC] C++ node, tensor hook capturing grad_fn (cycle), after backward: GC frees.
        self._check_freed_by_gc(
            NodeType.CPP, HookType.TENSOR_HOOK_CAPTURES_GRAD_FN, BackwardState.BACKWARD
        )

    def test_python_tensor_hook_captures_no_backward_gc(self):
        # [FUNCTION DOC] Python node, tensor hook capturing grad_fn (cycle), no backward: GC frees.
        self._check_freed_by_gc(
            NodeType.PYTHON, HookType.TENSOR_HOOK_CAPTURES_GRAD_FN, BackwardState.NONE
        )

    def test_python_tensor_hook_captures_backward_gc(self):
        # [FUNCTION DOC] Python node, tensor hook capturing grad_fn (cycle), after backward: GC.
        self._check_freed_by_gc(
            NodeType.PYTHON,
            HookType.TENSOR_HOOK_CAPTURES_GRAD_FN,
            BackwardState.BACKWARD,
        )

    # === grad_fn hooks (no cycle) ===

    def test_cpp_grad_fn_hook_no_backward_refcount(self):
        # [FUNCTION DOC] C++ node with a non-capturing grad_fn hook, no backward: refcount free.
        self._check_freed_by_refcount(
            NodeType.CPP, HookType.GRAD_FN_HOOK, BackwardState.NONE
        )

    def test_cpp_grad_fn_hook_backward_refcount(self):
        # [FUNCTION DOC] C++ node with a non-capturing grad_fn hook, after backward: refcount.
        self._check_freed_by_refcount(
            NodeType.CPP, HookType.GRAD_FN_HOOK, BackwardState.BACKWARD
        )

    def test_python_grad_fn_hook_no_backward_refcount(self):
        # [FUNCTION DOC] Python node with a non-capturing grad_fn hook, no backward: refcount.
        self._check_freed_by_refcount(
            NodeType.PYTHON, HookType.GRAD_FN_HOOK, BackwardState.NONE
        )

    def test_python_grad_fn_hook_backward_refcount(self):
        # [FUNCTION DOC] Python node with a non-capturing grad_fn hook, after backward: refcount.
        self._check_freed_by_refcount(
            NodeType.PYTHON, HookType.GRAD_FN_HOOK, BackwardState.BACKWARD
        )

    # === grad_fn hooks capturing grad_fn (cycle, needs GC) ===

    def test_cpp_grad_fn_hook_captures_no_backward_gc(self):
        # [FUNCTION DOC] C++ node, grad_fn hook capturing grad_fn (cycle), no backward: GC frees.
        self._check_freed_by_gc(
            NodeType.CPP, HookType.GRAD_FN_HOOK_CAPTURES_GRAD_FN, BackwardState.NONE
        )

    def test_cpp_grad_fn_hook_captures_backward_gc(self):
        # [FUNCTION DOC] C++ node, grad_fn hook capturing grad_fn (cycle), after backward: GC.
        self._check_freed_by_gc(
            NodeType.CPP, HookType.GRAD_FN_HOOK_CAPTURES_GRAD_FN, BackwardState.BACKWARD
        )

    def test_python_grad_fn_hook_captures_no_backward_gc(self):
        # [FUNCTION DOC] Python node, grad_fn hook capturing grad_fn (cycle), no backward: GC.
        self._check_freed_by_gc(
            NodeType.PYTHON, HookType.GRAD_FN_HOOK_CAPTURES_GRAD_FN, BackwardState.NONE
        )

    def test_python_grad_fn_hook_captures_backward_gc(self):
        # [FUNCTION DOC] Python node, grad_fn hook capturing grad_fn (cycle), after backward: GC.
        self._check_freed_by_gc(
            NodeType.PYTHON,
            HookType.GRAD_FN_HOOK_CAPTURES_GRAD_FN,
            BackwardState.BACKWARD,
        )


class TestLeafNodeLifecycle(TestCase):
    # [CLASS DOC] Tests for leaf handling (AccumulateGrad nodes): use counts of the leaf tensor
    # and freeing of post-accumulate-grad hooks.
    """Leaf-specific tests (AccumulateGrad)."""

    def test_accumulate_grad_freed_refcount(self):
        # [FUNCTION DOC] Asserts the leaf use count is 2 while a derived tensor exists and returns
        # to 1 once that tensor (and its graph) is deleted.
        """AccumulateGrad releases variable when graph is freed."""
        a = torch.randn(2, requires_grad=True)
        b = a * 2
        self.assertEqual(a._use_count(), 2)
        del b
        self.assertEqual(a._use_count(), 1)

    def test_post_accumulate_grad_hook_freed(self):
        # [FUNCTION DOC] Registers a marker-capturing post-accumulate-grad hook on a leaf, runs
        # backward, deletes everything and collects garbage; asserts the marker is freed.
        # The inner hook only references the marker.
        """Post-accumulate-grad hook freed when leaf is freed."""
        marker = Marker()
        ref = weakref.ref(marker)
        a = torch.randn(2, requires_grad=True)

        def hook(t):
            # [FUNCTION DOC] HELPER: Hook body that only references the captured marker.
            marker  # noqa: F821

        a.register_post_accumulate_grad_hook(hook)
        b = a * 2
        b.sum().backward()
        del a, b, marker, hook
        gc.collect()
        self.assertIsNone(ref())

    def test_accumulate_grad_freed_after_backward(self):
        # [FUNCTION DOC] After a normal backward, deleting the derived tensor returns the leaf
        # use count from 2 to 1.
        """AccumulateGrad releases variable after backward."""
        a = torch.randn(2, requires_grad=True)
        b = a * 2
        self.assertEqual(a._use_count(), 2)
        b.sum().backward()
        del b
        self.assertEqual(a._use_count(), 1)

    def test_accumulate_grad_freed_retain_graph(self):
        # [FUNCTION DOC] With retain_graph the leaf use count stays 2 after backward and drops to
        # 1 only once the output tensor is deleted.
        """AccumulateGrad keeps variable during retain_graph, frees after."""
        a = torch.randn(2, requires_grad=True)
        b = (a * 2).sum()
        b.backward(retain_graph=True)
        self.assertEqual(a._use_count(), 2)
        del b
        self.assertEqual(a._use_count(), 1)


class TestSharedGradFn(TestCase):
    # [CLASS DOC] Tests where several outputs (from split or a two-output custom Function) share
    # one grad_fn: the node must live until all outputs die, and cycles must be collectable.
    """Tests for grad_fn shared by multiple output tensors."""

    def test_cpp_shared_grad_fn_one_deleted(self):
        # [FUNCTION DOC] Splits a C++ result into two views; deleting one keeps the leaf use
        # count at 2, deleting both returns it to 1.
        """Shared grad_fn stays alive when one output is deleted."""
        a = torch.randn(4, requires_grad=True)
        b0, b1 = (a * 2).split(2)
        self.assertEqual(a._use_count(), 2)
        del b0
        self.assertEqual(a._use_count(), 2)
        del b1
        self.assertEqual(a._use_count(), 1)

    def test_python_shared_grad_fn_one_deleted(self):
        # [FUNCTION DOC] Same for the two-output Python Function: a weak reference to grad_fn
        # stays alive after deleting one output and dies after deleting both.
        """Python shared grad_fn stays alive when one output is deleted."""
        a = torch.randn(4, requires_grad=True)
        b0, b1 = PythonSplit.apply(a)
        ref = weakref.ref(b0.grad_fn)
        del b0
        self.assertIsNotNone(ref())
        del b1
        self.assertIsNone(ref())

    def test_cpp_shared_grad_fn_backward_one(self):
        # [FUNCTION DOC] Runs backward (retain_graph) through one shared C++ output, deletes both
        # outputs, and asserts the leaf use count returns to 1.
        """backward on one shared output, then free both."""
        a = torch.randn(4, requires_grad=True)
        b0, b1 = (a * 2).split(2)
        b0.sum().backward(retain_graph=True)
        del b0, b1
        self.assertEqual(a._use_count(), 1)

    def test_python_shared_grad_fn_backward_one(self):
        # [FUNCTION DOC] Runs backward (retain_graph) through one Python shared output, deletes
        # both outputs, and asserts the grad_fn weak reference is dead.
        a = torch.randn(4, requires_grad=True)
        b0, b1 = PythonSplit.apply(a)
        ref = weakref.ref(b0.grad_fn)
        b0.sum().backward(retain_graph=True)
        del b0, b1
        self.assertIsNone(ref())

    def test_cpp_shared_grad_fn_hook_cycle_gc(self):
        # [FUNCTION DOC] Registers a hook capturing the shared C++ grad_fn and a marker (a cycle),
        # deletes everything, collects garbage, and asserts the marker is freed.
        """Hook capturing shared grad_fn creates a cycle, GC collects it."""
        marker = Marker()
        ref = weakref.ref(marker)

        a = torch.randn(4, requires_grad=True)
        b0, b1 = (a * 2).split(2)
        grad_fn = b0.grad_fn

        def hook(*args):
            # [FUNCTION DOC] HELPER: Hook referencing grad_fn and marker to form a cycle.
            grad_fn  # noqa: F821
            marker  # noqa: F821

        b0.grad_fn.register_hook(hook)
        del a, b0, b1, grad_fn, hook, marker
        gc.collect()
        self.assertIsNone(ref())

    def test_python_shared_grad_fn_hook_cycle_gc(self):
        # [FUNCTION DOC] Same cycle scenario with the shared Python grad_fn; asserts the marker is
        # freed after garbage collection.
        """Hook capturing shared Python grad_fn, GC collects it."""
        marker = Marker()
        ref = weakref.ref(marker)

        a = torch.randn(4, requires_grad=True)
        b0, b1 = PythonSplit.apply(a)
        grad_fn = b0.grad_fn

        def hook(*args):
            # [FUNCTION DOC] HELPER: Hook referencing grad_fn and marker to form a cycle.
            grad_fn  # noqa: F821
            marker  # noqa: F821

        b0.grad_fn.register_hook(hook)
        del a, b0, b1, grad_fn, hook, marker
        gc.collect()
        self.assertIsNone(ref())


class TestChainLifecycle(TestCase):
    # [CLASS DOC] Tests for chains of C++, Python and mixed nodes being freed, including very
    # deep chains that must be destroyed without overflowing the stack.
    """Tests for chains of nodes."""

    def test_cpp_chain_freed_refcount(self):
        # [FUNCTION DOC] Builds a three-op C++ chain and asserts the leaf use count returns to 1
        # after the intermediates are deleted.
        a = torch.randn(2, requires_grad=True)
        b = a * 2
        c = b + 1
        d = c * 3
        self.assertEqual(a._use_count(), 2)
        del b, c, d
        self.assertEqual(a._use_count(), 1)

    def test_python_chain_freed_refcount(self):
        # [FUNCTION DOC] Chains two Python Function nodes; after deleting the tensors, asserts
        # weak references to both grad_fns are dead.
        a = torch.randn(2, requires_grad=True)
        b = PythonMul.apply(a)
        c = PythonId.apply(b)
        ref_b = weakref.ref(b.grad_fn)
        ref_c = weakref.ref(c.grad_fn)
        del b, c
        self.assertIsNone(ref_b())
        self.assertIsNone(ref_c())

    def test_mixed_chain_freed_refcount(self):
        # [FUNCTION DOC] Chain of C++, Python, C++ nodes; after deletion asserts the Python
        # node's weak reference is dead.
        a = torch.randn(2, requires_grad=True)
        b = a * 2  # C++
        c = PythonId.apply(b)  # Python
        d = c + 1  # C++
        ref_c = weakref.ref(c.grad_fn)
        del b, c, d
        self.assertIsNone(ref_c())

    def test_deep_chain_no_stack_overflow(self):
        # [FUNCTION DOC] Builds a 2000-op C++ chain then deletes it; passes if destruction does
        # not crash from deep recursion.
        x = torch.randn(2, requires_grad=True)
        for _ in range(2000):
            x = x + torch.randn(2)
        del x

    def test_deep_python_chain_no_stack_overflow(self):
        # [FUNCTION DOC] Same as the deep chain test but with 2000 Python Function nodes.
        x = torch.randn(2, requires_grad=True)
        for _ in range(2000):
            x = PythonId.apply(x)
        del x


class TestSavedVariableLifecycle(TestCase):
    # [CLASS DOC] Tests that tensors saved for backward by C++ and Python nodes are kept while the
    # graph is alive (or retained) and freed when the graph is freed or consumed by backward.
    """Tests that saved tensors are freed at the right time.

    Uses b**2 for C++ tests (PowBackward0 saves self) and PythonMul for
    Python tests (save_for_backward saves input). Both use a non-leaf
    intermediate tracked via weakref.
    """

    def test_cpp_saved_freed_on_graph_free(self):
        # [FUNCTION DOC] An intermediate saved by a power op stays alive after its name is
        # deleted and dies when the result graph is deleted.
        a = torch.randn(2, requires_grad=True)
        b = a + 1  # non-leaf intermediate
        ref = weakref.ref(b)
        c = b**2  # PowBackward0 saves b
        del b
        self.assertIsNotNone(ref())
        del c
        self.assertIsNone(ref())

    def test_python_saved_freed_on_graph_free(self):
        # [FUNCTION DOC] Same as the C++ variant using PythonMul's save_for_backward.
        a = torch.randn(2, requires_grad=True)
        b = a + 1
        ref = weakref.ref(b)
        c = PythonMul.apply(b)
        del b
        self.assertIsNotNone(ref())
        del c
        self.assertIsNone(ref())

    def test_cpp_saved_freed_after_backward(self):
        # [FUNCTION DOC] Saved intermediate stays alive until backward runs, then dies.
        a = torch.randn(2, requires_grad=True)
        b = a + 1
        ref = weakref.ref(b)
        c = (b**2).sum()
        del b
        self.assertIsNotNone(ref())
        c.backward()
        self.assertIsNone(ref())

    def test_python_saved_freed_after_backward(self):
        # [FUNCTION DOC] Same as the C++ variant for a Python Function's saved input.
        a = torch.randn(2, requires_grad=True)
        b = a + 1
        ref = weakref.ref(b)
        c = PythonMul.apply(b)
        del b
        self.assertIsNotNone(ref())
        c.sum().backward()
        self.assertIsNone(ref())

    def test_cpp_saved_kept_with_retain_graph(self):
        # [FUNCTION DOC] With retain_graph the saved intermediate survives backward and dies only
        # when the result is deleted.
        a = torch.randn(2, requires_grad=True)
        b = a + 1
        ref = weakref.ref(b)
        c = (b**2).sum()
        del b
        c.backward(retain_graph=True)
        self.assertIsNotNone(ref())
        del c
        self.assertIsNone(ref())

    def test_python_saved_kept_with_retain_graph(self):
        # [FUNCTION DOC] Same as the C++ variant for a Python Function's saved input.
        a = torch.randn(2, requires_grad=True)
        b = a + 1
        ref = weakref.ref(b)
        c = PythonMul.apply(b)
        del b
        c.sum().backward(retain_graph=True)
        self.assertIsNotNone(ref())
        del c
        self.assertIsNone(ref())

    def test_cpp_saved_output_freed_on_graph_free(self):
        # [FUNCTION DOC] An exp result saved as its own output is freed once the user drops it
        # (the saved output does not keep it alive).
        a = torch.randn(2, requires_grad=True)
        b = a.exp()  # ExpBackward0 saves its output (b)
        ref = weakref.ref(b)
        del b
        self.assertIsNone(ref())

    def test_python_saved_output_freed_on_graph_free(self):
        # [FUNCTION DOC] Same as the C++ variant using PythonExp.
        a = torch.randn(2, requires_grad=True)
        b = PythonExp.apply(a)
        ref = weakref.ref(b)
        del b
        self.assertIsNone(ref())

    def test_cpp_saved_output_freed_after_backward(self):
        # [FUNCTION DOC] After backward, deleting the exp output frees it.
        a = torch.randn(2, requires_grad=True)
        b = a.exp()
        ref = weakref.ref(b)
        b.sum().backward()
        del b
        self.assertIsNone(ref())

    def test_python_saved_output_freed_after_backward(self):
        # [FUNCTION DOC] Same as the C++ variant using PythonExp.
        a = torch.randn(2, requires_grad=True)
        b = PythonExp.apply(a)
        ref = weakref.ref(b)
        b.sum().backward()
        del b
        self.assertIsNone(ref())


if __name__ == "__main__":
    run_tests()
