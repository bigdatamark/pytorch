# Owner(s): ["oncall: profiler"]
# [MODULE DOC] Tests of the structured data the profiler's experimental event tree exposes for
# tensors, modules, optimizers and allocations: storage pointers and tensor ids, allocation ids
# across set and resize, tensor metadata (sizes, strides, layout, device, dtype), scalar and list
# arguments, nn.Module and optimizer parameter capture, allocation events and closure reference
# counts. Import-time side effect: if tqdm is installed its monitor thread interval is set to
# zero to avoid lingering threads. Defines the Json alias and a small two-layer network.

import gc
import re
import textwrap
import unittest
import weakref
from typing import Any

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch._C._profiler import _ExtraFields_PyCall, _TensorMetadata
from torch.profiler import _utils, profile
from torch.testing._internal.common_utils import (
    IS_LINUX,
    IS_MACOS,
    IS_WINDOWS,
    run_tests,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
)


# if tqdm is not shutdown properly, it will leave the monitor thread alive.
# This causes an issue in the multithreading test because we check all events
# in that test with their tids. The events that correspond to these lingering
# threads all have TID of (uint64_t)(-1) which is invalid.
# The work around is turning off monitoring thread when tqdm is loaded.
# Since these are unit tests, it is safe to turn off monitor thread.
try:
    import tqdm

    tqdm.tqdm.monitor_interval = 0
except ImportError:
    pass

Json = dict[str, Any]


def find_node_with_name(nodes, name):
    # [FUNCTION DOC] HELPER: Depth-first search of the given event tree roots; returns the first
    # node whose name equals the given string, or None implicitly if there is none.
    for node in _utils.traverse_dfs(nodes):
        if node.name == name:
            return node


def find_node_with_regex(nodes, pattern):
    # [FUNCTION DOC] HELPER: Depth-first search of the event tree roots; returns the first node
    # whose name matches the given regular expression anywhere, or None implicitly.
    for node in _utils.traverse_dfs(nodes):
        if re.search(pattern, node.name):
            return node


class SimpleNet(nn.Module):
    # [CLASS DOC] HELPER: Two-layer linear network (10 to 5, then 5 to 2) used as a profiling
    # target; the forward pass applies the first layer then the second.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the two linear layers fc1 and fc2.
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the second layer applied to the first layer's output.
        return self.fc2(self.fc1(x))


class TestTorchTidyProfiler(TestCase):
    # [CLASS DOC] Test case for tensor, module, optimizer and allocation metadata captured by the
    # profiler's event tree. Stateless; each test profiles its own code and inspects the tree.
    # Helpers: field extraction for tensor inputs, allocation table formatting, and optimizer
    # result checking.
    def _get_tensor_fields(self, node, index):
        # [FUNCTION DOC] HELPER: Asserts the node is a torch op node and that its input at the
        # given index is tensor metadata with implementation pointer, storage data pointer and id
        # all set. Returns those three values as a tuple.
        self.assertIsNotNone(node)
        self.assertIsInstance(
            node.extra_fields, torch._C._profiler._ExtraFields_TorchOp
        )
        tensor_info = node.extra_fields.inputs[index]
        self.assertIsInstance(tensor_info, _TensorMetadata)
        self.assertIsNotNone(tensor_info.impl_ptr)
        self.assertIsNotNone(tensor_info.storage_data_ptr)
        self.assertIsNotNone(tensor_info.id)
        return tensor_info.impl_ptr, tensor_info.storage_data_ptr, tensor_info.id

    def test_pointers_and_ids(self):
        # [FUNCTION DOC] Profiles adds, muls, a resize, a relu, and a set_ on tensors, then
        # checks: storage pointers match the Python API; views share storage but have distinct
        # impl pointers; a tensor reused across ops has identical impl, storage and id; resize
        # keeps impl and id but changes the storage pointer; and set_ to another tensor's storage
        # gives the two tensors the same id while impls differ.
        a = torch.randn(4, 3)
        a_initial_storage_data = a.storage().data_ptr()

        # Views of tensors can share the same storage, but have different TensorImpls
        b = a.view((1, 12))
        c = torch.randn(4, 1)
        c_initial_storage_data = c.storage().data_ptr()
        d = torch.randn(4, 3)

        with profile(with_stack=True, profile_memory=True, record_shapes=True) as p:
            _ = a + c
            _ = b * c

            # Resize should create a new data_ptr but keep the TensorImpl the same.
            f = a.resize_(128, 129)
            _ = torch.relu(f)

            # `.set_` points a Tensor at an existing storage.
            _ = d.sin()
            c.set_(d.storage())
            _ = c.cos()

        nodes = p.profiler.kineto_results.experimental_event_tree()

        def get_fields(op_name, index):
        # [FUNCTION DOC] HELPER: Finds the first node with the given op name in the profiled tree
        # and returns the impl pointer, storage pointer and id of its input at the given index.
            return self._get_tensor_fields(find_node_with_name(nodes, op_name), index)

        a_impl, a_storage_data, a_id = get_fields("aten::add", 0)
        b_impl, b_storage_data, _ = get_fields("aten::mul", 0)

        # Profiler matches ground truth from Python API.
        self.assertEqual(a_storage_data, a_initial_storage_data)

        # Views are handled correctly.
        self.assertEqual(a_storage_data, b_storage_data)
        self.assertNotEqual(a_impl, b_impl)

        # The same Tensor used in multiple calls gives identical results.
        c_impl, c_storage_data, c_id = get_fields("aten::add", 1)
        self.assertEqual((c_impl, c_storage_data, c_id), get_fields("aten::mul", 1))
        self.assertEqual(c_storage_data, c_initial_storage_data)

        # Mutations to the underlying storage are reflected. (But ID is shared.)
        f_impl, f_storage_data, f_id = get_fields("aten::relu", 0)
        self.assertEqual(a_impl, f_impl)
        self.assertNotEqual(a_storage_data, f_storage_data)
        self.assertEqual(a_id, f_id)

        # Calling `set_` with an existing Tensor makes them share an ID.
        d_impl, d_storage_data, d_id = get_fields("aten::sin", 0)
        c_impl_new, c_storage_data_new, c_id_new = get_fields("aten::cos", 0)
        self.assertNotEqual(d_impl, c_impl_new)
        self.assertEqual(d_storage_data, c_storage_data_new)
        self.assertEqual(c_id, c_id_new)
        self.assertEqual(d_id, c_id_new)

    @staticmethod
    def _format_allocations(profiled_code):
        # [FUNCTION DOC] HELPER: Static. Runs a garbage collection, profiles the callable with
        # memory and shape recording followed by another collection, and takes all allocation
        # events from the event tree sorted by start time. Returns a string with one line per
        # event holding the tensor id, allocation id and the word Allocation or Free (by sign of
        # size), indented by twelve spaces.
        gc.collect()
        with profile(profile_memory=True, record_shapes=True) as prof:
            profiled_code()
            gc.collect()

        root_events = prof.profiler.kineto_results.experimental_event_tree()
        events = sorted(_utils.traverse_dfs(root_events), key=lambda x: x.start_time_ns)
        allocations = tuple(
            event.extra_fields
            for event in events
            if isinstance(
                event.extra_fields, torch._C._profiler._ExtraFields_Allocation
            )
        )

        return textwrap.indent(
            "\n".join(
                f"{repr(i.id):>5}{' ' * 6}"
                f"{repr(i.allocation_id):>5}{' ' * 6}"
                f"{'Allocation' if i.alloc_size > 0 else 'Free'}"
                for i in allocations
            ),
            " " * 12,
        )

    def test_tensorimpl_invalidation_set(self) -> None:
        # [FUNCTION DOC] Checks allocation and free ordering and ids when a tensor is pointed
        # at a new storage via set_, with and without a preceding empty set_ that frees the
        # old storage first. Compares formatted allocation tables against expected strings
        # for both variants.
        def profiled_code(add_empty_set: bool):
            # [FUNCTION DOC] HELPER: Creates a one-element tensor, optionally calls an
            # argumentless set_ (controlling whether the old storage is released before the new
            # one is created), then sets it to a new storage and calls view_as.
            x = torch.ones((1,))

            # Determines if new storage is created before or after the old one
            # is destroyed.
            if add_empty_set:
                x.set_()

            x.set_(torch.ones((1,)).storage())
            x.view_as(x)

        self.assertExpectedInline(
            self._format_allocations(lambda: profiled_code(add_empty_set=False)),
            """\
                0          1      Allocation
                0          2      Allocation
                0          1      Free
                0          2      Free""",
        )

        self.assertExpectedInline(
            self._format_allocations(lambda: profiled_code(add_empty_set=True)),
            """\
                0          1      Allocation
                0          1      Free
                0          2      Allocation
                0          2      Free""",
        )

    def test_tensorimpl_invalidation_keep_alive(self) -> None:
        # [FUNCTION DOC] Checks allocation ids when old storages are kept alive in a list across
        # repeated set_ calls and then released deterministically with garbage collection;
        # compares the formatted allocation table to expected text, with and without an extra
        # empty set_.
        def profiled_code(add_empty_set: bool):
            # [FUNCTION DOC] HELPER: Repeatedly re-sets a tensor to new storages while
            # keeping the old storages in a list, frees them one at a time with collections,
            # optionally does an empty set_, sets storage three more times, then deletes the
            # tensor and collects.
            x = torch.ones((1,))
            x_storages = [x.storage()]
            for _ in range(3):
                x.set_()
                x.set_(torch.ones((1,)).storage())

                # This keeps the StorageImpls alive and preserves the chain.
                # (Despite the `set_()` call.)
                x_storages.append(x.storage())
            x.view_as(x)

            # Free storage in a deterministic fashion.
            while x_storages:
                x_storages.pop()
                gc.collect()

            # Determines if new storage is created before or after the old one
            # is destroyed.
            if add_empty_set:
                x.set_()

            for _ in range(3):
                x.set_(torch.ones((1,)).storage())
            x.view_as(x)

            del x
            gc.collect()

        self.assertExpectedInline(
            self._format_allocations(lambda: profiled_code(add_empty_set=False)),
            """\
                0          1      Allocation
                0          2      Allocation
                0          4      Allocation
                0          5      Allocation
                0          4      Free
                0          2      Free
                0          1      Free
                0          6      Allocation
                0          5      Free
                0          7      Allocation
                0          6      Free
                0          8      Allocation
                0          7      Free
                0          8      Free""",
        )

        self.assertExpectedInline(
            self._format_allocations(lambda: profiled_code(add_empty_set=True)),
            """\
                0          1      Allocation
                0          2      Allocation
                0          4      Allocation
                0          5      Allocation
                0          4      Free
                0          2      Free
                0          1      Free
                0          5      Free
                0          6      Allocation
                0          7      Allocation
                0          6      Free
                0          8      Allocation
                0          7      Free
                0          8      Free""",
        )

    def test_tensorimpl_invalidation_full(self) -> None:
        # [FUNCTION DOC] Combined scenario of kept-alive storages, repeated set_ (with and
        # without empty set_ calls) and resize_ steps; compares the formatted allocation table
        # against the expected text.
        def profiled_code():
            # [FUNCTION DOC] HELPER: Runs the combined set_ / resize_ workload on a one-element
            # tensor, releasing kept storages in a deterministic order.
            x = torch.ones((1,))
            x_storages = [x.storage()]
            for _ in range(3):
                x.set_()
                x.set_(torch.ones((1,)).storage())
                x_storages.append(x.storage())
            x.view_as(x)

            # Free storage in a deterministic fashion.
            while x_storages:
                x_storages.pop()
                gc.collect()

            for _ in range(3):
                x.set_(torch.ones((1,)).storage())

            for _ in range(3):
                x.set_()
                x.set_(torch.ones((1,)).storage())

            for i in range(4):
                x.resize_((1 + i,))
            x.view_as(x)

        self.assertExpectedInline(
            self._format_allocations(profiled_code),
            """\
                0          1      Allocation
                0          2      Allocation
                0          4      Allocation
                0          5      Allocation
                0          4      Free
                0          2      Free
                0          1      Free
                0          6      Allocation
                0          5      Free
                0          7      Allocation
                0          6      Free
                0          8      Allocation
                0          7      Free
                0          8      Free
                0          9      Allocation
                0          9      Free
                0         10      Allocation
                0         10      Free
                0         11      Allocation
                0         12      Allocation
                0         11      Free
                0         13      Allocation
                0         12      Free
                0         14      Allocation
                0         13      Free
                0         14      Free""",
        )

    def test_tensorimpl_invalidation_scalar_args(self) -> None:
        # [FUNCTION DOC] Checks allocation tables for repeated in-place adds with a scalar
        # argument under no-grad: each add yields allocation events for the scalar wrapper tensor
        # with increasing ids, compared to expected text.
        def profiled_code():
            # [FUNCTION DOC] HELPER: Under no-grad, creates a one-element tensor and adds the
            # number two to it in place ten times.
            with torch.no_grad():
                x = torch.ones((1,))
                for _ in range(10):
                    x.add_(2)

        self.assertExpectedInline(
            self._format_allocations(profiled_code),
            """\
                0          1      Allocation
                1          2      Allocation
                2          3      Allocation
                2          3      Free
                1          2      Free
                3          4      Allocation
                4          5      Allocation
                4          5      Free
                3          4      Free
                5          6      Allocation
                6          7      Allocation
                6          7      Free
                5          6      Free
                7          8      Allocation
                8          9      Allocation
                8          9      Free
                7          8      Free
                9         10      Allocation
               10         11      Allocation
               10         11      Free
                9         10      Free
               11         12      Allocation
               12         13      Allocation
               12         13      Free
               11         12      Free
               13         14      Allocation
               14         15      Allocation
               14         15      Free
               13         14      Free
               15         16      Allocation
               16         17      Allocation
               16         17      Free
               15         16      Free
               17         18      Allocation
               18         19      Allocation
               18         19      Free
               17         18      Free
               19         20      Allocation
               20         21      Allocation
               20         21      Free
               19         20      Free
                0          1      Free""",
        )

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO or IS_LINUX or IS_MACOS or IS_WINDOWS,
        "https://github.com/pytorch/pytorch/issues/87581",
    )
    def test_module_and_optimizer_ids(self) -> None:
        # [FUNCTION DOC] Profiles a linear model forward, backward and an SGD step and, for a
        # cold start (first step, weight grad not yet present) and warm run, verifies tensor ids
        # captured for module parameters and optimizer parameters and state match ground truth
        # tensors marked by unary ops. Skipped on every supported OS and under dynamo (tracking
        # issue 87581), so it effectively never runs.
        model = torch.nn.Linear(2, 1, bias=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

        def check(cold_start: bool) -> None:
            # [FUNCTION DOC] HELPER: Runs one profiled iteration and asserts: ids of marked
            # tensors are distinct; the module event for the linear layer exposes weight with the
            # matching id and a grad id only when not a cold start; the optimizer step event
            # exposes two parameters (weight and bias) with their grad ids and a single momentum
            # buffer state with the marked id.
            with profile(with_stack=True, profile_memory=True, record_shapes=True) as p:
                x = torch.ones((1, 2))
                _ = x.sin()  # Mark `x`
                model(x).backward()
                optimizer.step()
                _ = optimizer.state[model.weight][
                    "momentum_buffer"
                ].cos()  # Mark weight momentum
                _ = model.weight.grad.tan()  # Mark weight gradient

            nodes = p.profiler.kineto_results.experimental_event_tree()

            def get_fields(op_name, index):
                # [FUNCTION DOC] HELPER: Returns impl pointer, storage pointer and id of the
                # input at the given index of the first node with the given op name.
                return self._get_tensor_fields(
                    find_node_with_name(nodes, op_name), index
                )

            # Marked Tensors act as ground truth for python tracer IDs.
            _, _, x_id = get_fields("aten::sin", 0)
            _, _, weight_momenumtum_id = get_fields("aten::cos", 0)
            _, _, weight_grad_id = get_fields("aten::tan", 0)
            self.assertNotEqual(x_id, weight_momenumtum_id)
            self.assertNotEqual(x_id, weight_grad_id)
            self.assertNotEqual(weight_momenumtum_id, weight_grad_id)

            # Use linear op to identify weight ground truth.
            linear_op_node = find_node_with_name(nodes, "aten::linear")
            self.assertIsNotNone(linear_op_node)
            x_metadata, weight_metadata, _ = linear_op_node.extra_fields.inputs
            self.assertEqual(x_id, x_metadata.id)

            # Module
            linear_module_node = find_node_with_name(nodes, "nn.Module: Linear_0")
            self.assertIsNotNone(linear_module_node)
            self.assertIsNotNone(linear_module_node.extra_fields.module)
            self.assertIsNone(linear_module_node.extra_fields.optimizer)

            linear_parameters = linear_module_node.extra_fields.module.parameters
            name, weight, weight_grad = linear_parameters[0]
            self.assertEqual(name, "weight")
            self.assertEqual(weight.id, weight_metadata.id)

            self.assertEqual(weight_grad is None, cold_start)
            if not cold_start:
                self.assertEqual(weight_grad.id, weight_grad_id)

            # Optimizer
            step_node = find_node_with_regex(nodes, "_optimizer_step_code")
            self.assertIsNotNone(step_node)
            self.assertIsNone(step_node.extra_fields.module)
            self.assertIsNotNone(step_node.extra_fields.optimizer)
            optimizer_parameters = step_node.extra_fields.optimizer.parameters
            self.assertEqual(len(optimizer_parameters), 2)  # Weight and bias
            weight, weight_grad, state = optimizer_parameters[0]
            self.assertEqual(weight.id, weight_metadata.id)
            self.assertEqual(weight_grad.id, weight_grad_id)
            self.assertEqual(len(state), 1)
            self.assertEqual(state[0][0], "momentum_buffer")
            self.assertEqual(state[0][1].id, weight_momenumtum_id)

        # Check that we handle first step (lazy initialization) and steady state.
        check(cold_start=True)
        check(cold_start=False)

    def _test_allocation_ids(self, before_fn, after_fn) -> None:
        # [FUNCTION DOC] HELPER: Profiles a random tensor created, resized, used and deleted,
        # with the two given callables run before and after to add unrelated work. Checks the
        # allocation event under rand and empty matches the uniform fill input pointer and id;
        # the resize has two children (new allocation, free of the old) keeping the id with a new
        # pointer; and the final top-level allocation event is the free of the new storage with
        # the same id and pointer.
        with profile(profile_memory=True, record_shapes=True) as p:
            # Introduce other operations and allocations to check robustness
            _ = before_fn()

            x = torch.rand(4, 3)
            x.resize_(4, 4)

            # We need to use `x` post resize for profiler to determine its ID.
            x.sin()

            # Introduce other operations and allocations to check robustness
            _ = after_fn()

            # Ensure `x` is the last variable collected to make it easier to
            # find the deallocation event.
            gc.collect()
            del x
            gc.collect()

        nodes = p.profiler.kineto_results.experimental_event_tree()

        def find_chain(names: list[str]):
            # [FUNCTION DOC] HELPER: Given a list of op names, finds each successive node
            # starting from the roots and then in the previous found node's subtree, asserting
            # each is found. Returns the list of nodes.
            out = []
            for name in names:
                root = [out[-1]] if out else nodes
                out.append(find_node_with_name(root, name))
                self.assertIsNotNone(out[-1], name)
            return out

        allocation = find_chain(["aten::rand", "aten::empty", "[memory]"])[
            -1
        ].extra_fields
        _, uniform_node = find_chain(["aten::rand", "aten::uniform_"])
        _, x_storage_data, x_id = self._get_tensor_fields(uniform_node, 0)

        # Make sure IDs are consistent between allocations and op inputs
        self.assertEqual(allocation.ptr, x_storage_data)
        self.assertEqual(allocation.id, x_id)

        resize_node = find_node_with_name(nodes, "aten::resize_")
        self.assertIsNotNone(resize_node)
        self.assertEqual(len(resize_node.children), 2)
        allocate_new = resize_node.children[0].extra_fields
        free_old = resize_node.children[1].extra_fields

        # Destruction of the old storage for x.
        self.assertEqual(free_old.id, allocation.id)
        self.assertEqual(free_old.ptr, allocation.ptr)

        # Make sure ID is retained through change in storage.
        self.assertEqual(allocate_new.id, allocation.id)
        self.assertNotEqual(allocate_new.ptr, allocation.ptr)

        # Deletion when `x` goes out of scope.
        free_new = [
            i for i in nodes if i.tag == torch._C._profiler._EventType.Allocation
        ][-1].extra_fields
        self.assertIsInstance(free_new, torch._C._profiler._ExtraFields_Allocation)
        self.assertEqual(free_new.id, allocate_new.id)
        self.assertEqual(free_new.ptr, allocate_new.ptr)

    def test_allocation_ids(self) -> None:
        # [FUNCTION DOC] Runs the allocation id check with no extra work before or after.
        self._test_allocation_ids(lambda: None, lambda: None)

    def test_allocation_ids_with_other_ops(self) -> None:
        # [FUNCTION DOC] Runs the allocation id check with extra ops (add then in-place relu, and
        # zeros then cos) before and after.
        x = torch.ones((1,))
        self._test_allocation_ids(
            lambda: (x + 1).relu_(), lambda: torch.zeros((1,)).cos()
        )

    def test_impl_reuse(self) -> None:
        # [FUNCTION DOC] Profiles a thousand short-lived ones tensors and asserts the thousand
        # fill ops have a thousand distinct impl pointers, i.e. tensor impl addresses are not
        # reported as reused.
        repeats = 1_000
        with profile(profile_memory=True, record_shapes=True) as p:
            for _ in range(repeats):
                torch.ones((1,))
            gc.collect()

        roots = p.profiler.kineto_results.experimental_event_tree()
        tensor_impls = tuple(
            e.extra_fields.inputs[0].impl_ptr
            for e in _utils.traverse_dfs(roots)
            if e.name == "aten::fill_"
        )

        self.assertEqual(len(tensor_impls), repeats)
        self.assertEqual(len(set(tensor_impls)), repeats)

    def test_allocation_id_uniqueness(self) -> None:
        # [FUNCTION DOC] Profiles a thousand short-lived ones tensors and gathers allocation ids
        # from tensor inputs of ops and from allocation events, ignoring None. Asserts the set
        # has exactly a thousand entries, i.e. ids are unique per allocation.
        repeats = 1_000
        with profile(profile_memory=True, record_shapes=True) as p:
            for _ in range(repeats):
                torch.ones((1,))
            gc.collect()

        roots = p.profiler.kineto_results.experimental_event_tree()
        id_set = set()
        for e in _utils.traverse_dfs(roots):
            fields = e.extra_fields
            if isinstance(fields, torch._C._profiler._ExtraFields_TorchOp):
                id_set |= {
                    t.allocation_id
                    for t in fields.inputs
                    if isinstance(t, _TensorMetadata)
                }

            elif isinstance(fields, torch._C._profiler._ExtraFields_Allocation):
                id_set.add(fields.allocation_id)

        id_set.difference_update([None])
        self.assertEqual(repeats, len(id_set))

    def test_extra_fields(self):
        # [FUNCTION DOC] Profiles creation of a ones tensor with stack and memory profiling and
        # checks the node for the op has torch-op fields, its parent has Python C-call fields,
        # its first child is the empty op, and that has a memory child with allocation fields.
        with profile(with_stack=True, profile_memory=True) as p:
            _ = torch.ones((1,))

        nodes = p.profiler.kineto_results.experimental_event_tree()
        node = find_node_with_name(nodes, "aten::ones")
        self.assertIsNotNone(node)

        self.assertIsInstance(
            node.extra_fields, torch._C._profiler._ExtraFields_TorchOp
        )

        self.assertIsInstance(
            node.parent.extra_fields, torch._C._profiler._ExtraFields_PyCCall
        )

        self.assertEqual(node.children[0].name, "aten::empty")
        self.assertEqual(node.children[0].children[0].name, "[memory]")
        self.assertIsInstance(
            node.children[0].children[0].extra_fields,
            torch._C._profiler._ExtraFields_Allocation,
        )

    def test_tensor_properties(self):
        # [FUNCTION DOC] Profiles an add and a mul of a strided view and a grad-requiring tensor.
        # Asserts the add node's input sizes, strides, layouts, devices and dtypes (the trailing
        # non-tensor argument gives defaults), its record scope is function, and the mul's
        # sequence number is one more than the add's.
        x = torch.ones(10, 10).as_strided([4, 4], [12, 3])
        y = torch.ones(4, 1, requires_grad=True)

        with profile(with_stack=True, profile_memory=True, record_shapes=True) as p:
            _ = x + y
            _ = x * y

        nodes = p.profiler.kineto_results.experimental_event_tree()
        node = find_node_with_name(nodes, "aten::add")
        self.assertIsNotNone(node)

        self.assertIsInstance(
            node.extra_fields, torch._C._profiler._ExtraFields_TorchOp
        )

        def getattr_inputs(name, default):
        # [FUNCTION DOC] HELPER: Returns, for each input of the add node, the named attribute or
        # the given default if the input lacks it.
            return [getattr(i, name, default) for i in node.extra_fields.inputs]

        self.assertEqual(getattr_inputs("sizes", []), [[4, 4], [4, 1], []])
        self.assertEqual(getattr_inputs("strides", []), [[12, 3], [1, 1], []])
        self.assertEqual(
            getattr_inputs("layout", None), [torch.strided, torch.strided, None]
        )
        self.assertEqual(
            getattr_inputs("device", None),
            [torch.device("cpu"), torch.device("cpu"), None],
        )
        self.assertEqual(
            getattr_inputs("dtype", None), [torch.float32, torch.float32, None]
        )
        self.assertEqual(node.extra_fields.scope, torch.profiler.RecordScope.FUNCTION)

        mul_node = find_node_with_name(nodes, "aten::mul")
        self.assertIsNotNone(mul_node)
        self.assertEqual(
            node.extra_fields.sequence_number + 1, mul_node.extra_fields.sequence_number
        )

    def test_sparse_tensors(self):
        # [FUNCTION DOC] Profiles adding a sparse COO tensor to itself and asserts the inputs
        # report full sizes, empty strides, sparse COO layout and cpu device (trailing argument
        # gives defaults).
        i = [[0, 1, 1], [2, 0, 2]]
        v = [3, 4, 5]
        s = torch.sparse_coo_tensor(i, v, (2, 3))

        with profile(with_stack=True, profile_memory=True, record_shapes=True) as p:
            _ = s + s

        nodes = p.profiler.kineto_results.experimental_event_tree()
        node = find_node_with_name(nodes, "aten::add")
        self.assertIsNotNone(node)

        self.assertIsInstance(
            node.extra_fields, torch._C._profiler._ExtraFields_TorchOp
        )

        def getattr_inputs(name, default):
        # [FUNCTION DOC] HELPER: Returns, for each input of the add node, the named attribute or
        # the given default if the input lacks it.
            return [getattr(i, name, default) for i in node.extra_fields.inputs]

        self.assertEqual(getattr_inputs("sizes", []), [[2, 3], [2, 3], []])
        self.assertEqual(getattr_inputs("strides", []), [[], [], []])
        self.assertEqual(
            getattr_inputs("layout", None), [torch.sparse_coo, torch.sparse_coo, None]
        )
        self.assertEqual(
            getattr_inputs("device", None),
            [torch.device("cpu"), torch.device("cpu"), None],
        )

    @unittest.skipIf(
        not torch.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    def test_mkldnn_tensors(self):
        # [FUNCTION DOC] Profiles adding an MKL-DNN tensor to itself and asserts sizes,
        # empty strides, the mkldnn layout and cpu device for the inputs. Skipped when
        # MKL-DNN is unavailable.
        x = torch.ones(4, 3).to_mkldnn()

        with profile(with_stack=True, profile_memory=True, record_shapes=True) as p:
            _ = x + x

        nodes = p.profiler.kineto_results.experimental_event_tree()
        node = find_node_with_name(nodes, "aten::add")
        self.assertIsNotNone(node)

        self.assertIsInstance(
            node.extra_fields, torch._C._profiler._ExtraFields_TorchOp
        )

        def getattr_inputs(name, default):
        # [FUNCTION DOC] HELPER: Returns, for each input of the add node, the named attribute or
        # the given default if the input lacks it.
            return [getattr(i, name, default) for i in node.extra_fields.inputs]

        self.assertEqual(getattr_inputs("sizes", []), [[4, 3], [4, 3], []])
        self.assertEqual(getattr_inputs("strides", []), [[], [], []])
        self.assertEqual(
            getattr_inputs("layout", None), [torch._mkldnn, torch._mkldnn, None]
        )
        self.assertEqual(
            getattr_inputs("device", None),
            [torch.device("cpu"), torch.device("cpu"), None],
        )

    def test_scalar_ins(self):
        # [FUNCTION DOC] Profiles adding a Python float to a float tensor with an alpha keyword.
        # Asserts the float is promoted to a zero-dimensional double tensor input (dtype and
        # sizes), and that the third input is the alpha value itself.
        x = torch.ones(5, 5)
        alpha = 0.9

        with profile(with_stack=True, profile_memory=True, record_shapes=True) as p:
            _ = torch.add(x, 9.1, alpha=alpha)

        nodes = p.profiler.kineto_results.experimental_event_tree()
        node = find_node_with_name(nodes, "aten::add")
        self.assertIsNotNone(node)

        def getattr_inputs(name, default):
        # [FUNCTION DOC] HELPER: Returns, for each input of the add node, the named attribute or
        # the given default if the input lacks it.
            return [getattr(i, name, default) for i in node.extra_fields.inputs]

        # The second argument to the add gets promotoed to a zerodim Tensor
        self.assertEqual(
            getattr_inputs("dtype", None), [torch.float32, torch.float64, None]
        )
        self.assertEqual(getattr_inputs("sizes", []), [[5, 5], [], []])
        self.assertEqual(node.extra_fields.inputs[2], alpha)

    def test_tensor_lists(self):
        # [FUNCTION DOC] Profiles stacking two tensors and asserts the stack op has two
        # inputs, the first a list of two tensor metadata entries whose storage pointers
        # match the tensors.
        x = torch.ones((1,))
        y = torch.ones((1,))
        with profile(with_stack=True, profile_memory=True, record_shapes=True) as p:
            _ = torch.stack((x, y))

        nodes = p.profiler.kineto_results.experimental_event_tree()
        node = find_node_with_name(nodes, "aten::stack")
        inputs = node.extra_fields.inputs
        self.assertEqual(len(inputs), 2)
        self.assertIsInstance(inputs[0], list)
        self.assertEqual(len(inputs[0]), 2)
        self.assertEqual(x.storage().data_ptr(), inputs[0][0].storage_data_ptr)
        self.assertEqual(y.storage().data_ptr(), inputs[0][1].storage_data_ptr)

    def test_nnmodule_params(self):
        # [FUNCTION DOC] Profiles a forward pass of the two-layer net after a backward (so grads
        # exist) and checks that exactly two module records with parameters are captured and that
        # their (name, storage pointer, grad storage pointer) triples equal those of the first
        # then second layer parameters.
        def flat_out_extrafields(nodes, out=None):
            # [FUNCTION DOC] HELPER: Recursively collects module info objects, from Python call
            # nodes that have a module with a non-empty parameter list, across the tree into an
            # output list (created if absent) and returns it.
            if out is None:
                out = []
            for node in nodes:
                if (
                    isinstance(node.extra_fields, _ExtraFields_PyCall)
                    and node.extra_fields.module
                ):
                    if node.extra_fields.module.parameters:
                        out.append(node.extra_fields.module)
                flat_out_extrafields(node.children, out)
            return out

        inputs = torch.rand(10)
        net = SimpleNet()
        out = net(inputs)
        torch.nn.functional.cross_entropy(out, torch.rand(2)).backward()
        with torch.profiler.profile(with_stack=True, profile_memory=True) as p:
            _ = net(inputs)

        modules = flat_out_extrafields(
            p.profiler.kineto_results.experimental_event_tree()
        )
        self.assertEqual(
            len(modules),
            2,
            lambda msg: f"{msg}\nExpected two parameter list, but got {len(modules)}",
        )

        params = [
            (n, p.storage_data_ptr, g.storage_data_ptr)
            for module in modules
            for (n, p, g) in module.parameters
        ]
        expected = [
            (name, val.storage().data_ptr(), val.grad.storage().data_ptr())
            for name, val in net.fc1._parameters.items()
        ]
        expected += [
            (name, val.storage().data_ptr(), val.grad.storage().data_ptr())
            for name, val in net.fc2._parameters.items()
        ]
        self.assertEqual(
            expected, params, lambda msg: f"{msg}\n{expected} vs. {params}"
        )

    def _flat_out_extrafields(self, nodes, out=None):
        # [FUNCTION DOC] HELPER: Recursively collects optimizer info objects from Python call
        # nodes with an optimizer having parameters, skipping any whose first parameter storage
        # pointer duplicates one already collected (repeated iterations). Returns the list.
        if out is None:
            out = []
        for node in nodes:
            if (
                isinstance(node.extra_fields, _ExtraFields_PyCall)
                and node.extra_fields.optimizer
                and node.extra_fields.optimizer.parameters
            ):
                # avoiding OptInfo duplicates from iterations
                addr = node.extra_fields.optimizer.parameters[0][0].storage_data_ptr
                if not [o for o in out if addr == o.parameters[0][0].storage_data_ptr]:
                    out.append(node.extra_fields.optimizer)
            self._flat_out_extrafields(node.children, out)
        return out

    def _check_results(self, opt, opts, check_items=False):
        # [FUNCTION DOC] HELPER: Asserts exactly one profiled optimizer was found and that its
        # recorded self pointer equals the identity of the real optimizer. If the flag is true,
        # also checks the parameter group count equals the optimizer count, each group's
        # parameter storage pointers match the recorded ones, and recorded state storage pointers
        # match the real state for every parameter.
        self.assertEqual(
            len(opts),
            1,
            lambda msg: f"{msg}\nExpected 1 optimizer: len(opts): {len(opts)}",
        )
        self.assertEqual(
            id(opt),
            opts[0].self_ptr,
            lambda msg: f"{msg}\nOptimizer addr ({id(opt)}) vs. profiled addr ({opts[0].self_ptr})",
        )
        if check_items:
            self.assertEqual(len(opt.param_groups), len(opts))
            for group, opt_ in zip(opt.param_groups, opts):
                self.assertEqual(
                    [(v.storage().data_ptr()) for v in group.get("params", [])],
                    [(o.storage_data_ptr) for (o, _, _) in opt_.parameters],
                )
            for opt_ in opts:
                observed_state = {
                    p.storage_data_ptr: {name: s.storage_data_ptr for name, s in state}
                    for (p, _, state) in opt_.parameters
                }

                # Make sure the profiler collected all optimizer state and check
                # that the address recorded by the profiler is correct.
                for parameter, parameter_state in opt.state.items():
                    self.assertEqual(
                        {
                            name: value.storage().data_ptr()
                            for name, value in parameter_state.items()
                        },
                        observed_state.get(parameter.storage().data_ptr(), []),
                    )

    def test_optimizer(self):
        # [FUNCTION DOC] Profiles building a net and SGD optimizer and one training step, then
        # checks the captured optimizer record (identity only, without items).
        inputs = torch.rand(10)
        with torch.profiler.profile(with_stack=True, profile_memory=True) as p:
            net = SimpleNet()
            opt = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

            opt.zero_grad()
            out = net(inputs)
            loss = torch.nn.functional.cross_entropy(out, torch.rand(2))
            loss.backward()
            opt.step()
        self._check_results(
            opt,
            self._flat_out_extrafields(
                p.profiler.kineto_results.experimental_event_tree()
            ),
            False,
        )

    def _test_optimizer_parameters(self, optimizer_factory):
        # [FUNCTION DOC] HELPER: Profiles building a net and an optimizer from the given factory
        # and two training steps, then runs the full check of identity, parameters and state.
        inputs = torch.rand(10)
        with torch.profiler.profile(with_stack=True, profile_memory=True) as p:
            net = SimpleNet()
            opt = optimizer_factory(net.parameters())
            for _ in range(2):
                opt.zero_grad()
                out = net(inputs)
                loss = torch.nn.functional.cross_entropy(out, torch.rand(2))
                loss.backward()
                opt.step()
        self._check_results(
            opt,
            self._flat_out_extrafields(
                p.profiler.kineto_results.experimental_event_tree()
            ),
            True,
        )

    def test_optimizer_parameters_sgd(self):
        # [FUNCTION DOC] Runs the optimizer parameter check with SGD with momentum.
        self._test_optimizer_parameters(
            lambda params: torch.optim.SGD(params, lr=0.01, momentum=0.9)
        )

    def test_optimizer_parameters_adam(self):
        # [FUNCTION DOC] Runs the optimizer parameter check with Adam using foreach.
        self._test_optimizer_parameters(
            lambda params: torch.optim.Adam(params, foreach=True)
        )

    def test_allocations(self):
        # [FUNCTION DOC] Profiles allocating then freeing a 3x4 float tensor. Asserts the memory
        # event has a positive pointer, size twelve floats in bytes (48), cpu device, and zero
        # total reserved; then in a second profile the free event has the same pointer, the
        # negative size and total allocated reduced by that size.
        gc.collect()
        with profile(profile_memory=True) as p:
            x = torch.empty((3, 4))

        nodes = p.profiler.kineto_results.experimental_event_tree()
        node = find_node_with_name(nodes, "[memory]")
        self.assertIsNotNone(node)

        alloc_size = 3 * 4 * 4  # fp32 -> 4 bytes
        ptr = node.extra_fields.ptr
        self.assertGreater(ptr, 0)
        self.assertEqual(node.extra_fields.alloc_size, alloc_size)
        self.assertEqual(node.extra_fields.device, torch.device("cpu"))
        total_allocated = node.extra_fields.total_allocated

        # total_reserved is only for CUDACachingAllocator
        self.assertEqual(node.extra_fields.total_reserved, 0)

        with profile(profile_memory=True) as p:
            del x
            gc.collect()

        nodes = p.profiler.kineto_results.experimental_event_tree()
        node = find_node_with_name(nodes, "[memory]")
        self.assertIsNotNone(node)

        self.assertEqual(node.extra_fields.ptr, ptr)
        self.assertEqual(node.extra_fields.alloc_size, -alloc_size)
        self.assertEqual(node.extra_fields.device, torch.device("cpu"))
        self.assertEqual(
            node.extra_fields.total_allocated, total_allocated - alloc_size
        )

    def test_refcounts(self):
        # [FUNCTION DOC] Checks that profiling with stack does not leak references through
        # closures: after calling the outer function, the inner sentinel is gone, while the outer
        # sentinel stays alive until the outer function is deleted.
        class Sentinel:
            # [CLASS DOC] HELPER: Empty marker class whose instances are tracked with
            # weak references.
            pass

        def make():
            # [FUNCTION DOC] HELPER: Creates an outer sentinel and a nested function; returns
            # that function and a weak reference to the sentinel.
            outer_sentinel = Sentinel()

            def outer():
                # [FUNCTION DOC] HELPER: Touches the outer sentinel to capture it, creates an
                # inner sentinel, runs a nested function using it under a stack-recording
                # profiler, and returns a weak reference to the inner sentinel.
                # Python will only close over variables used in the function.
                _ = outer_sentinel
                inner_sentinel = Sentinel()

                def inner():
                    # [FUNCTION DOC] HELPER: Reads the inner sentinel so the closure captures it;
                    # no result.
                    _ = inner_sentinel

                with profile(with_stack=True):
                    inner()

                return weakref.ref(inner_sentinel)

            return outer, weakref.ref(outer_sentinel)

        # Use a factory function to ensure the test scope never sees strong
        # references. `del` has strange semantics that interact with closures
        # at an AST level, so this is simpler.
        outer, outer_sentinel_ref = make()
        inner_sentinel_ref = outer()

        self.assertIsNone(inner_sentinel_ref())

        # `outer` holds the last reference via closure.
        self.assertIsNotNone(outer_sentinel_ref())

        del outer
        self.assertIsNone(outer_sentinel_ref())


if __name__ == "__main__":
    run_tests()
