# Owner(s): ["module: functorch"]
# [MODULE DOC] Inductor tests for TorchBind (script object) support: compiling and exporting
# modules that hold custom class objects, the call_torchbind schema, buffer byte sizes, and
# AOTInductor packaging and loading. Imports have no side effects; tests register torchbind
# implementations in setUp.
import json
import zipfile
from pathlib import Path

import torch
import torch._dynamo
import torch._functorch
import torch._inductor
import torch._inductor.decomposition
from torch._higher_order_ops.torchbind import CallTorchBind, enable_torchbind_tracing
from torch._inductor import aot_compile, ir
from torch._inductor.codecache import WritableTempFile
from torch._inductor.package import package_aoti
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import skipIfWindows
from torch.testing._internal.inductor_utils import GPU_TYPE, requires_gpu
from torch.testing._internal.torchbind_impls import (
    _empty_tensor_queue,
    init_torchbind_implementations,
)


class TestTorchbind(TestCase):
    # [CLASS DOC] Test case for TorchBind support in Inductor and AOTInductor. setUp registers the
    # test torchbind implementations. Helper methods build exported models; tests compile or
    # package them and compare against eager results.
    def setUp(self):
        # [FUNCTION DOC] Calls the parent setUp, then registers the torchbind test implementations.
        super().setUp()
        init_torchbind_implementations()

    def get_dummy_exported_model(self):
        # [FUNCTION DOC] Exports a trivial module that adds one to its input (no torchbind
        # objects); returns the exported program, the example inputs, the eager result and
        # the module.
        """
        Returns the ExportedProgram, example inputs, and result from calling the
        eager model with those inputs
        """

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward adds one to the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input plus one.
                return x + 1

        m = M()
        inputs = (torch.ones(2, 3),)
        orig_res = m(*inputs)

        ep = torch.export.export(m, inputs, strict=False)

        return ep, inputs, orig_res, m

    def get_exported_model(self):
        # [FUNCTION DOC] Exports a small module that holds a Foo torchbind object and a random
        # tensor and calls two custom ops and a method on the object, using torchbind tracing;
        # returns the exported program, the inputs, the eager result and the module.
        """
        Returns the ExportedProgram, example inputs, and result from calling the
        eager model with those inputs
        """

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a Foo object of 10 and 20 and a random 2x3
            # tensor buffer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the Foo torchbind object and the random tensor.
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)
                self.b = torch.randn(2, 3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds the tensor to the input, calls two custom ops
                # on the Foo object, calls its add_tensor method, and returns the sum of the
                # input and the results.
                x = x + self.b
                a = torch.ops._TorchScriptTesting.takes_foo_tuple_return(self.attr, x)
                y = a[0] + a[1]
                b = torch.ops._TorchScriptTesting.takes_foo(self.attr, y)
                c = self.attr.add_tensor(x)
                return x + b + c

        m = M()
        inputs = (torch.ones(2, 3),)
        orig_res = m(*inputs)

        # We can't directly torch.compile because dynamo doesn't trace ScriptObjects yet
        with enable_torchbind_tracing():
            ep = torch.export.export(m, inputs, strict=False)

        return ep, inputs, orig_res, m

    def test_torchbind_inductor(self):
        # [FUNCTION DOC] Compiles the exported torchbind module with Inductor and checks the output
        # matches eager.
        ep, inputs, orig_res, _ = self.get_exported_model()
        compiled = torch._inductor.compile(ep.module(), inputs)

        new_res = compiled(*inputs)
        self.assertTrue(torch.allclose(orig_res, new_res))

    def test_torchbind_compile_symint(self):
        # [FUNCTION DOC] Compiles with the inductor backend a module that calls a custom op on a
        # Foo object returning a tensor, and checks the output is close to eager.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a Foo object of 2 and 3 that calls a custom op
            # returning a tensor.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the Foo torchbind object.
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(2, 3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the tensor-returning custom op on the object and
                # returns the result.
                a = torch.ops._TorchScriptTesting.takes_foo_tensor_return(self.attr, x)
                return a

        m = M()
        inputs = (torch.ones(2, 3),)
        orig_res = m(*inputs)
        new_res = torch.compile(m, backend="inductor")(*inputs)
        self.assertTrue(torch.allclose(orig_res, new_res))

    def test_torchbind_compile(self):
        # [FUNCTION DOC] Compiles the original torchbind module with torch.compile and the inductor
        # backend and checks the output is close to eager.
        _, inputs, orig_res, mod = self.get_exported_model()
        new_res = torch.compile(mod, backend="inductor")(*inputs)
        self.assertTrue(torch.allclose(orig_res, new_res))

    def test_torchbind_get_buf_bytes(self):
        # [FUNCTION DOC] Checks get_buf_bytes of a TorchBindObject: zero for a plain Foo, 24 bytes
        # (2x3 float elements) for a ContainsTensor, and for a tensor queue zero when empty and 24
        # after pushing a 2x3 tensor.
        a = torch.classes._TorchScriptTesting._Foo(10, 20)
        buffer = ir.TorchBindObject(name="a", value=a)
        size = buffer.get_buf_bytes()
        self.assertEqual(size, 0)

        t = torch.randn(2, 3)
        b = torch.classes._TorchScriptTesting._ContainsTensor(t)
        buffer = ir.TorchBindObject(name="b", value=b)
        size = buffer.get_buf_bytes()
        self.assertEqual(size, 2 * 3 * 4)

        q = _empty_tensor_queue()
        buffer = ir.TorchBindObject(name="q", value=q)
        size = buffer.get_buf_bytes()
        self.assertEqual(size, 0)

        q.push(torch.ones(2, 3))
        size = buffer.get_buf_bytes()
        self.assertEqual(size, 2 * 3 * 4)

    def test_torchbind_hop_schema(self):
        # [FUNCTION DOC] Checks the call_torchbind HOP schema string for the add method on a Foo
        # object: object and method name and an int input, with an int output.
        foo = torch.classes._TorchScriptTesting._Foo(10, 20)
        foo_ir = ir.TorchBindObject(name="foo", value=foo)
        schema = CallTorchBind.schema(foo_ir, "add")
        self.assertEqual(
            str(schema),
            "call_torchbind(__torch__.torch.classes._TorchScriptTesting._Foo _0, str method, int _1) -> int _0",
        )

    def test_torchbind_config_not_generated(self):
        # [FUNCTION DOC] Compiles a module without torchbind objects to an AOTI package and checks
        # no custom objects config json file is in the produced file list.
        # custom_objs_config.json should not be generated when its empty
        ep, inputs, _, _ = self.get_dummy_exported_model()
        aoti_files = aot_compile(
            ep.module(), inputs, options={"aot_inductor.package": True}
        )
        for file in aoti_files:
            self.assertTrue(not file.endswith("/custom_objs_config.json"))

    def test_torchbind_hop_schema_no_input(self):
        # [FUNCTION DOC] Checks the call_torchbind schema string for popping a tensor queue: no
        # extra inputs and a Tensor output.
        q = _empty_tensor_queue()
        q_ir = ir.TorchBindObject(name="q", value=q)
        schema = CallTorchBind.schema(q_ir, "pop")
        self.assertEqual(
            str(schema),
            "call_torchbind(__torch__.torch.classes._TorchScriptTesting._TensorQueue _0, str method) -> Tensor _0",
        )

    def test_torchbind_hop_schema_no_output(self):
        # [FUNCTION DOC] Checks the call_torchbind schema string for pushing onto a tensor queue: a
        # Tensor input and a NoneType output.
        q = _empty_tensor_queue()
        q_ir = ir.TorchBindObject(name="q", value=q)
        schema = CallTorchBind.schema(q_ir, "push")
        self.assertEqual(
            str(schema),
            "call_torchbind(__torch__.torch.classes._TorchScriptTesting._TensorQueue _0, str method, Tensor _1) -> NoneType _0",
        )

    @skipIfWindows(msg="AOTI is not fully support on Windows")
    def test_torchbind_aot_compile(self):
        # [FUNCTION DOC] Compiles the exported torchbind module with AOTI packaging, locates the
        # custom objects config, the custom object file and the wrapper json in the output files,
        # and checks their contents (object name mapping and the expected extern call nodes). Then
        # packages the files into a pt2 archive and checks the config and the object file appear in
        # the expected zip folders. Skipped on Windows.
        ep, inputs, _, _ = self.get_exported_model()
        aoti_files = aot_compile(
            ep.module(), inputs, options={"aot_inductor.package": True}
        )

        custom_objs_config = None
        custom_obj_0 = None
        extern_json = None
        for file in aoti_files:
            if file.endswith("/custom_objs_config.json"):
                custom_objs_config = file
            elif file.endswith("/custom_obj_0"):
                custom_obj_0 = file
            elif file.endswith("wrapper.json") and "metadata" not in file:
                extern_json = file

        self.assertIsNotNone(custom_objs_config)
        self.assertIsNotNone(custom_obj_0)
        self.assertIsNotNone(extern_json)

        with open(custom_objs_config) as file:
            data = json.load(file)
            self.assertEqual(data, {"_torchbind_obj0": "custom_obj_0"})

        with open(extern_json) as file:
            data = json.load(file)
            self.assertEqual(
                data,
                {
                    "nodes": [
                        {
                            "name": "buf1",
                            "node": {
                                "target": "_TorchScriptTesting::takes_foo_tuple_return",
                                "inputs": [
                                    {
                                        "name": "foo",
                                        "arg": {
                                            "as_custom_obj": {
                                                "name": "_torchbind_obj0",
                                                "class_fqn": "__torch__.torch.classes._TorchScriptTesting._Foo",
                                            }
                                        },
                                        "kind": 1,
                                    },
                                    {
                                        "name": "x",
                                        "arg": {"as_tensor": {"name": "buf0"}},
                                        "kind": 1,
                                    },
                                ],
                                "outputs": [
                                    {"as_tensor": {"name": "buf2"}},
                                    {"as_tensor": {"name": "buf3"}},
                                ],
                                "metadata": {},
                                "is_hop_single_tensor_return": None,
                                "name": None,
                            },
                        },
                        {
                            "name": "buf5",
                            "node": {
                                "target": "_TorchScriptTesting::takes_foo",
                                "inputs": [
                                    {
                                        "name": "foo",
                                        "arg": {
                                            "as_custom_obj": {
                                                "name": "_torchbind_obj0",
                                                "class_fqn": "__torch__.torch.classes._TorchScriptTesting._Foo",
                                            }
                                        },
                                        "kind": 1,
                                    },
                                    {
                                        "name": "x",
                                        "arg": {"as_tensor": {"name": "buf4"}},
                                        "kind": 1,
                                    },
                                ],
                                "outputs": [{"as_tensor": {"name": "buf5"}}],
                                "metadata": {},
                                "is_hop_single_tensor_return": None,
                                "name": None,
                            },
                        },
                        {
                            "name": "buf6",
                            "node": {
                                "target": "call_torchbind",
                                "inputs": [
                                    {
                                        "name": "_0",
                                        "arg": {
                                            "as_custom_obj": {
                                                "name": "_torchbind_obj0",
                                                "class_fqn": "__torch__.torch.classes._TorchScriptTesting._Foo",
                                            }
                                        },
                                        "kind": 1,
                                    },
                                    {
                                        "name": "method",
                                        "arg": {"as_string": "add_tensor"},
                                        "kind": 1,
                                    },
                                    {
                                        "name": "_1",
                                        "arg": {"as_tensor": {"name": "buf0"}},
                                        "kind": 1,
                                    },
                                ],
                                "outputs": [{"as_tensor": {"name": "buf6"}}],
                                "metadata": {},
                                "is_hop_single_tensor_return": None,
                                "name": None,
                            },
                        },
                    ]
                },
            )

        # Test that the files are packaged
        with WritableTempFile(suffix=".pt2") as f:
            package_path = package_aoti(f.name, aoti_files)

            with zipfile.ZipFile(package_path, "r") as zip_ref:
                all_files = zip_ref.namelist()
                base_folder = all_files[0].split("/")[0]
                tmp_path_model = Path(base_folder) / "data" / "aotinductor" / "model"
                tmp_path_constants = Path(base_folder) / "data" / "constants"

                self.assertTrue(
                    str(tmp_path_model / "custom_objs_config.json") in all_files
                )
                self.assertTrue(str(tmp_path_constants / "custom_obj_0") in all_files)

    def test_torchbind_aoti(self):
        # [FUNCTION DOC] Compiles the exported torchbind module to a pt2 package, loads it, runs it
        # and checks the result equals eager.
        ep, inputs, orig_res, _ = self.get_exported_model()
        pt2_path = torch._inductor.aoti_compile_and_package(ep)
        optimized = torch._inductor.aoti_load_package(pt2_path)
        result = optimized(*inputs)
        self.assertEqual(result, orig_res)

    @torch._inductor.config.patch("aot_inductor.use_runtime_constant_folding", True)
    @skipIfWindows(msg="AOTI is not fully support on Windows")
    def test_torchbind_aot_compile_constant_folding(self):
        # [FUNCTION DOC] Same as the AOTI package test but with runtime constant folding enabled in
        # the AOTI config; skipped on Windows.
        ep, inputs, orig_res, _ = self.get_exported_model()
        pt2_path = torch._inductor.aoti_compile_and_package(ep)
        optimized = torch._inductor.aoti_load_package(pt2_path)
        result = optimized(*inputs)
        self.assertEqual(result, orig_res)

    def test_torchbind_list_return_aot_compile(self):
        # [FUNCTION DOC] Exports a module calling a list-returning custom op on a Foo object and a
        # second op, using torchbind tracing; packages and loads it with AOTI and checks the result
        # equals eager.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a Foo object that uses a list-returning custom op.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the Foo torchbind object.
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Sums the three list elements from the custom op, passes
                # the sum to another op, and adds the result to the input.
                a = torch.ops._TorchScriptTesting.takes_foo_list_return(self.attr, x)
                y = a[0] + a[1] + a[2]
                b = torch.ops._TorchScriptTesting.takes_foo(self.attr, y)
                return x + b

        m = M()
        inputs = (torch.ones(2, 3),)
        orig_res = m(*inputs)

        # We can't directly torch.compile because dynamo doesn't trace ScriptObjects yet
        with enable_torchbind_tracing():
            ep = torch.export.export(m, inputs, strict=False)

        pt2_path = torch._inductor.aoti_compile_and_package(ep)
        optimized = torch._inductor.aoti_load_package(pt2_path)
        result = optimized(*inputs)
        self.assertEqual(result, orig_res)

    def test_torchbind_queue(self):
        # [FUNCTION DOC] Builds a module around a tensor queue torchbind object, running forward
        # with eager on one queue and exporting a module with a second queue; packages and loads it
        # with AOTI and checks results equal eager.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that pushes cos and sin of the input onto a tensor queue,
            # pops both and returns them as sin first then cos.
            def __init__(self, tq) -> None:
                # [FUNCTION DOC] HELPER: Stores the given tensor queue.
                super().__init__()
                self.tq = tq

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Pushes the cosine then the sine of the input onto the
                # queue, pops two tensors, and returns them in the order popped second then first.
                self.tq.push(x.cos())
                self.tq.push(x.sin())
                # TODO: int return type in fallback kernel not support yet
                x_cos = self.tq.pop()  # + self.tq.size()
                x_sin = self.tq.pop()  # - self.tq.size()
                return x_sin, x_cos

        inputs = (torch.randn(3, 2),)

        q = _empty_tensor_queue()
        m = Foo(q)
        orig_res = m(*inputs)

        q2 = _empty_tensor_queue()
        m2 = Foo(q2)

        # We can't directly torch.compile because dynamo doesn't trace ScriptObjects yet
        with enable_torchbind_tracing():
            ep = torch.export.export(m2, inputs, strict=False)

        pt2_path = torch._inductor.aoti_compile_and_package(ep)
        optimized = torch._inductor.aoti_load_package(pt2_path)
        result = optimized(*inputs)
        self.assertEqual(result, orig_res)

    @requires_gpu()
    @torch._dynamo.config.patch("capture_dynamic_output_shape_ops", True)
    @torch._inductor.config.patch("graph_partition", True)
    def test_torchbind_compile_gpu_op_symint_graph_partition(self):
        # [FUNCTION DOC] Compiles with inductor, with dynamic output shape capture and graph
        # partition enabled, a module whose tensor from a torchbind op is moved to the GPU and
        # incremented, and checks the result is close to eager. Requires a GPU.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a Foo object that moves a custom op result to the
            # GPU and adds one.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the Foo torchbind object.
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(2, 3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the tensor-returning op, moves the result to the GPU
                # device and returns it plus one.
                a = torch.ops._TorchScriptTesting.takes_foo_tensor_return(self.attr, x)
                a_cuda = a.to(device=GPU_TYPE)
                return a_cuda + 1

        m = M()
        inputs = (torch.ones(2, 3),)
        orig_res = m(*inputs)
        new_res = torch.compile(m, backend="inductor")(*inputs)
        self.assertTrue(torch.allclose(orig_res, new_res))

    def test_torchbind_input_aot_compile(self):
        # [FUNCTION DOC] Exports a module taking a Foo object as an input and checks AOTI
        # compilation raises a UserError saying TorchBind object inputs are not supported in
        # AOTInductor.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward takes a Foo object and a tensor.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the parent initializer.
                super().__init__()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Calls the list-returning custom op on the Foo input and
                # tensor and returns its result.
                a = torch.ops._TorchScriptTesting.takes_foo_list_return(x, y)
                return a

        m = M()
        inputs = (torch.classes._TorchScriptTesting._Foo(10, 20), torch.ones(2, 3))

        # We can't directly torch.compile because dynamo doesn't trace ScriptObjects yet
        with enable_torchbind_tracing():
            ep = torch.export.export(m, inputs, strict=False)

        from torch._dynamo.exc import UserError

        with self.assertRaisesRegex(
            UserError,
            expected_regex="TorchBind object inputs are not supported in AOTInductor",
        ):
            aot_compile(ep.module(), inputs, options={"aot_inductor.package": True})

    def test_aoti_torchbind_name_collision(self):
        # [FUNCTION DOC] Exports a module holding a Foo attribute named like the generated
        # torchbind object plus a second Foo created inside forward, packages and loads with AOTI,
        # and checks the result equals eager, i.e. names do not collide.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an attribute named like the generated object names
            # and a second Foo created inside forward.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the Foo object attribute.
                super().__init__()
                self._torchbind_obj0 = torch.classes._TorchScriptTesting._Foo(2, 3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls add_tensor on the attribute and on a newly created
                # Foo and returns their sum.
                a = self._torchbind_obj0.add_tensor(x)
                torchbind = torch.classes._TorchScriptTesting._Foo(4, 5)
                b = torchbind.add_tensor(x)
                return a + b

        m = M()
        inputs = (torch.ones(2, 3),)
        orig_res = m(*inputs)

        with enable_torchbind_tracing():
            ep = torch.export.export(m, inputs, strict=False)

        pt2_path = torch._inductor.aoti_compile_and_package(ep)
        optimized = torch._inductor.aoti_load_package(pt2_path)
        result = optimized(*inputs)
        self.assertEqual(result, orig_res)


if __name__ == "__main__":
    run_tests()
