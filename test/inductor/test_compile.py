# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for using TorchInductor directly (without Dynamo) and for the C++ build and
# CUDA architecture option logic in torch._inductor.cpp_builder and codegen.cuda.compile_utils.
# Defines a subprocess helper, three small modules and one TestCase. Module-level flags record
# whether the platform is macOS or Windows. The main block runs the tests only when a CPU test
# environment is available.
import os
import shlex
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch
from torch import _dynamo as dynamo, _inductor as inductor
from torch._inductor import config, cpp_builder
from torch._inductor.codecache import _cuda_fatbin_command, write
from torch._inductor.cpp_builder import (
    BuildOptionsBase,
    CppBuilder,
    CppOptions,
    CppTorchOptions,
)
from torch._inductor.cpu_vec_isa import invalid_vec_isa
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import gen_gm_and_inputs
from torch.fx import symbolic_trace
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.inductor_utils import HAS_CPU


_IS_MACOS = sys.platform.startswith("darwin")
_IS_WINDOWS = sys.platform == "win32"


def safe_command_output(cmd, timeout=30):
    # [FUNCTION DOC] HELPER: Runs a command (shell mode if given a string) capturing merged stdout
    # and stderr, and returns the stripped output text. A failing command yields a message string
    # with the exit code and output instead of raising; a timeout (default 30 seconds) yields a
    # short timeout message string.
    try:
        return subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            shell=isinstance(cmd, str),
        ).strip()
    except subprocess.CalledProcessError as e:
        return f"run failed（error code {e.returncode}）: {e.output.strip()}"
    except subprocess.TimeoutExpired:
        return "runt timeout"


class MyModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Small test network with two 10x10 linear layers, a ReLU and a sigmoid,
    # used as the compile target.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates linear layers a and b (10 to 10 each) and a ReLU
        # activation.
        super().__init__()
        self.a = torch.nn.Linear(10, 10)
        self.b = torch.nn.Linear(10, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies layer a then ReLU, then layer b then sigmoid, and returns
        # the tensor.
        x = self.relu(self.a(x))
        x = torch.sigmoid(self.b(x))
        return x


class MyModule2(MyModule):
    # [CLASS DOC] HELPER: Variant of MyModule whose input is a dict holding a pair of tensors;
    # inherits the layers.
    def forward(self, x):  # takes a dict of list
        # [FUNCTION DOC] HELPER: Unpacks the pair stored under the key entry, runs the base
        # network on the first element, adds the second element, and returns a dict with the sum
        # under the result entry.
        a, b = x["key"]
        return {"result": super().forward(a) + b}


class MyModule3(MyModule):
    # [CLASS DOC] HELPER: Variant of MyModule whose output is wrapped in a one-element tuple
    # (needed when compiling a bare module).
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Runs the base network and returns the result inside a tuple.
        return (super().forward(x),)


class TestStandaloneInductor(TestCase):
    # [CLASS DOC] Tests that inductor.compile works standalone on FX graphs, bare modules and
    # exported graphs, plus build-system tests (debug flags, architecture flags, CUDA gencode
    # options, compiler detection) that mock subprocess output and config. Stateless apart from
    # helpers that clear compiler lookup caches.
    """
    These test check that you can call TorchInductor directly without
    going through TorchDynamo.
    """

    def _clear_cpp_builder_compiler_caches(self):
        # [FUNCTION DOC] HELPER: Clears the memoization caches of the cpp_builder compiler probing
        # functions (version string, apple clang, clang, gcc, intel and version info) so mocked
        # subprocess output is not shadowed by earlier results.
        for func in (
            cpp_builder._compiler_version_string,
            cpp_builder._is_apple_clang,
            cpp_builder._is_clang,
            cpp_builder._is_gcc,
            cpp_builder._is_intel_compiler,
            cpp_builder.get_compiler_version_info,
        ):
            func.cache_clear()

    def test_inductor_via_fx(self):
        # [FUNCTION DOC] Compiles a symbolically traced module (tuple output) with
        # inductor.compile and checks the compiled result equals eager.
        mod = MyModule3().eval()
        inp = torch.randn(10)
        correct = mod(inp)
        mod_opt = inductor.compile(symbolic_trace(mod), [inp])
        actual = mod_opt(inp)
        self.assertEqual(actual, correct)

    def test_inductor_via_fx_tensor_return(self):
        # [FUNCTION DOC] Same as the previous FX test but for a module returning a single tensor.
        mod = MyModule().eval()
        inp = torch.randn(10)
        correct = mod(inp)
        mod_opt = inductor.compile(symbolic_trace(mod), [inp])
        actual = mod_opt(inp)
        self.assertEqual(actual, correct)

    @unittest.skipUnless(torch.cuda.is_available(), "requires cuda")
    @config.patch(cpp_wrapper=True)
    def test_compiled_conv_rejects_mixed_input_weight_devices(self):
        # [FUNCTION DOC] Requires CUDA and enables the C++ wrapper. Compiles a conv with the
        # inductor backend, then moves only the module to CUDA while the input stays on CPU;
        # calling it must raise a RuntimeError saying all tensors are expected to be on the same
        # device.
        mod = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1).eval()
        inp = torch.randn(1, 3, 8, 8)
        mod_opt = torch.compile(mod, backend="inductor")

        mod.to("cuda")
        with self.assertRaisesRegex(
            RuntimeError, "Expected all tensors to be on the same device"
        ):
            mod_opt(inp)

    def test_inductor_via_fx_dict_input(self):
        # [FUNCTION DOC] Compiles a traced module that takes a dict containing a list of two
        # tensors, and checks the result equals eager.
        mod = MyModule2().eval()
        inp = {"key": [torch.randn(10), torch.randn(10)]}
        correct = mod(inp)
        mod_opt = inductor.compile(symbolic_trace(mod), [inp])
        actual = mod_opt(inp)
        self.assertEqual(actual, correct)

    def test_inductor_via_make_fx(self):
        # [FUNCTION DOC] Compiles a graph produced by make_fx tracing and checks the result equals
        # eager.
        mod = MyModule().eval()
        inp = torch.randn(10)
        correct = mod(inp)
        mod_opt = inductor.compile(make_fx(mod)(inp), [inp])
        actual = mod_opt(inp)
        self.assertEqual(actual, correct)

    def test_inductor_via_bare_module(self):
        # [FUNCTION DOC] Compiles the module directly with no FX graph (works since it returns a
        # tuple) and checks equality with eager.
        mod = MyModule3().eval()
        inp = torch.randn(10)
        correct = mod(inp)
        # no FX graph at all (mod must return list/tuple in this case)
        mod_opt = inductor.compile(mod, [inp])
        actual = mod_opt(inp)
        self.assertEqual(actual, correct)

    def test_inductor_via_export1(self):
        # [FUNCTION DOC] Exports the module with Dynamo into an aten-level graph, compiles it with
        # inductor.compile, and checks equality with eager.
        mod = MyModule3().eval()
        inp = torch.randn(10)
        correct = mod(inp)
        gm, _ = dynamo.export(mod, inp, aten_graph=True)
        mod_opt = inductor.compile(gm, [inp])
        actual = mod_opt(inp)
        self.assertEqual(actual, correct)

    def test_inductor_via_export2(self):
        # [FUNCTION DOC] Exports the dict-input module with Dynamo (not aten level), compiles the
        # graph and checks equality with eager.
        mod = MyModule2().eval()
        inp = {"key": [torch.randn(10), torch.randn(10)]}
        correct = mod(inp)
        gm, _ = dynamo.export(mod, inp)
        mod_opt = inductor.compile(gm, [inp])
        actual = mod_opt(inp)
        self.assertEqual(actual, correct)

    def test_inductor_via_op_with_multiple_outputs(self):
        # [FUNCTION DOC] Builds a graph and inputs for the native layer norm aten op (which
        # returns several outputs) using gen_gm_and_inputs, compiles it, and checks the compiled
        # outputs equal the uncompiled ones.
        x1 = torch.randn((2, 512, 128))
        x2 = [128]
        x3 = torch.randn(128)
        x4 = torch.randn((128,))
        x5 = 1e-6
        mod, inp = gen_gm_and_inputs(
            torch.ops.aten.native_layer_norm.default, (x1, x2, x3, x4, x5), {}
        )
        mod_opt = inductor.compile(mod, inp)
        self.assertEqual(mod(*inp), mod_opt(*inp))

    @mock.patch.dict(os.environ, {"TORCHINDUCTOR_DEBUG_COMPILE": "1"})
    def test_inductor_generate_debug_compile(self):
        # [FUNCTION DOC] With the debug-compile environment variable set to 1 (patched for the
        # test), writes a trivial C++ main program to a source file, builds it with CppBuilder and
        # default options, and checks the binary has debug information: on Windows a pdb file next
        # to it, on Linux a debug_info section found via readelf, on macOS nothing is checked.
        cpp_code = """
        int main(){
            return 0;
        }
        """

        _, source_path = write(
            cpp_code,
            "cpp",
        )
        build_option = CppOptions()
        cpp_builder = CppBuilder(
            name="test_compile",
            sources=source_path,
            output_dir=os.path.dirname(source_path),
            BuildOption=build_option,
        )
        cpp_builder.build()
        binary_path = cpp_builder.get_target_file_path()

        """
        When we turn on generate debug compile.
        On Windows, it should create a [module_name].pdb file. It helps debug by WinDBG.
        On Linux, it should create some debug sections in binary file.
        """

        def check_linux_debug_section(module_path: str):
            # [FUNCTION DOC] HELPER: Runs readelf section listing on the module and asserts the
            # output contains a debug_info section.
            check_cmd = shlex.split(f"readelf -S {module_path}")
            output = safe_command_output(check_cmd)
            has_debug_sym = ".debug_info" in output
            self.assertEqual(has_debug_sym, True)

        def check_windows_pdb_exist(module_path: str):
            # [FUNCTION DOC] HELPER: Asserts that a pdb file with the module's base name exists
            # next to the module path.
            file_name_no_ext = os.path.splitext(module_path)[0]
            file_name_pdb = f"{file_name_no_ext}.pdb"
            has_pdb_file = os.path.exists(file_name_pdb)
            self.assertEqual(has_pdb_file, True)

        if _IS_WINDOWS:
            check_windows_pdb_exist(binary_path)
        elif _IS_MACOS:
            pass  # MacOS not sure that if it should be works.
        else:
            check_linux_debug_section(binary_path)

    @mock.patch.dict(os.environ, {"TORCHINDUCTOR_DEBUG_SYMBOL": "1"})
    def test_inductor_generate_debug_symbol(self):
        # [FUNCTION DOC] Same as the debug-compile test but with the debug-symbol environment
        # variable set to 1: builds a trivial C++ program and asserts debug information is present
        # (pdb on Windows, debug_info section on Linux, no check on macOS).
        cpp_code = """
        int main(){
            return 0;
        }
        """

        _, source_path = write(
            cpp_code,
            "cpp",
        )
        build_option = CppOptions()
        cpp_builder = CppBuilder(
            name="test_symbol",
            sources=source_path,
            output_dir=os.path.dirname(source_path),
            BuildOption=build_option,
        )
        cpp_builder.build()
        binary_path = cpp_builder.get_target_file_path()

        """
        When we turn on generate debug symbol.
        On Windows, it should create a [module_name].pdb file. It helps debug by WinDBG.
        On Linux, it should create some debug sections in binary file.
        """

        def check_linux_debug_section(module_path: str):
            # [FUNCTION DOC] HELPER: Runs readelf section listing on the module and asserts the
            # output contains a debug_info section.
            check_cmd = shlex.split(f"readelf -S {module_path}")
            output = safe_command_output(check_cmd)
            has_debug_sym = ".debug_info" in output
            self.assertEqual(has_debug_sym, True)

        def check_windows_pdb_exist(module_path: str):
            # [FUNCTION DOC] HELPER: Asserts that a pdb file with the module's base name exists
            # next to the module path.
            file_name_no_ext = os.path.splitext(module_path)[0]
            file_name_pdb = f"{file_name_no_ext}.pdb"
            has_pdb_file = os.path.exists(file_name_pdb)
            self.assertEqual(has_pdb_file, True)

        if _IS_WINDOWS:
            check_windows_pdb_exist(binary_path)
        elif _IS_MACOS:
            pass  # MacOS not sure that if it should be works.
        else:
            check_linux_debug_section(binary_path)

    def _aot_cpp_arch_flags(self):
        # [FUNCTION DOC] HELPER: Builds AOT-mode compile-only torch C++ options with an invalid
        # vector ISA and returns the list of compiler flags starting with the march or mcpu
        # prefix.
        build_option = CppTorchOptions(
            aot_mode=True,
            compile_only=True,
            vec_isa=invalid_vec_isa,
        )
        return [
            flag
            for flag in build_option.get_cflags()
            if flag.startswith(("march=", "mcpu="))
        ]

    @unittest.skipIf(config.is_fbcode(), "fbcode does not emit CPU architecture flags")
    def test_aot_cpp_march_config(self):
        # [FUNCTION DOC] Skipped in fbcode. With the cpp march config set to x86-64 and the
        # machine reported as x86_64, the architecture flags must be exactly the single march flag
        # for x86-64.
        with (
            config.patch({"cpp.march": "x86-64"}),
            mock.patch(
                "torch._inductor.cpp_builder.platform.machine",
                return_value="x86_64",
            ),
        ):
            arch_flags = self._aot_cpp_arch_flags()
        self.assertEqual(arch_flags, ["march=x86-64"])

    @unittest.skipIf(config.is_fbcode(), "fbcode does not emit CPU architecture flags")
    def test_aot_cpp_march_config_ppc64le(self):
        # [FUNCTION DOC] Skipped in fbcode. With the march config set to power9 and the machine
        # reported as ppc64le, the only architecture flag must be the mcpu flag for power9.
        with (
            config.patch({"cpp.march": "power9"}),
            mock.patch(
                "torch._inductor.cpp_builder.platform.machine",
                return_value="ppc64le",
            ),
        ):
            arch_flags = self._aot_cpp_arch_flags()
        self.assertEqual(arch_flags, ["mcpu=power9"])

    @unittest.skipIf(config.is_fbcode(), "fbcode does not emit CPU architecture flags")
    def test_cpp_march_config_can_disable_arch_flag(self):
        # [FUNCTION DOC] Skipped in fbcode. An empty march config must produce no architecture
        # flags.
        with config.patch({"cpp.march": ""}):
            arch_flags = self._aot_cpp_arch_flags()
        self.assertEqual(arch_flags, [])

    @mock.patch.dict(
        os.environ,
        {"TORCH_CUDA_ARCH_LIST": "7.0;8.0;8.6;9.0+PTX"},
    )
    @unittest.skipIf(torch.version.hip is not None, "CUDA-only")
    def test_aoti_cuda_multi_arch_gencode_options(self):
        # [FUNCTION DOC] CUDA only. With the arch list environment variable listing several
        # architectures, the multi-arch gencode options for target 80 must contain sm and compute
        # entries for 80 plus sm entries for 86 and 90, and a warning log must report that the
        # sm_70 entry is ignored (older than the target).
        from torch._inductor.codegen.cuda import compile_utils

        with self.assertLogs(
            "torch._inductor.codegen.cuda.compile_utils", level="WARNING"
        ) as log_ctx:
            self.assertEqual(
                compile_utils._cuda_multi_arch_gencode_options("80"),
                [
                    "arch=compute_80,code=sm_80",
                    "arch=compute_80,code=compute_80",
                    "arch=compute_86,code=sm_86",
                    "arch=compute_90,code=sm_90",
                ],
            )
        self.assertIn("Ignoring TORCH_CUDA_ARCH_LIST entry sm_70", log_ctx.output[0])

    @mock.patch.dict(
        os.environ,
        {"TORCH_CUDA_ARCH_LIST": "9.0;9.0a;10.0"},
    )
    @unittest.skipIf(torch.version.hip is not None, "CUDA-only")
    def test_aoti_cuda_multi_arch_gencode_options_suffix_arch(self):
        # [FUNCTION DOC] CUDA only. With an arch list including a suffixed 9.0a, the gencode
        # options for target 90a must be exactly the sm and compute entries for 90a.
        from torch._inductor.codegen.cuda import compile_utils

        self.assertEqual(
            compile_utils._cuda_multi_arch_gencode_options("90a"),
            [
                "arch=compute_90a,code=sm_90a",
                "arch=compute_90a,code=compute_90a",
            ],
        )

    @mock.patch(
        "torch._inductor.codegen.cuda.compile_utils._nvcc_arch_as_compile_option",
        return_value="100a",
    )
    @unittest.skipIf(torch.version.hip is not None, "CUDA-only")
    def test_aoti_cuda_target_arch_preserves_suffix(self, _):
        # [FUNCTION DOC] CUDA only. The nvcc architecture lookup is mocked to return 100a. Checks
        # the AOTI target arch returns 100a by default, and that a configured cuda arch of 90 or
        # 90a both yield 90a. The extra mock argument is unused.
        from torch._inductor.codegen.cuda import compile_utils

        self.assertEqual(compile_utils._aoti_cuda_target_arch(), "100a")
        with config.patch({"cuda.arch": "90"}):
            self.assertEqual(compile_utils._aoti_cuda_target_arch(), "90a")
        with config.patch({"cuda.arch": "90a"}):
            self.assertEqual(compile_utils._aoti_cuda_target_arch(), "90a")

    @mock.patch.dict(os.environ, {"TORCH_CUDA_ARCH_LIST": "8.0;8.6"})
    @mock.patch(
        "torch._inductor.codegen.cuda.compile_utils._nvcc_arch_as_compile_option",
        return_value="100a",
    )
    @unittest.skipIf(torch.version.hip is not None, "CUDA-only")
    def test_aoti_cuda_fatbin_command_uses_nvcc_for_extra_archs(self, _):
        # [FUNCTION DOC] CUDA only. With the arch list 8.0 and 8.6 and the mocked nvcc arch 100a,
        # builds the fatbin command for target 80 and checks it begins with the nvcc fatbin
        # invocation for the given files, contains gencode entries for 80 and 86, and omits the
        # 100a entry. A temporary directory holds an empty cubin file. The mock argument is
        # unused.
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_cubin = os.path.join(tmp_dir, "kernel.cubin")
            with open(raw_cubin, "wb"):
                pass
            cmd = _cuda_fatbin_command(
                "kernel.ptx",
                "kernel.fatbin",
                raw_cubin,
                "nvcc",
                "fatbinary",
                "80",
            )

        self.assertEqual(
            cmd[:5], ["nvcc", "-fatbin", "kernel.ptx", "-o", "kernel.fatbin"]
        )
        self.assertIn("arch=compute_80,code=sm_80", cmd)
        self.assertIn("arch=compute_86,code=sm_86", cmd)
        self.assertNotIn("arch=compute_100a,code=sm_100a", cmd)

    @mock.patch.dict(os.environ, {"TORCH_CUDA_ARCH_LIST": "7.0;8.0;8.6;9.0"})
    @mock.patch(
        "torch._inductor.codegen.cuda.compile_utils._nvcc_arch_as_compile_option",
        return_value="100a",
    )
    @unittest.skipIf(torch.version.hip is not None, "CUDA-only")
    def test_aoti_cuda_cmake_uses_multi_arch_gencode_flags(self, _):
        # [FUNCTION DOC] CUDA only. With arch list 7.0 to 9.0 and mocked nvcc arch 100a, saves the
        # compile command of a CppBuilder to a CMake file for cuda with the cuda arch config 80,
        # then checks the file lacks compute_70 and 100a and contains gencode flags for 80, 86 and
        # 90. The mock argument is unused.
        build_option = BuildOptionsBase(compiler="c++")
        with tempfile.TemporaryDirectory() as tmp_dir:
            cmake_path = os.path.join(tmp_dir, "CMakeLists.txt")
            cpp_builder = CppBuilder(
                name="test_compile",
                sources=[],
                output_dir=tmp_dir,
                BuildOption=build_option,
            )
            with config.patch({"cuda.arch": "80"}):
                cpp_builder.save_compile_cmd_to_cmake(cmake_path, "cuda")
            with open(cmake_path) as f:
                cmake_contents = f.read()

        self.assertNotIn("compute_70", cmake_contents)
        self.assertNotIn("compute_100a", cmake_contents)
        self.assertIn("-gencode arch=compute_80,code=sm_80", cmake_contents)
        self.assertIn("-gencode arch=compute_86,code=sm_86", cmake_contents)
        self.assertIn("-gencode arch=compute_90,code=sm_90", cmake_contents)

    @unittest.skipIf(torch.version.hip is not None, "CUDA-only")
    def test_aoti_cuda_save_kernel_recompiles_for_target_arch(self):
        # [FUNCTION DOC] CUDA only. Builds a bare CachingAutotuner with fake device properties
        # (compute capability 100) and a fake launcher holding a current binary, plus a fake
        # compile result holding a target binary. With multi-arch emission on and cuda arch 90a,
        # save_gpu_kernel must recompile for capability 90 and store the target cubin and ptx with
        # cuda_arch 90a in the kernel param cache. With cuda arch unset and the mocked nvcc arch
        # 100a, it must not recompile and must store the current binary's cubin and ptx with
        # cuda_arch 100a.
        from torch._inductor.runtime.triton_heuristics import (
            CachingAutotuner,
            TritonCompileResult,
        )

        autotuner = object.__new__(CachingAutotuner)
        autotuner.inductor_meta = {"kernel_name": "triton_kernel"}
        autotuner.triton_meta = {}
        autotuner.device_props = types.SimpleNamespace(type="cuda", cc=100)

        current_binary = types.SimpleNamespace(
            metadata=types.SimpleNamespace(name="kernel", num_warps=1, shared=0),
            asm={"cubin": b"current cubin", "ptx": "current ptx"},
        )
        target_binary = types.SimpleNamespace(
            metadata=types.SimpleNamespace(name="kernel", num_warps=1, shared=0),
            asm={"cubin": b"target cubin", "ptx": "target ptx"},
        )
        target_result = object.__new__(TritonCompileResult)
        target_result.kernel = target_binary

        launcher = types.SimpleNamespace(
            bin=current_binary,
            config=types.SimpleNamespace(kwargs={}, num_warps=1, num_stages=1),
            def_args=[],
            call_args=[],
            global_scratch=None,
            profile_scratch=None,
        )

        with (
            config.patch(
                {
                    "aot_inductor.emit_multi_arch_kernel": True,
                    "cuda.arch": "90a",
                }
            ),
            mock.patch.object(
                CachingAutotuner,
                "_precompile_config",
                return_value=target_result,
            ) as precompile_config,
            mock.patch(
                "torch._inductor.codecache.CudaKernelParamCache.set"
            ) as cache_set,
        ):
            autotuner.save_gpu_kernel("stream", launcher)

        precompile_config.assert_called_once_with(launcher.config, cc_override=90)
        _, params, cubin, bin_type, asm, asm_type = cache_set.call_args.args
        self.assertEqual(params["cuda_arch"], "90a")
        self.assertEqual(cubin, b"target cubin")
        self.assertEqual(bin_type, "cubin")
        self.assertEqual(asm, "target ptx")
        self.assertEqual(asm_type, "ptx")

        with (
            config.patch(
                {
                    "aot_inductor.emit_multi_arch_kernel": True,
                    "cuda.arch": None,
                }
            ),
            mock.patch(
                "torch._inductor.codegen.cuda.compile_utils._nvcc_arch_as_compile_option",
                return_value="100a",
            ),
            mock.patch.object(
                CachingAutotuner,
                "_precompile_config",
                return_value=target_result,
            ) as precompile_config,
            mock.patch(
                "torch._inductor.codecache.CudaKernelParamCache.set"
            ) as cache_set,
        ):
            autotuner.save_gpu_kernel("stream", launcher)

        precompile_config.assert_not_called()
        _, params, cubin, bin_type, asm, asm_type = cache_set.call_args.args
        self.assertEqual(params["cuda_arch"], "100a")
        self.assertEqual(cubin, b"current cubin")
        self.assertEqual(bin_type, "cubin")
        self.assertEqual(asm, "current ptx")
        self.assertEqual(asm_type, "ptx")

    @unittest.skipIf(_IS_WINDOWS, "compiler command splitting is for POSIX shells")
    def test_cpp_compiler_search_splits_compiler_command(self):
        # [FUNCTION DOC] Skipped on Windows. Checks that the compiler search splits a multi-word
        # compiler string into separate arguments for the version probe, and returns the original
        # string. Caches are cleared before and after.
        cpp_builder.cpp_compiler_search.cache_clear()

        def fake_check_output(cmd, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Fake subprocess output function: asserts the command is split
            # into zig, c++ and the version flag, and returns bytes that look like a clang version
            # banner.
            self.assertEqual(cmd, ["zig", "c++", "--version"])
            return b"clang version fake-zig"

        try:
            with mock.patch("subprocess.check_output", side_effect=fake_check_output):
                self.assertEqual(
                    cpp_builder.cpp_compiler_search(("zig c++",)), "zig c++"
                )
        finally:
            cpp_builder.cpp_compiler_search.cache_clear()

    @unittest.skipIf(_IS_WINDOWS, "compiler command splitting is for POSIX shells")
    def test_compiler_version_info_splits_compiler_command(self):
        # [FUNCTION DOC] Skipped on Windows. With subprocess output mocked, the compiler version
        # info for a multi-word compiler must be the banner lines joined with underscores, and the
        # probed command must be the split words plus the verbose flag. Caches are cleared before
        # and after.
        self._clear_cpp_builder_compiler_caches()

        try:
            with mock.patch(
                "subprocess.check_output",
                return_value=b"clang version fake-zig\nTarget: fake\n",
            ) as check_output_mock:
                self.assertEqual(
                    cpp_builder.get_compiler_version_info("zig c++"),
                    "clang version fake-zig_Target: fake_",
                )
            self.assertEqual(
                check_output_mock.call_args[0][0],
                ["zig", "c++", "-v"],
            )
        finally:
            self._clear_cpp_builder_compiler_caches()

    @unittest.skipIf(_IS_WINDOWS, "compiler command splitting is for POSIX shells")
    def test_is_clang_uses_version_output_for_multiword_compiler(self):
        # [FUNCTION DOC] Skipped on Windows. With a fake clang banner, a multi-word compiler is
        # detected as clang and not as gcc. Caches are cleared before and after.
        self._clear_cpp_builder_compiler_caches()

        def fake_check_output(cmd, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Fake subprocess output function: asserts the command is split
            # into zig, c++ and the version flag, and returns a clang banner as bytes.
            self.assertEqual(cmd, ["zig", "c++", "--version"])
            return b"clang version fake-zig\nTarget: fake\n"

        try:
            with mock.patch("subprocess.check_output", side_effect=fake_check_output):
                self.assertTrue(cpp_builder._is_clang("zig c++"))
                self.assertFalse(cpp_builder._is_gcc("zig c++"))
        finally:
            self._clear_cpp_builder_compiler_caches()

    @unittest.skipIf(_IS_WINDOWS, "compiler command splitting is for POSIX shells")
    def test_is_clang_preserves_apple_clang_detection(self):
        # [FUNCTION DOC] Skipped on Windows. With the platform patched to darwin and an Apple LLVM
        # banner, the compiler is detected as apple clang and clang but not gcc. Caches are
        # cleared before and after.
        self._clear_cpp_builder_compiler_caches()

        try:
            with (
                mock.patch("torch._inductor.cpp_builder.sys.platform", "darwin"),
                mock.patch(
                    "subprocess.check_output",
                    return_value=b"Apple LLVM version 10.0.0 (clang-1000.0.42)\n",
                ),
            ):
                self.assertTrue(cpp_builder._is_apple_clang("gcc"))
                self.assertTrue(cpp_builder._is_clang("gcc"))
                self.assertFalse(cpp_builder._is_gcc("gcc"))
        finally:
            self._clear_cpp_builder_compiler_caches()

    @unittest.skipIf(_IS_WINDOWS, "compiler command splitting is for POSIX shells")
    def test_cpp_options_use_clang_identity_for_multiword_compiler(self):
        # [FUNCTION DOC] Skipped on Windows. With march empty, platform patched to Darwin, gcc
        # detection mocked off and a fake clang banner, the C++ options for a multi-word compiler
        # (link on, warnings off) must have a compile flag ending with the ignored-optimization-
        # argument suffix and an undefined dynamic lookup flag. Caches are cleared before and
        # after.
        self._clear_cpp_builder_compiler_caches()

        def fake_check_output(cmd, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Fake subprocess output function: asserts the command is split
            # into zig, c++ and the version flag, and returns a clang banner as bytes.
            self.assertEqual(cmd, ["zig", "c++", "--version"])
            return b"clang version fake-zig\nTarget: fake\n"

        try:
            with (
                config.patch({"cpp.march": ""}),
                mock.patch("subprocess.check_output", side_effect=fake_check_output),
                mock.patch(
                    "torch._inductor.cpp_builder.platform.system",
                    return_value="Darwin",
                ),
                mock.patch("torch._inductor.cpp_builder.sys.platform", "darwin"),
                mock.patch("torch._inductor.cpp_builder.is_gcc", return_value=False),
            ):
                cflags = cpp_builder.get_cpp_options(
                    "zig c++", do_link=True, warning_all=False
                )[2]

            self.assertTrue(
                any(flag.endswith("ignored-optimization-argument") for flag in cflags)
            )
            self.assertIn("undefined dynamic_lookup", cflags)
        finally:
            self._clear_cpp_builder_compiler_caches()

    def test_is_msvc_cl_handles_invalid_help_bytes(self):
        # [FUNCTION DOC] Mocks the Windows flag, strict decode arguments and the subprocess run
        # result to return help output containing an invalid utf-8 byte. The MSVC cl detection
        # must still return true, i.e. it must not decode the help output. The cache is cleared
        # before and after.
        help_output = (
            b"Compilateur d'optimisation Microsoft (R) C/C++ version"
            b"\xff19.35.32216.1\r\n"
        )
        completed_process = subprocess.CompletedProcess(
            ["cl", "/help"], 0, stdout=help_output
        )

        cpp_builder._is_msvc_cl.cache_clear()
        try:
            # Keep decode settings strict; this probe should not decode /help output.
            with (
                mock.patch("torch._inductor.cpp_builder._IS_WINDOWS", True),
                mock.patch(
                    "torch._inductor.cpp_builder.SUBPROCESS_DECODE_ARGS", ("utf-8",)
                ),
                mock.patch(
                    "torch._inductor.cpp_builder.subprocess.run",
                    return_value=completed_process,
                ),
            ):
                self.assertTrue(cpp_builder._is_msvc_cl("cl"))
        finally:
            cpp_builder._is_msvc_cl.cache_clear()


if __name__ == "__main__":
    if HAS_CPU:
        run_tests()
