# Owner(s): ["oncall: export"]
# flake8: noqa
# [MODULE DOC] Export round-trip tests for higher-order operators (HOPs). At import time it filters
# the HOP OpInfo database to drop operators on an allowlist of ones lacking OpInfo tests, defines
# the set of expected export failures, and instantiates device-generic tests.
import copy
import io
import unittest

import torch
import torch._dynamo as torchdynamo
import torch.utils._pytree as pytree
from torch._dynamo.test_case import TestCase
from torch.export import export, load, save
from torch.export._trace import _export
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
    skipOps,
    xfail,
)
from torch.testing._internal.common_utils import IS_WINDOWS, run_tests
from torch.testing._internal.hop_db import (
    FIXME_hop_that_doesnt_have_opinfo_test_allowlist,
    hop_db,
)


hop_tests = []

for op_info in hop_db:
    op_info_hop_name = op_info.name
    if op_info_hop_name in FIXME_hop_that_doesnt_have_opinfo_test_allowlist:
        continue
    hop_tests.append(op_info)


hop_export_failures = {
    xfail("invoke_quant", "simple"),
    xfail("flex_attention", "simple"),
    xfail("flex_attention_backward", "simple"),
    xfail("flex_attention_backward", "explicit_buffers"),
    xfail("local_map_hop", "simple"),
    xfail("register_hook", "simple"),
}


@unittest.skipIf(IS_WINDOWS, "Windows isn't supported for this case")
@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo isn't support")
class TestHOP(TestCase):
    # [CLASS DOC] Device-generic tests (skipped on Windows or when Dynamo is unsupported) that
    # export each HOP OpInfo sample in several ways and compare exported outputs to eager outputs.
    def _compare(self, eager_model, export, args, kwargs):
        # [FUNCTION DOC] HELPER: Deep-copies the arguments separately for eager and exported runs,
        # runs the eager model and the exported program module, flattens outputs and asserts each
        # pair of leaves has the same type and equal values.
        eager_args = copy.deepcopy(args)
        eager_kwargs = copy.deepcopy(kwargs)
        export_args = copy.deepcopy(args)
        export_kwargs = copy.deepcopy(kwargs)

        flat_orig_outputs = pytree.tree_leaves(eager_model(*eager_args, **eager_kwargs))
        flat_loaded_outputs = pytree.tree_leaves(
            export.module()(*export_args, **export_kwargs)
        )

        for orig, loaded in zip(flat_orig_outputs, flat_loaded_outputs):
            self.assertEqual(type(orig), type(loaded))
            self.assertEqual(orig, loaded)

    @ops(hop_tests, allowed_dtypes=(torch.float,))
    @skipOps(hop_export_failures)
    def test_aot_export(self, device, dtype, op):
        # [FUNCTION DOC] Strict-mode export of a model calling the HOP on each sample (with
        # gradients enabled), compared against eager. Afterwards resets Dynamo guarded backend
        # cache by hand to avoid leaks seen under the CUDA memory leak check.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Throwaway module wrapping the higher-order operator under test so
            # it can be exported.
            def forward(self, *args):
                # [FUNCTION DOC] HELPER: Applies the operator under test to the positional
                # arguments and returns the result.
                return op.op(*args)

        sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=True)
        for inp in sample_inputs_itr:
            model = Foo()
            input = inp.input if isinstance(inp.input, tuple) else (inp.input,)
            args = (*input, *inp.args)
            kwargs = inp.kwargs
            ep = export(model, args, kwargs, strict=True)
            self._compare(model, ep, args, kwargs)
        # With PYTORCH_TEST_CUDA_MEM_LEAK_CHECK=1, a memory leak occurs during
        # strict-mode export. We need to manually reset the cache of backends.
        # Specifically, `cached_backends.clear()` is required.
        # Upon examining the items in `cached_backends`,
        # we notice that under strict-mode export, there exists
        # the `dynamo_normalization_capturing_compiler`, which must be
        # cleared to avoid memory leaks. An educated guess is that
        # the `dynamo_normalization_capturing_compiler` references input tensors
        # on CUDA devices and fails to free them.
        torchdynamo._reset_guarded_backend_cache()

    @ops(hop_tests, allowed_dtypes=(torch.float,))
    @skipOps(hop_export_failures)
    def test_pre_dispatch_export(self, device, dtype, op):
        # [FUNCTION DOC] Pre-dispatch export of a model calling the HOP on each sample, compared
        # against eager; resets the Dynamo guarded backend cache at the end.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Throwaway module wrapping the higher-order operator under test so
            # it can be exported.
            def forward(self, *args):
                # [FUNCTION DOC] HELPER: Applies the operator under test to the positional
                # arguments and returns the result.
                return op.op(*args)

        sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=True)
        for inp in sample_inputs_itr:
            model = Foo()
            input = inp.input if isinstance(inp.input, tuple) else (inp.input,)
            args = (*input, *inp.args)
            kwargs = inp.kwargs
            ep = _export(model, args, kwargs, pre_dispatch=True)
            self._compare(model, ep, args, kwargs)
        torchdynamo._reset_guarded_backend_cache()

    @ops(hop_tests, allowed_dtypes=(torch.float,))
    @skipOps(hop_export_failures)
    def test_retrace_export(self, device, dtype, op):
        # [FUNCTION DOC] Pre-dispatch export followed by run_decompositions on each sample, then
        # compares against eager; resets the Dynamo guarded backend cache.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Throwaway module wrapping the higher-order operator under test so
            # it can be exported.
            def forward(self, *args):
                # [FUNCTION DOC] HELPER: Applies the operator under test to the positional
                # arguments and returns the result.
                return op.op(*args)

        sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=True)
        for inp in sample_inputs_itr:
            model = Foo()
            input = inp.input if isinstance(inp.input, tuple) else (inp.input,)
            args = (*input, *inp.args)
            kwargs = inp.kwargs
            ep = _export(model, args, kwargs, pre_dispatch=True)
            ep = ep.run_decompositions()
            self._compare(model, ep, args, kwargs)
        torchdynamo._reset_guarded_backend_cache()

    @ops(hop_tests, allowed_dtypes=(torch.float,))
    @skipOps(hop_export_failures | {xfail("switch", "simple")})
    def test_serialize_export(self, device, dtype, op):
        # [FUNCTION DOC] Pre-dispatch export plus decomposition, then saves the program into an
        # in-memory buffer, loads it back and compares the loaded program with eager; the switch
        # simple sample is an additional expected failure. Resets Dynamo guarded backend cache.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Throwaway module wrapping the higher-order operator under test so
            # it can be exported.
            def forward(self, *args):
                # [FUNCTION DOC] HELPER: Applies the operator under test to the positional
                # arguments and returns the result.
                return op.op(*args)

        sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=True)
        for inp in sample_inputs_itr:
            model = Foo()
            input = inp.input if isinstance(inp.input, tuple) else (inp.input,)
            args = (*input, *inp.args)
            kwargs = inp.kwargs
            ep = _export(model, args, kwargs, pre_dispatch=True)
            ep = ep.run_decompositions()
            buffer = io.BytesIO()
            save(ep, buffer)
            buffer.seek(0)
            ep = load(buffer)
            self._compare(model, ep, args, kwargs)
        torchdynamo._reset_guarded_backend_cache()


instantiate_device_type_tests(TestHOP, globals())

if __name__ == "__main__":
    run_tests()
