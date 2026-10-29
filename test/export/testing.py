# [MODULE DOC] Shared helpers for the export test suite. Defines a list of composite ATen
# overloads that tests may ask to preserve during decomposition (explicitly not exhaustive), a
# factory that clones a test class with every test method wrapped so the module level export
# function used by test_export is patched with a replacement, and a family of decorators that
# tag a test function with an attribute marking it as an expected failure for one particular
# export variant. Import-time side effect: resolves the aten op namespace and builds the op
# list, which requires those overloads to exist.
import functools
import unittest
from unittest.mock import patch

import torch


aten = torch.ops.aten

# This list is not meant to be comprehensive
_COMPOSITE_OPS_THAT_CAN_BE_PRESERVED_TESTING_ONLY = [
    aten.arctan2.default,
    aten.divide.Tensor,
    aten.divide.Scalar,
    aten.divide.Tensor_mode,
    aten.divide.Scalar_mode,
    aten.multiply.Tensor,
    aten.multiply.Scalar,
    aten.subtract.Tensor,
    aten.subtract.Scalar,
    aten.true_divide.Tensor,
    aten.true_divide.Scalar,
    aten.greater.Tensor,
    aten.greater.Scalar,
    aten.greater_equal.Tensor,
    aten.greater_equal.Scalar,
    aten.less_equal.Tensor,
    aten.less_equal.Scalar,
    aten.less.Tensor,
    aten.less.Scalar,
    aten.not_equal.Tensor,
    aten.not_equal.Scalar,
    aten.var.default,
    aten.var.dim,
    aten.std.default,
    aten.std.dim,
    aten.absolute.default,
    aten.arccos.default,
    aten.arccosh.default,
    aten.arcsin.default,
    aten.arcsinh.default,
    aten.arctan.default,
    aten.arctanh.default,
    aten.clip.default,
    aten.clip.Tensor,
    aten.fix.default,
    aten.negative.default,
    aten.square.default,
    aten.size.int,
    aten.stride.int,
    aten.repeat_interleave.self_Tensor,
    aten.repeat_interleave.self_int,
    aten.sym_size.int,
    aten.sym_stride.int,
    aten.atleast_1d.Sequence,
    aten.atleast_2d.Sequence,
    aten.atleast_3d.Sequence,
    aten.linear.default,
    aten.conv2d.default,
    aten.conv2d.padding,
    aten.mish_backward.default,
    aten.silu_backward.default,
    aten.pad_sequence.default,
    aten.upsample_nearest1d.vec,
    aten.upsample_nearest2d.vec,
    aten.upsample_nearest3d.vec,
    aten._upsample_nearest_exact1d.vec,
    aten._upsample_nearest_exact2d.vec,
    aten._upsample_nearest_exact3d.vec,
    aten.rnn_tanh.input,
    aten.rnn_tanh.data,
    aten.rnn_relu.input,
    aten.rnn_relu.data,
    aten.lstm.input,
    aten.lstm.data,
    aten.gru.input,
    aten.gru.data,
    aten._upsample_bilinear2d_aa.vec,
    aten._upsample_bicubic2d_aa.vec,
    aten.upsample_bilinear2d.vec,
    aten.upsample_trilinear3d.vec,
    aten.upsample_linear1d.vec,
    aten.matmul.default,
    aten.upsample_bicubic2d.vec,
    aten.__and__.Scalar,
    aten.__and__.Tensor,
    aten.__or__.Tensor,
    aten.__or__.Scalar,
    aten.__xor__.Tensor,
    aten.__xor__.Scalar,
    aten.is_complex.default,
    aten.where.ScalarOther,
    aten.where.ScalarSelf,
    aten.where.Scalar,
    aten.where.default,
    aten.item.default,
    aten.std_mean.default,
    aten.std_mean.dim,
    aten.var_mean.default,
    aten.var_mean.dim,
    aten.broadcast_tensors.default,
    aten.stft.default,
    aten.stft.center,
    aten.istft.default,
    aten.diag.default,
    aten.meshgrid.default,
    aten.meshgrid.indexing,
    aten.fft_fft.default,
    aten.fft_ifft.default,
    aten.fft_rfft.default,
    aten.fft_irfft.default,
    aten.fft_hfft.default,
    aten.fft_ihfft.default,
    aten.fft_fftn.default,
    aten.fft_ifftn.default,
    aten.fft_rfftn.default,
    aten.fft_ihfftn.default,
    aten.fft_irfftn.default,
    aten.fft_hfftn.default,
    aten.fft_fft2.default,
    aten.fft_ifft2.default,
    aten.fft_rfft2.default,
    aten.fft_irfft2.default,
    aten.fft_hfft2.default,
    aten.fft_ihfft2.default,
    aten.fft_fftshift.default,
    aten.fft_ifftshift.default,
    aten.selu.default,
    aten.margin_ranking_loss.default,
    aten.hinge_embedding_loss.default,
    aten.nll_loss.default,
    aten.prelu.default,
    aten.relu6.default,
    aten.pairwise_distance.default,
    aten.pdist.default,
    aten.special_ndtr.default,
    aten.max.other,
    aten.min.other,
    aten.linalg_eigvals.default,
    aten.argsort.default,
    aten.rrelu.default,
    aten.conv_transpose1d.default,
    aten.conv_transpose2d.input,
    aten.conv_transpose3d.input,
    aten.conv1d.default,
    aten.conv1d.padding,
    aten.conv3d.default,
    aten.conv3d.padding,
    aten.float_power.Tensor_Tensor,
    aten.float_power.Tensor_Scalar,
    aten.float_power.Scalar,
    aten.ldexp.Tensor,
    aten._version.default,
]


def make_test_cls_with_mocked_export(
    cls,
    cls_prefix,
    fn_suffix,
    mocked_export_fn,
    xfail_prop=None,
    test_only_if_no_xfail=False,
):
    # [FUNCTION DOC] Builds a new test class from cls by creating a type named with cls_prefix
    # followed by the original name, using the original base classes and an empty namespace,
    # and setting its qualified name to match. For each attribute of the original class whose
    # name begins with test_: non-callable ones are copied as is; callables are wrapped so
    # export is mocked, renamed with fn_suffix appended, and: if xfail_prop is set and the
    # function carries that attribute it is marked expected failure; otherwise if
    # test_only_if_no_xfail is true and the function has any attribute starting with
    # _expected_failure it is skipped with a reason message. Other attributes not already on
    # the new class are copied across (slots are not handled). Returns the new class; the
    # original is not modified.
    MockedTestClass = type(f"{cls_prefix}{cls.__name__}", cls.__bases__, {})
    MockedTestClass.__qualname__ = MockedTestClass.__name__

    for name in dir(cls):
        if name.startswith("test_"):
            fn = getattr(cls, name)
            if not callable(fn):
                setattr(MockedTestClass, name, getattr(cls, name))
                continue
            new_name = f"{name}{fn_suffix}"
            new_fn = _make_fn_with_mocked_export(fn, mocked_export_fn)
            new_fn.__name__ = new_name
            if xfail_prop is not None and hasattr(fn, xfail_prop):
                new_fn = unittest.expectedFailure(new_fn)
            elif test_only_if_no_xfail and any(
                x.startswith("_expected_failure") for x in dir(fn)
            ):
                new_fn = unittest.skip(
                    "Will only be tested if no other tests are failing"
                )(new_fn)
            setattr(MockedTestClass, new_name, new_fn)
        # NB: Doesn't handle slots correctly, but whatever
        elif not hasattr(MockedTestClass, name):
            setattr(MockedTestClass, name, getattr(cls, name))

    return MockedTestClass


def _make_fn_with_mocked_export(fn, mocked_export_fn):
    # [FUNCTION DOC] HELPER: Returns a wrapper around fn that preserves its metadata. The
    # wrapper, when called, imports test_export (relative import, falling back to a top level
    # import) and runs fn with the export name inside that module patched to mocked_export_fn
    # for the duration of the call, returning fn's result.
    @functools.wraps(fn)
    def _fn(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Wrapper body described by the enclosing helper: imports
        # test_export lazily on each call, patches its export attribute with the mock for the
        # call duration, forwards all arguments to fn and returns its result. The patch is
        # undone on exit, also when fn raises.
        try:
            from . import test_export
        except ImportError:
            import test_export  # @manual=fbcode//caffe2/test:test_export-library

        with patch(f"{test_export.__name__}.export", mocked_export_fn):
            return fn(*args, **kwargs)

    return _fn


# Controls tests generated in test/export/test_export_training_ir_to_run_decomp.py
def expectedFailureTrainingIRToRunDecomp(fn):
    # [FUNCTION DOC] Decorator that sets the marker attribute
    # _expected_failure_training_ir_to_run_decomp to true on the given test function and
    # returns the same function, so the generated training IR to run_decomp test variant is
    # treated as an expected failure. No other side effects.
    fn._expected_failure_training_ir_to_run_decomp = True
    return fn


# Controls tests generated in test/export/test_export_training_ir_to_run_decomp.py
def expectedFailureTrainingIRToRunDecompNonStrict(fn):
    # [FUNCTION DOC] Decorator that sets the marker attribute
    # _expected_failure_training_ir_to_run_decomp_non_strict to true on the given test
    # function and returns the same function, so the generated training IR to run_decomp
    # non-strict test variant is treated as an expected failure. No other side effects.
    fn._expected_failure_training_ir_to_run_decomp_non_strict = True
    return fn


# Controls tests generated in test/export/test_export_strict_v2.py
def expectedFailureStrictV2(fn):
    # [FUNCTION DOC] Decorator that sets the marker attribute _expected_failure_strict_v2 to
    # true on the given test function and returns the same function, so the generated strict
    # v2 export test variant is treated as an expected failure. No other side effects.
    fn._expected_failure_strict_v2 = True
    return fn


# Controls tests generated in test/export/test_export_strict.py
def expectedFailureStrict(fn):
    # [FUNCTION DOC] Decorator that sets the marker attribute _expected_failure_strict to true
    # on the given test function and returns the same function, so the generated strict export
    # test variant is treated as an expected failure. No other side effects.
    fn._expected_failure_strict = True
    return fn


# Controls tests generated in test/export/test_retraceability.py
def expectedFailureRetraceability(fn):
    # [FUNCTION DOC] Decorator that sets the marker attribute _expected_failure_retrace to
    # true on the given test function and returns the same function, so the generated
    # retraceability test variant is treated as an expected failure. No other side effects.
    fn._expected_failure_retrace = True
    return fn


# Controls tests generated in test/export/test_retraceability.py
def expectedFailureRetraceabilityNonStrict(fn):
    # [FUNCTION DOC] Decorator that sets the marker attribute
    # _expected_failure_retrace_non_strict to true on the given test function and returns the
    # same function, so the generated non-strict retraceability test variant is treated as an
    # expected failure. No other side effects.
    fn._expected_failure_retrace_non_strict = True
    return fn


# Controls tests generated in test/export/test_serdes.py
def expectedFailureSerDer(fn):
    # [FUNCTION DOC] Decorator that sets the marker attribute _expected_failure_serdes to true
    # on the given test function and returns the same function, so the generated
    # serialize/deserialize round trip test variant is treated as an expected failure. No
    # other side effects.
    fn._expected_failure_serdes = True
    return fn


# Controls tests generated in test/export/test_serdes.py
def expectedFailureSerDerNonStrict(fn):
    # [FUNCTION DOC] Decorator that sets the marker attribute
    # _expected_failure_serdes_non_strict to true on the given test function and returns the
    # same function, so the generated non-strict serialize/deserialize round trip test variant
    # is treated as an expected failure. No other side effects.
    fn._expected_failure_serdes_non_strict = True
    return fn


def expectedFailureSerDerPreDispatch(fn):
    # [FUNCTION DOC] Decorator that sets the marker attribute
    # _expected_failure_serdes_pre_dispatch to true on the given test function and returns the
    # same function, so the generated pre-dispatch serialize/deserialize round trip test
    # variant is treated as an expected failure. No other side effects.
    fn._expected_failure_serdes_pre_dispatch = True
    return fn


def expectedFailurePreDispatchRunDecomp(fn):
    # [FUNCTION DOC] Decorator that sets the marker attribute _expected_failure_pre_dispatch
    # to true on the given test function and returns the same function, so the generated
    # pre-dispatch run_decomp test variant is treated as an expected failure. No other side
    # effects (it returns the function unchanged otherwise).
    fn._expected_failure_pre_dispatch = True
    return fn


def expectedFailureCppSerDes(fn):
    # [FUNCTION DOC] Decorator that sets the marker attribute _expected_failure_cpp_serdes to
    # true on the given test function and returns the same function, so the generated C++
    # serialization test variant is treated as an expected failure. No other side effects.
    fn._expected_failure_cpp_serdes = True
    return fn


def expectedFailureCppRuntime(fn):
    # [FUNCTION DOC] Decorator that sets the marker attribute _expected_failure_cpp_runtime to
    # true on the given test function and returns the same function, so the generated C++
    # runtime test variant is treated as an expected failure. No other side effects.
    fn._expected_failure_cpp_runtime = True
    return fn


def expectedFailureCppRuntimeNonStrict(fn):
    # [FUNCTION DOC] Decorator that sets the marker attribute
    # _expected_failure_cpp_runtime_non_strict to true on the given test function and returns
    # the same function, so the generated non-strict C++ runtime test variant is treated as an
    # expected failure. No other side effects.
    fn._expected_failure_cpp_runtime_non_strict = True
    return fn


# Controls tests generated in test/export/test_export_legacy.py
def expectedFailureLegacyExportStrict(fn):
    # [FUNCTION DOC] Decorator that sets the marker attribute _expected_failure_legacy_export
    # to true on the given test function and returns the same function, so the generated
    # legacy strict export test variant is treated as an expected failure. No other side
    # effects.
    fn._expected_failure_legacy_export = True
    return fn


def expectedFailureLegacyExportNonStrict(fn):
    # [FUNCTION DOC] Decorator that sets the marker attribute
    # _expected_failure_legacy_export_non_strict to true on the given test function and
    # returns the same function, so the generated legacy non-strict export test variant is
    # treated as an expected failure. No other side effects.
    fn._expected_failure_legacy_export_non_strict = True
    return fn
