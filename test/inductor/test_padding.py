# Owner(s): ["module: inductor"]
# [MODULE DOC] Inductor tests for comprehensive padding of tensor strides and matmul shapes.
# Defines env-flag driven switches (perf, accuracy, stack, cudagraphs), an optional
# transformers import check, helper models and stride helpers, a GPU test base class, perf-
# only test classes and the PaddingTest class. Runs tests only when a GPU is available.
import copy
import functools
import os
import unittest

import torch
from torch import nn, Tensor
from torch._dynamo.convert_frame import maybe_cprofile
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.testing import rand_strided, reduce_to_scalar_loss
from torch._inductor import config, ir, metrics
from torch._inductor.fx_passes import pad_mm as pad_mm_pass
from torch._inductor.runtime.benchmarking import benchmarker
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import ceildiv, run_and_get_code
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    serialTest,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU, requires_gpu


DO_PERF_TEST = os.environ.get("DO_PERF_TEST") == "1"
DO_ACC_TEST = os.environ.get("DO_ACC_TEST", "1") == "1"
WITH_STACK = os.environ.get("WITH_STACK") == "1"
USE_CUDA_GRAPHS = os.environ.get("USE_CUDA_GRAPHS", "1") == "1"

try:
    import transformers  # noqa: F401

    HAS_TRANSFORMER = True
except ImportError:
    HAS_TRANSFORMER = False


def get_optim(m):
    # [FUNCTION DOC] HELPER: Builds an Adam optimizer over the model parameters with
    # learning rate 0.01, capturable and foreach enabled.
    return torch.optim.Adam(m.parameters(), lr=0.01, capturable=True, foreach=True)


def gen_transformer_inputs(vocab_size, bs, seq_length):
    # [FUNCTION DOC] HELPER: Returns a dict with input_ids and labels, each a random int64
    # tensor of shape batch size by sequence length with values below vocab_size, using the
    # nested generator for both.
    def geninp():
        # [FUNCTION DOC] HELPER: Generates one random int64 token tensor without gradient,
        # using the enclosing vocab size, batch size and sequence length.
        return torch.randint(
            0, vocab_size, (bs, seq_length), dtype=torch.int64, requires_grad=False
        )

    input_dict = {"input_ids": geninp(), "labels": geninp()}
    return input_dict


def get_padded_stride(shape, alignment_bytes, pad_output, itemsize):
    # [FUNCTION DOC] HELPER: Computes the expected strides of a contiguous tensor of the
    # given shape. The alignment in elements is the byte alignment divided by item size;
    # going from the innermost dimension outward, each stride is rounded up to a multiple of
    # that alignment when pad_output is true. Returns a tuple.
    align = alignment_bytes // itemsize
    new_strides = [0 for _ in range(len(shape))]
    new_strides[len(shape) - 1] = 1
    for i in range(len(shape) - 1, 0, -1):
        stride = shape[i] * new_strides[i]
        if pad_output and stride % align != 0:
            stride = (stride + align - 1) // align * align
        new_strides[i - 1] = stride
    return tuple(new_strides)


class LinearAndSoftmax(nn.Module):
    # [CLASS DOC] HELPER: Toy model of a linear layer feeding a cross entropy loss,
    # mimicking the vocabulary projection plus softmax of transformers. Holds the vocab
    # size, a 768-to-vocab linear layer, and a loss module.
    """
    It's very common that a transformer model will do a matmul and then
    softmax/log_softmax in the end.

    Creating this toy model to capture the pattern and make sure we do
    proper padding.
    """

    def __init__(self, vocab_size=30523, bias=True):
        # [FUNCTION DOC] HELPER: Stores the vocab size and creates the linear layer
        # (optional bias) and the cross entropy loss module.
        """
        The default vocab size for BertForMaskedLM is 30522.
        We run a few test cases with good or bad vocab_size around Bert's
        default value.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.linear = nn.Linear(768, vocab_size, bias=bias)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x, label):
        # [FUNCTION DOC] HELPER: Applies the linear layer to x, flattens logits to rows of
        # vocab size and labels to a flat vector, and returns the cross entropy loss.
        x = self.linear(x)
        return self.ce(x.view(-1, self.vocab_size), label.view(-1))

    def get_example_inputs(self, batch_size=16):
        # [FUNCTION DOC] HELPER: Returns a pair of random inputs: a float tensor of batch by
        # 512 by 768 and random integer labels in the vocab range of shape batch by 512.
        return torch.randn(batch_size, 512, 768), torch.randint(
            0, self.vocab_size, (batch_size, 512)
        )


def forward_and_backward_pass(m, inputs):
    # [FUNCTION DOC] HELPER: Runs the model on the inputs, sums the output and runs
    # backward.
    m(*inputs).sum().backward()


@config.patch(
    {
        "benchmark_kernel": True,
        "triton.unique_kernel_names": True,
        "triton.cudagraphs": USE_CUDA_GRAPHS,
    }
)
@requires_gpu()
class TestCaseBase(TestCase):
    # [CLASS DOC] HELPER: Base class for the padding tests. Patched with kernel
    # benchmarking, unique kernel names and cudagraphs per the USE_CUDA_GRAPHS env flag, and
    # requires a GPU. Provides class-level precision and default-device setup, a closeness
    # check, a compiled-vs-eager numeric check and a profiling helper.
    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] HELPER: When a GPU exists, saves the current float32 matmul
        # precision and default device on the class, sets precision to highest on ROCm and
        # high otherwise, and makes the GPU the default device.
        if HAS_GPU:
            cls.prior_float32_matmul_precision = torch.get_float32_matmul_precision()
            cls.prior_default_device = torch.get_default_device()
            if torch.version.hip:
                torch.set_float32_matmul_precision("highest")
            else:
                torch.set_float32_matmul_precision("high")
            torch.set_default_device(GPU_TYPE)

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] HELPER: When a GPU exists, restores the saved matmul precision and
        # default device, then clears the saved class attributes.
        if HAS_GPU:
            torch.set_float32_matmul_precision(cls.prior_float32_matmul_precision)
            torch.set_default_device(cls.prior_default_device)

            cls.prior_float32_matmul_precision = None
            cls.prior_default_device = None

    def check_close(self, ref, act, tol=1e-3):
        # [FUNCTION DOC] HELPER: Asserts two results are close with the given absolute and
        # relative tolerance. For Longformer masked LM outputs it compares the loss fields,
        # for sequence classifier outputs the logits, and for dicts containing a loss entry
        # the loss values. The failure message includes both values.
        if type(ref).__name__ == "LongformerMaskedLMOutput":
            ref = ref.loss
            act = act.loss
        if type(ref).__name__ == "SequenceClassifierOutput":
            ref = ref.logits
            act = act.logits
        if isinstance(ref, dict) and "loss" in ref:
            ref = ref["loss"]
            act = act["loss"]
        self.assertTrue(
            torch.allclose(ref, act, atol=tol, rtol=tol),
            lambda msg: f"{msg}\nref:\n{ref}\nact:\n{act}",
        )

    def common_numeric_check(self, f, *args, tol=1e-3, **kwargs):
        # [FUNCTION DOC] HELPER: Runs the function eagerly, compiles it and runs the
        # compiled version with the same arguments, then asserts the results are close using
        # the given tolerance.
        ref = f(*args, **kwargs)
        opt_f = torch.compile(f)
        act = opt_f(*args, **kwargs)
        self.check_close(ref, act, tol)

    def do_profiling(
        self,
        f_lhs,
        f_rhs,
        tag_lhs="With padding",
        tag_rhs="Without padding",
        args=(),
        kwargs=None,
    ):
        # [FUNCTION DOC] HELPER: Profiles two callables with the torch profiler: after a
        # device synchronize, runs three iterations of each, labeled by the tags, passing
        # the args and kwargs, synchronizes, and writes a chrome trace to a fixed file under
        # the tmp directory, printing its path.
        if kwargs is None:
            kwargs = {}
        device_interface = get_interface_for_device(GPU_TYPE)
        device_interface.synchronize()
        with torch.profiler.profile(with_stack=WITH_STACK) as p:
            niter = 3
            for _ in range(niter):
                with torch.profiler.record_function(tag_lhs):
                    f_lhs(*args, **kwargs)

                with torch.profiler.record_function(tag_rhs):
                    f_rhs(*args, **kwargs)
            device_interface.synchronize()

        profile_path = "/tmp/chrome.json"
        p.export_chrome_trace(profile_path)
        print(f"Chrome trace is written to {profile_path}")


class PerfTestBetweenGoodAndBadShape(TestCaseBase):
    # [CLASS DOC] Perf-only tests comparing good (aligned) and bad (unaligned) vocab sizes.
    @unittest.skipIf(not DO_PERF_TEST, "Perf test not enabled")
    def test_nobias_LinearAndSoftmax_both_shapes(self):
        # [FUNCTION DOC] Perf test: skipped unless the DO_PERF_TEST environment flag is on.
        # Runs the shared body test with bias disabled.
        self.test_LinearAndSoftmax_both_shapes(bias=False)

    @unittest.skipIf(not DO_PERF_TEST, "Perf test not enabled")
    def test_LinearAndSoftmax_both_shapes(self, bias=True):
        # [FUNCTION DOC] Perf test: skipped unless the DO_PERF_TEST environment flag is on.
        # Compiles the toy linear and softmax model with a bad vocab size and a good vocab
        # size, benchmarks forward plus backward of each on the GPU, and prints the
        # latencies. No assertions.
        """
        Compare the perf with good and bad shape.
        """
        m_bad_shape = LinearAndSoftmax(vocab_size=30523, bias=bias)
        inptus_bad_shape = m_bad_shape.get_example_inputs()
        m_good_shape = LinearAndSoftmax(vocab_size=30528, bias=bias)
        inputs_good_shape = m_good_shape.get_example_inputs()

        m_bad_shape_opt = torch.compile(m_bad_shape)
        m_good_shape_opt = torch.compile(m_good_shape)

        latency_good_shape = benchmarker.benchmark_gpu(
            lambda: forward_and_backward_pass(m_good_shape_opt, inputs_good_shape)
        )
        latency_bad_shape = benchmarker.benchmark_gpu(
            lambda: forward_and_backward_pass(m_bad_shape_opt, inptus_bad_shape)
        )
        print(
            f"Latency for good shape v.s. bad shape: {latency_good_shape:.3f}ms v.s. {latency_bad_shape:.3f}ms"
        )

    @unittest.skipIf(not DO_PERF_TEST or not HAS_TRANSFORMER, "Perf test not enabled")
    def test_BertForMaskedLM(self, num_layers=1):
        # [FUNCTION DOC] Perf test: skipped unless the DO_PERF_TEST environment flag is on.
        # Also needs the transformers package. Builds compiled Bert masked LM training steps
        # for vocab sizes 30528 (aligned) and 30522 (not), benchmarks both, prints latencies
        # and writes a profile. No assertions.
        """
        Compare the perf between doing padding and good shape.
        """
        from transformers import BertForMaskedLM

        config_cls = BertForMaskedLM.config_class
        bs = 16
        seq_length = 512

        def create_model(vocab_size):
            # [FUNCTION DOC] HELPER: Builds a Bert masked LM config with the given vocab
            # size and layer count, creates random inputs and the model and an optimizer,
            # and returns the compiled train step with its inputs.
            config = config_cls()
            config.num_hidden_layers = num_layers
            config.vocab_size = vocab_size
            inputs = gen_transformer_inputs(config.vocab_size, bs, seq_length)
            model = BertForMaskedLM(config)

            optim = get_optim(model)

            def f(**inputs):
                # [FUNCTION DOC] HELPER: One training step: clears gradients, runs the model
                # under autocast, takes the first output as loss, runs backward and steps
                # the optimizer.
                optim.zero_grad(True)
                with torch.autocast(GPU_TYPE):
                    pred = model(**inputs)
                    loss = pred[0]
                loss.backward()
                optim.step()

            return torch.compile(f), inputs

        f_good_shape, inputs_good_shape = create_model(30528)
        f_bad_shape, inputs_bad_shape = create_model(30522)

        print("benchmark for good shape")
        latency_good_shape = benchmarker.benchmark_gpu(
            lambda: f_good_shape(**inputs_good_shape)
        )
        print("benchmark for bad shape")
        latency_bad_shape = benchmarker.benchmark_gpu(
            lambda: f_bad_shape(**inputs_bad_shape)
        )
        print(
            f"Latency with good and bad shape: {latency_good_shape:.3f} v.s. {latency_bad_shape:.3f}"
        )

        self.do_profiling(
            lambda: f_good_shape(**inputs_good_shape),
            lambda: f_bad_shape(**inputs_bad_shape),
            tag_lhs="With good shape",
            tag_rhs="With bad shape",
        )


class PerfTestWithAndWithoutPadding(TestCaseBase):
    # [CLASS DOC] Tests that run models with accuracy checks and, if enabled, perf
    # comparison with and without comprehensive padding.
    @maybe_cprofile
    def run_acc_and_perf_test(self, model, inputs, perf_inputs=None, tol=1e-3):
        # [FUNCTION DOC] Wrapped by the optional cprofile decorator. Runs an accuracy test
        # (eval mode, compiled vs eager, unless the DO_ACC_TEST env flag disables it) and
        # then, when DO_PERF_TEST is on, benchmarks training steps compiled with
        # comprehensive padding on and off using deep copies of the model and fresh
        # optimizers, prints latencies and profiles both. perf_inputs defaults to inputs.
        """
        Run accuracy test.

        Also compare the perf with and without the comprehensive padding if
        DO_PERF_TEST is true.
        """
        if perf_inputs is None:
            perf_inputs = inputs

        def _process_inputs(x):
            # [FUNCTION DOC] HELPER: Splits an input into a positional argument list and
            # keyword dict: a dict becomes keyword arguments, anything else is wrapped into
            # a list unless the outer inputs value is already a tuple or list. Returns the
            # pair.
            """
            return args and kwargs
            """
            if isinstance(x, dict):
                return [], x

            if not isinstance(inputs, (tuple, list)):
                x = [x]

            return x, {}

        args, kwargs = _process_inputs(inputs)
        perf_args, perf_kwargs = _process_inputs(perf_inputs)

        if DO_ACC_TEST:
            model.eval()
            self.common_numeric_check(model, *args, **kwargs, tol=tol)
        else:
            print("Accuracy test skipped")

        model.train()

        if DO_PERF_TEST:
            print("Do performance test")

            def get_f(m, optim):
                # [FUNCTION DOC] HELPER: Returns a training-step function for the given
                # model and optimizer.
                def f(*args, **kwargs):
                    # [FUNCTION DOC] HELPER: Training step: clears gradients, runs the model
                    # under autocast, reduces output to a scalar loss, runs backward and
                    # steps the optimizer.
                    optim.zero_grad(True)
                    with torch.autocast(GPU_TYPE):
                        pred = m(*args, **kwargs)
                        loss = reduce_to_scalar_loss(pred)
                    loss.backward()
                    optim.step()

                return f

            latency_with_padding = None
            print("Benchmark with padding")
            with config.patch(comprehensive_padding=True):
                m_copy_with_padding = copy.deepcopy(model)
                optim_with_padding = get_optim(m_copy_with_padding)
                opt_f_with_padding = torch.compile(
                    get_f(m_copy_with_padding, optim_with_padding)
                )
                latency_with_padding = benchmarker.benchmark_gpu(
                    lambda: opt_f_with_padding(*perf_args, **perf_kwargs)
                )
            latency_without_padding = None
            print("bencmark without padding")
            with config.patch(comprehensive_padding=False):
                m_copy_without_padding = copy.deepcopy(model)
                optim_without_padding = get_optim(m_copy_without_padding)
                opt_f_without_padding = torch.compile(
                    get_f(m_copy_without_padding, optim_without_padding)
                )
                latency_without_padding = benchmarker.benchmark_gpu(
                    lambda: opt_f_without_padding(*perf_args, **perf_kwargs)
                )
            print(
                f"Latency with and without padding: {latency_with_padding:.3f} v.s. {latency_without_padding:.3f}"
            )

            # profiling
            self.do_profiling(
                opt_f_with_padding,
                opt_f_without_padding,
                args=perf_args,
                kwargs=perf_kwargs,
            )

    def test_nvidia_deeprecommender(self):
        # [FUNCTION DOC] Builds a deep-recommender style stack of linear and SELU layers
        # (with a dropout after the third linear) on very wide layer sizes and runs the
        # accuracy and perf helper with a small input and a larger perf input.
        """
        Compared the perf with and without comprehensive padding.
        """
        layer_sizes = [197951, 512, 512, 1024, 512, 512, 197951]
        x = torch.randn(4, layer_sizes[0])

        class Model(nn.Module):
            # [CLASS DOC] HELPER: Sequential stack of linear and SELU layers built from the
            # enclosing layer sizes list, with a dropout inserted after the third linear.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Builds the module list of linear and SELU layers
                # and wraps it in a sequential container.
                super().__init__()
                mod_list = []
                for i in range(len(layer_sizes) - 1):
                    mod_list.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
                    mod_list.append(nn.SELU())

                    if i == 2:
                        mod_list.append(nn.Dropout(0.8))
                self.seq = nn.Sequential(*mod_list)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Runs the sequential stack on the input.
                return self.seq(x)

        m = Model()
        perf_inputs = torch.randn(256, layer_sizes[0])
        self.run_acc_and_perf_test(m, x, perf_inputs)

    @unittest.skipIf(not DO_PERF_TEST or not HAS_TRANSFORMER, "Perf test not enabled")
    def test_longformer(self, bs=4):
        # [FUNCTION DOC] Perf test: skipped unless the DO_PERF_TEST environment flag is on.
        # Also needs the transformers package and downloads the Longformer base
        # configuration. Builds a masked LM from that config and runs the accuracy and perf
        # helper on random token inputs of length 1024.
        from transformers import AutoConfig, AutoModelForMaskedLM

        config = AutoConfig.from_pretrained("allenai/longformer-base-4096")
        model = AutoModelForMaskedLM.from_config(config)

        vocab_size = model.config.vocab_size
        seq_length = 1024
        input_dict = gen_transformer_inputs(vocab_size, bs, seq_length)

        self.run_acc_and_perf_test(model, input_dict)

    @unittest.skipIf(not DO_PERF_TEST or not HAS_TRANSFORMER, "Perf test not enabled")
    def test_longformer_small_bs(self):
        # [FUNCTION DOC] Perf test: skipped unless the DO_PERF_TEST environment flag is on.
        # Also needs the transformers package. Runs the Longformer test with batch size 2.
        """
        The model exists in both HF and TB. In TB it uses a smaller batch size.
        """
        self.test_longformer(bs=2)


@instantiate_parametrized_tests
class PaddingTest(TestCaseBase):
    # [CLASS DOC] Tests of comprehensive padding: stride padding, codegen alignment, output
    # padding and dynamic shape handling. Inherits GPU setup from the base class.
    @unittest.skipIf(not DO_PERF_TEST, "Perf test not enabled")
    def test_mm_padding_perf(self):
        # [FUNCTION DOC] Perf test: skipped unless the DO_PERF_TEST environment flag is on.
        # Benchmarks plain mm and a padded mm for several bad and good shapes and prints the
        # latencies. No assertions.
        def naive_mm(a, b):
            # [FUNCTION DOC] HELPER: Returns the matrix product of the two inputs.
            return a @ b

        def _compute_padding(s, align):
            # [FUNCTION DOC] HELPER: Returns how many elements must be added to a size to
            # reach the next multiple of the alignment.
            return (s + align - 1) // align * align - s

        @torch.compile
        def pad_mm(a, b, align=16):
            # [FUNCTION DOC] HELPER: Compiled function that computes padding amounts for M,
            # K and N from the alignment and calls the pad_mm pass function. Only the
            # described dimensions are padded to the alignment.
            """
            NOTE: this function only pad a single dimension which is good
            enough for testing.
            """
            m_padding = _compute_padding(a.size(0), align)
            k_padding = _compute_padding(a.size(1), align)
            n_padding = _compute_padding(b.size(1), align)
            return pad_mm_pass.pad_mm(a, b, m_padding, k_padding, n_padding)

        for M, K, N, f in (
            (8192, 768, 30523, naive_mm),
            (8192, 768, 30523, pad_mm),
            (8192, 768, 30528, naive_mm),
            (30523, 8192, 768, naive_mm),
            (30528, 8192, 768, naive_mm),
        ):
            a = torch.randn(M, K)
            b = torch.randn(K, N)
            ms = benchmarker.benchmark_gpu(lambda: f(a, b))
            print(f"MxKxN {M}x{K}x{N} {f.__name__}: {ms:.3f}ms")

    @unittest.skipIf(not DO_PERF_TEST, "Perf test not enabled")
    def test_padmm(self):
        # [FUNCTION DOC] Perf test: skipped unless the DO_PERF_TEST environment flag is on.
        # Compares an unpadded half-precision matmul with a version that concatenates zero
        # padding of 6 to the K dimension of both operands before an aten mm, benchmarks
        # both, prints latencies and profiles. No assertions.
        """
        Latency between original matmul and padded matmul: 2.717 v.s. 2.356
        """
        mat1_pad = torch.randn(8192, 30522, dtype=torch.float16)
        mat2_pad = torch.randn(30522, 768, dtype=torch.float16)

        def f():
            # [FUNCTION DOC] HELPER: Returns the matrix product of the two captured tensors.
            return mat1_pad @ mat2_pad

        def pad_dim(x: Tensor, padded_length: int, dim: int) -> Tensor:
            # [FUNCTION DOC] HELPER: Returns the tensor with zeros of the requested length
            # concatenated on the given dimension.
            pad = x.new_zeros(*x.shape[:dim], padded_length, *x.shape[dim + 1 :])
            return torch.cat([x, pad], dim=dim)

        @torch.compile(fullgraph=True, options={"triton.cudagraphs": False})
        def g():
            # [FUNCTION DOC] HELPER: Compiled function (full graph, no cudagraphs) padding
            # both captured matrices on the K dimension and returning their aten mm.
            mat1 = mat1_pad
            mat2 = mat2_pad
            mat1 = pad_dim(mat1, 6, 1)
            mat2 = pad_dim(mat2, 6, 0)
            return torch.ops.aten.mm(mat1, mat2)

        ori_time = benchmarker.benchmark_gpu(f)
        pad_time = benchmarker.benchmark_gpu(g)

        print(
            f"Latency between original matmul and padded matmul: {ori_time:.3f} v.s. {pad_time:.3f}"
        )
        self.do_profiling(f, g, "No MM Padding", "With mm padding")

    @unittest.skipIf(not DO_PERF_TEST, "Perf test not enabled")
    def test_matmul(self):
        # [FUNCTION DOC] Perf test: skipped unless the DO_PERF_TEST environment flag is on.
        # Compares compiled mm with out argument on aligned and unaligned shapes (the
        # unaligned input is strided), prints latencies and profiles. No assertions.
        """
        Latency with good and bad shapes: 1.705 v.s. 2.625
        """
        x_good_shape = torch.randn(8192, 30528, dtype=torch.float16)
        weight_good_shape = torch.randn(30528, 768, dtype=torch.float16)
        out_good_shape = torch.randn(8192, 768, dtype=torch.float16)

        # Using stride (30522, 1) does not make a difference here.
        x_bad_shape = rand_strided(
            (8192, 30522), (30528, 1), device=GPU_TYPE, dtype=torch.float16
        )
        weight_bad_shape = torch.randn(30522, 768, dtype=torch.float16)
        out_bad_shape = torch.randn(8192, 768, dtype=torch.float16)

        def f(x, weight, out):
            # [FUNCTION DOC] HELPER: Runs mm writing into the given output tensor and
            # returns that tensor.
            torch.mm(x, weight, out=out)
            return out

        f1 = torch.compile(
            functools.partial(f, x_good_shape, weight_good_shape, out_good_shape)
        )
        f2 = torch.compile(
            functools.partial(f, x_bad_shape, weight_bad_shape, out_bad_shape)
        )
        latency_good_shape = benchmarker.benchmark_gpu(f1)
        latency_bad_shape = benchmarker.benchmark_gpu(f2)
        print(
            f"Latency with good and bad shapes: {latency_good_shape:.3f} v.s. {latency_bad_shape:.3f}"
        )
        self.do_profiling(f1, f2)

    @serialTest()
    def test_nobias_LinearAndSoftmax_codegen(self):
        # [FUNCTION DOC] Runs the shared codegen test with bias disabled. Marked serial.
        self.test_LinearAndSoftmax_codegen(bias=False)

    def test_LinearAndSoftmax_codegen(self, bias=True):
        # [FUNCTION DOC] Compiles the linear and softmax model with an unaligned vocab size,
        # runs forward plus backward capturing generated code, then compares the weight
        # gradient with eager. Asserts there are two wrapper codes (forward, backward) and
        # that the forward code loads softmax input with the row stride of the padded vocab
        # size 30528. Optionally prints latency.
        m_bad_shape = LinearAndSoftmax(vocab_size=30523, bias=bias)
        inputs_bad_shape = m_bad_shape.get_example_inputs()
        m_bad_shape_opt = torch.compile(copy.deepcopy(m_bad_shape))

        _, wrapper_codes = run_and_get_code(
            forward_and_backward_pass, m_bad_shape_opt, inputs_bad_shape
        )
        forward_and_backward_pass(m_bad_shape, inputs_bad_shape)
        self.assertEqual(
            m_bad_shape.linear.weight.grad, m_bad_shape_opt.linear.weight.grad
        )
        self.assertTrue(len(wrapper_codes) == 2)  # one for forward and one for backward
        forward_wrapper = wrapper_codes[0]

        # make sure the load for softmax is aligned
        self.assertTrue(
            "tl.load(in_ptr0 + (r0_1 + 30528*x0)" in forward_wrapper,
            lambda msg: f"{msg}\nforward_wrapper: {forward_wrapper}",
        )

        if DO_PERF_TEST:
            latency = benchmarker.benchmark_gpu(
                lambda: forward_and_backward_pass(m_bad_shape_opt, inputs_bad_shape)
            )
            print(f"latency: {latency:.3f}ms")

    @config.patch(pattern_matcher=False)
    def test_attention(self):
        # [FUNCTION DOC] With the pattern matcher off, builds a small attention module and
        # checks compiled output matches eager.
        batch_size, seq_len, num_heads, hidden_size = 1, 4, 1, 16
        inv_scale = (num_heads / hidden_size) ** 0.5

        class Attention(nn.Module):
            # [CLASS DOC] HELPER: Single-head-style attention block with query, key and
            # value linear layers sized by the enclosing hidden size, using the enclosing
            # batch, sequence and head counts.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates three linear projection layers for query,
                # key and value.
                super().__init__()
                self.query = nn.Linear(hidden_size, hidden_size)
                self.key = nn.Linear(hidden_size, hidden_size)
                self.value = nn.Linear(hidden_size, hidden_size)

            @staticmethod
            def reshape(x):
                # [FUNCTION DOC] HELPER: Static method: views the input into batch,
                # sequence, heads and per-head size, then permutes to put heads before
                # sequence.
                return x.view(batch_size, seq_len, num_heads, -1).permute(0, 2, 1, 3)

            @staticmethod
            def cancel_reshape(x):
                # [FUNCTION DOC] HELPER: Static method: inverse of the reshape helper,
                # permuting heads back after sequence and viewing as batch, sequence,
                # hidden.
                return x.permute(0, 2, 1, 3).view(batch_size, seq_len, hidden_size)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Projects the input, computes scaled attention
                # weights with softmax over the last dimension, applies them to the values
                # and returns the merged heads.
                query, key, value = self.query(x), self.key(x), self.value(x)
                weights = (
                    torch.matmul(
                        self.reshape(query), self.reshape(key).permute(0, 1, 3, 2)
                    )
                    * inv_scale
                ).softmax(dim=-1)
                return self.cancel_reshape(torch.matmul(weights, self.reshape(value)))

        attn = Attention()
        x = torch.randn(batch_size, seq_len, hidden_size)

        self.common_numeric_check(attn, x)

    def test_view(self):
        # [FUNCTION DOC] Checks that a compiled view of a 2D tensor into a 3D shape matches
        # eager.
        def f(x):
            # [FUNCTION DOC] HELPER: Views the input as three by three by three.
            return x.view(3, 3, 3)

        x = torch.randn(3, 9)
        self.common_numeric_check(f, x)

    def test_pad_strides(self):
        # [FUNCTION DOC] Checks the stride padding helper pads the inner stride to 2048 and
        # consequently the outer one, given sizes 2, 16, 2047 and contiguous strides, for
        # float32.
        """
        Note that dim0's stride is also padded even though its previous value
        is already multiple of 16. The reason is we padded dim1's stride.
        We have to correspondingly increase the stride for dim0.
        """
        sizes = [2, 16, 2047]
        in_strides = [2047 * 16, 2047, 1]
        out_strides = list(ir.Layout._pad_strides(in_strides, sizes, torch.float32))
        expected_strides = [2048 * 16, 2048, 1]
        self.assertEqual(
            expected_strides,
            out_strides,
            lambda msg: f"{msg}\n{expected_strides} v.s. {out_strides}",
        )

    def test_pad_strides_skip(self):
        # [FUNCTION DOC] Checks the stride padding helper leaves strides unchanged for sizes
        # 2, 32, 127 with the contiguous strides, because padding would add too much memory.
        """
        The padding is skipped to avoid too much memory overhead.
        """
        sizes = [2, 32, 127]
        in_strides = [4064, 127, 1]
        out_strides = list(ir.Layout._pad_strides(in_strides, sizes, torch.float32))
        expected_strides = [4064, 127, 1]
        self.assertEqual(
            expected_strides,
            out_strides,
            lambda msg: f"{msg}\n{expected_strides} v.s. {out_strides}",
        )

    def test_pad_3d_tensor(self):
        # [FUNCTION DOC] Compiles a matmul, add, matmul chain on a 3D input (so intermediate
        # strides can be padded), checks numerics with loose tolerance and that the
        # comprehensive padding metric counter is positive.
        """
        Constructing this test case guided by the fact that we don't pad
        placeholder or user visible output's strides.

        Add a matmul in the beginning and end so we can pad strides for
        intermediate tensors.
        """

        def f(x, y):
            # [FUNCTION DOC] HELPER: Matrix-multiplies x by y, adds one, and multiplies by y
            # again.
            x = torch.matmul(x, y)
            x = x + 1
            return torch.matmul(x, y)

        x = torch.randn(2, 16, 2047)
        y = torch.randn(2047, 2047)
        self.common_numeric_check(f, x, y, tol=1e-2)
        self.assertTrue(metrics.num_comprehensive_padding > 0)

    def test_conv(self):
        # [FUNCTION DOC] Runs a 3x3 convolution on a contiguous input and on a copy with
        # padded strides and checks the results are close. With the perf flag on it also
        # times both and profiles.
        """
        Padding the input for convolution may cause extra copy kernel being called.
        Check this example trace: https://gist.github.com/shunting314/ce45398f7d51a63ce05fc8d411faddb3
        """
        x_shape = (1, 128, 640, 959)
        x1 = torch.randn(*x_shape)

        padded_stride = ir.Layout._pad_strides(x1.stride(), x1.shape, torch.float32)
        x2 = rand_strided(x_shape, padded_stride, device=GPU_TYPE)
        x2.copy_(x1)

        weight = torch.randn(64, 128, 3, 3)

        def fun(x, weight):
            # [FUNCTION DOC] HELPER: Calls the convolution op with unit stride, padding and
            # dilation, no groups and no bias.
            return torch.convolution(
                x,
                weight,
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                transposed=False,
                output_padding=(0, 0),
                groups=1,
                bias=None,
            )

        ref = fun(x1, weight)
        act = fun(x2, weight)
        self.check_close(ref, act)
        if DO_PERF_TEST:
            latency_with_padding = benchmarker.benchmark_gpu(lambda: fun(x2, weight))
            latency_without_padding = benchmarker.benchmark_gpu(lambda: fun(x1, weight))
            print(
                f"Latency with and without padding: {latency_with_padding:.3f} v.s. {latency_without_padding:.3f}"
            )

            self.do_profiling(lambda: fun(x2, weight), lambda: fun(x1, weight))

    @unittest.skipIf(not DO_PERF_TEST, "Perf test not enabled")
    def test_cat(self):
        # [FUNCTION DOC] Perf test: skipped unless the DO_PERF_TEST environment flag is on.
        # Compares eager concatenation of a half-precision tensor with six zero columns
        # against the compiled version with cudagraphs disabled, prints latencies and
        # profiles. No assertions.
        """
        Compare the perf between aten cat and compiled cat.

        Latency between eager and compiled: 1.596 v.s. 0.601

        Eager cat can be 2.66x slower than inductor kernel.
        """
        x = torch.randn(8192, 30522, dtype=torch.float16)

        def f(x):
            # [FUNCTION DOC] HELPER: Concatenates the input with a zero tensor of six
            # columns on dimension 1.
            pad = x.new_zeros(x.size(0), 6)
            return torch.cat([x, pad], dim=1)

        # disable cudagraphs since cudagraphs need copy the input which
        # distort the latency a lot! (double the latency here for compiled
        # version)
        with config.patch("triton.cudagraphs", False):
            opt_f = torch.compile(f)
            opt_f(x)
        eager_time = benchmarker.benchmark_gpu(lambda: f(x))
        opt_time = benchmarker.benchmark_gpu(lambda: opt_f(x))
        print(
            f"Latency between eager and compiled: {eager_time:.3f} v.s. {opt_time:.3f}"
        )
        self.do_profiling(lambda: f(x), lambda: opt_f(x), "Eager Cat", "Compiled Cat")

    def test_pad_channels_last(self):
        # [FUNCTION DOC] Checks the stride padding helper changes strides of a contiguous 4D
        # tensor but leaves a channels-last tensor's strides unchanged.
        t = torch.randn(2, 3, 5, 1025)
        in_strides = t.stride()
        out_strides = ir.Layout._pad_strides(in_strides, t.shape, torch.float32)
        self.assertTrue(in_strides != out_strides)

        t = t.to(memory_format=torch.channels_last)
        in_strides = t.stride()
        out_strides = ir.Layout._pad_strides(in_strides, t.shape, torch.float32)
        self.assertTrue(in_strides == out_strides)

    @parametrize("alignment_bytes", (32, 128))
    @parametrize("shape", [(21, 19), (3, 5, 71)])
    @parametrize("dtype", (torch.float16, torch.float32))
    def test_pad_outputs(
        self, dtype: torch.dtype, shape: tuple[int], alignment_bytes: int
    ):
        # [FUNCTION DOC] Parametrized over alignment bytes, shape and dtype. Compiles an add
        # under config enabling comprehensive padding and output padding, checks numerics,
        # then computes the expected strides by rounding each slice size up to the alignment
        # in elements and asserts the output is non-contiguous and has those strides. Also
        # asserts the alignment is a multiple of and larger than the element size.
        """
        Tests padding output tensors to a specific alignment.
        This is enabled by a config flag.
        """
        func = torch.add
        inputs = tuple(torch.randn(*shape, dtype=dtype) for input_idx in range(2))

        # Compile and run
        with config.patch(
            {
                "comprehensive_padding": True,
                "padding_alignment_bytes": alignment_bytes,
                "padding_stride_threshold": 0,
                "pad_outputs": True,
            }
        ):
            compiled_func = torch.compile(func)
            compiled_out = compiled_func(*inputs)

        # Check numerics
        eager_out = func(*inputs)
        self.check_close(eager_out, compiled_out)

        # Compute the expected padding
        element_size = torch.tensor([], dtype=dtype).element_size()
        self.assertGreater(alignment_bytes, element_size)
        self.assertEqual(alignment_bytes % element_size, 0)
        alignment_elements = alignment_bytes // element_size
        contiguous_stride = inputs[0].stride()
        expected_stride = [1]
        for dim in reversed(shape[1:]):
            slice_size = dim * expected_stride[0]
            new_stride = alignment_elements * ceildiv(slice_size, alignment_elements)
            expected_stride.insert(0, new_stride)
        expected_stride = tuple(expected_stride)
        self.assertNotEqual(expected_stride, contiguous_stride)

        # Check strides
        self.assertFalse(compiled_out.is_contiguous())
        self.assertEqual(compiled_out.stride(), expected_stride)

    @parametrize(
        "shape,alignment_bytes,pad_output",
        [
            ((512, 1), 32, False),
            ((512, 1), 32, True),
            ((32, 30), 64, False),
            ((32, 30), 64, True),
        ],
    )
    def test_noop_concat_output_padding(self, shape, alignment_bytes, pad_output):
        # [FUNCTION DOC] Parametrized over shape, alignment and whether to pad. Compiles
        # concatenation of twelve slightly padded input views on dimension 0 with a config
        # suited to a no-op concat, and asserts the generated first code contains the
        # expected output allocation line with the input's strides for buffer 12.
        """
        When we generate no-op concat kernel, alignment of the inputs
        and outputs should be honored based on padding_alignment_bytes.
        """

        def get_input(size: tuple[int], alignment_bytes: int) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Makes a float32 tensor whose last dimension is rounded
            # up to the alignment (if padding is on) and returns a strided view with the
            # requested size and the padded strides.
            size_padded = list(size)
            elem_size = 4  # float32
            pad_elems = alignment_bytes // elem_size
            if pad_output:
                size_padded[-1] = (
                    (size_padded[-1] + pad_elems - 1) // pad_elems * pad_elems
                )
            full = torch.randn(size_padded, dtype=torch.float32)
            view = torch.as_strided(full, size, full.stride())
            return view

        num_inputs = 12
        input_tensors = [get_input(shape, alignment_bytes) for _ in range(num_inputs)]

        config_patches = {
            "compile_threads": 1,
            "comprehensive_padding": pad_output,
            "cpu_backend": "triton",
            "disable_padding_cpu": False,
            "implicit_fallbacks": False,
            "inplace_buffers": False,
            "padding_alignment_bytes": alignment_bytes,
            "pad_channels_last": True,
            "pad_outputs": True,
            "padding_stride_threshold": 0,
            "triton.prefer_nd_tiling": True,
            "triton.use_block_ptr": True,
            "triton.codegen_upcast_to_fp32": False,
            "unroll_reductions_threshold": 1,
        }
        with config.patch(config_patches):
            compiled = torch.compile(torch.cat)
            _, code = run_and_get_code(compiled, input_tensors, 0)

        output_shape = (shape[0] * num_inputs, shape[1])
        output_stride = input_tensors[0].stride()
        output_line = f"buf12 = empty_strided_{GPU_TYPE}({output_shape}, {output_stride}, torch.float32)"
        self.assertTrue(output_line in code[0])

    @requires_gpu()
    def test_concat_output_no_redundant_copy_with_padding(self):
        # [FUNCTION DOC] Requires a GPU. With comprehensive padding and output padding,
        # compiles a function whose concat output has padded strides and checks the result
        # matches eager and that the generated code has exactly one buffer allocation (no
        # redundant copy kernel).
        """
        When comprehensive_padding is enabled, ConcatKernel pads its output
        buffer strides. The graph output should accept the padded strides
        directly instead of generating a redundant copy kernel.
        """

        def f(x):
            # [FUNCTION DOC] HELPER: Adds one, multiplies that by two, and concatenates both
            # on dimension 1; the first value has two consumers.
            a = x + 1
            # a has two consumers (mul and cat), which forces ConcatKernel
            # over pointwise_cat. Both inputs are Pointwise with
            # FlexibleLayout so they realize directly into concat slices.
            b = a * 2
            return torch.cat([a, b], dim=1)

        # Use dim=131 so concat output dim (262) is not aligned to
        # padding_alignment_bytes/4=32, triggering stride padding.
        x = torch.randn(128, 131, device=GPU_TYPE)

        with config.patch(
            {
                "comprehensive_padding": True,
                "pad_outputs": True,
                "padding_stride_threshold": 0,
                "inplace_buffers": False,
            }
        ):
            result, code = run_and_get_code(torch.compile(f), x)

        ref = f(x)
        self.assertTrue(torch.allclose(ref, result, atol=1e-3, rtol=1e-3))
        # Only one output buffer should be allocated for the concat result.
        # Without the fix, a second empty_strided is allocated and a copy
        # kernel is generated to copy from the padded concat buffer to it.
        # Count actual buffer allocations (not import lines) by matching
        # "= empty_strided_<device>(" pattern.
        import re

        num_allocs = len(re.findall(rf"= empty_strided_{GPU_TYPE}\(", code[0]))
        self.assertEqual(
            num_allocs,
            1,
            "Expected exactly one buffer allocation for concat output (no redundant copy)",
        )

    @parametrize(
        "shape,alignment_bytes,enable_pad",
        [
            ((512, 1), 32, False),
            ((512, 1), 32, True),
            ((32, 30), 64, False),
            ((32, 30), 64, True),
            ((512, 100, 1), 32, False),
            ((512, 100, 1), 32, True),
            ((32, 50, 30), 64, False),
            ((32, 50, 30), 64, True),
        ],
    )
    def test_outer_dynamic_shape_padding(self, shape, alignment_bytes, enable_pad):
        # [FUNCTION DOC] Parametrized over shape, alignment and pad flag. Marks dimension 0
        # of both inputs dynamic, compiles an add with the dynamic padding option on, and
        # asserts the result strides equal the expected padded strides from the helper
        # (padded only when the flag is on).
        """
        When only the outermost dim is dynamic shape, the output can still be padded up
        based on padding configuration.
        """
        num_inputs = 2
        input_tensors = [
            torch.randn(shape, dtype=torch.float32) for _ in range(num_inputs)
        ]

        config_patches = {
            "comprehensive_padding": enable_pad,
            "pad_dynamic_shapes": True,
            "cpu_backend": "triton",
            "padding_alignment_bytes": alignment_bytes,
            "pad_outputs": True,
            "padding_stride_threshold": 0,
        }
        with config.patch(config_patches):
            torch._dynamo.mark_dynamic(input_tensors[0], 0)
            torch._dynamo.mark_dynamic(input_tensors[1], 0)
            compiled = torch.compile(torch.add)
            result, _ = run_and_get_code(compiled, *input_tensors)

        expected_stride = get_padded_stride(
            result.shape, alignment_bytes, enable_pad, result.dtype.itemsize
        )
        self.assertEqual(result.stride(), expected_stride)

    @parametrize(
        "shape,perm,alignment_bytes,enable_pad",
        [
            ((500, 10, 1), (2, 1, 0), 32, False),
            ((500, 20, 1), (2, 1, 0), 32, True),
            ((30, 10, 20), (2, 1, 0), 64, True),
            ((30, 10, 20), (2, 1, 0), 64, False),
            ((500, 10, 1), (1, 2, 0), 32, False),
            ((500, 20, 1), (1, 2, 0), 32, True),
            ((30, 10, 20), (1, 2, 0), 64, True),
            ((30, 10, 20), (1, 2, 0), 64, False),
        ],
    )
    def test_perm_outer_dynamic_shape_padding(
        self, shape, perm, alignment_bytes, enable_pad
    ):
        # [FUNCTION DOC] Parametrized over shape, permutation, alignment and pad flag. Marks
        # the last dimension of a single input dynamic, compiles a permute plus contiguous
        # function and asserts the result strides equal the expected padded strides.
        """
        When only the outermost dim is dynamic shape, the output can still be padded up
        based on padding configuration. Test when this occurs after a permute op.
        """

        def permute_contig(x):
            # [FUNCTION DOC] HELPER: Permutes the input by the enclosing permutation and
            # makes it contiguous.
            return torch.permute(x, perm).contiguous()

        num_inputs = 1
        input_tensors = [
            torch.randn(shape, dtype=torch.float32) for _ in range(num_inputs)
        ]

        config_patches = {
            "comprehensive_padding": enable_pad,
            "pad_dynamic_shapes": True,
            "cpu_backend": "triton",
            "padding_alignment_bytes": alignment_bytes,
            "pad_outputs": True,
            "padding_stride_threshold": 0,
            "triton.use_block_ptr": True,
        }
        with config.patch(config_patches):
            torch._dynamo.mark_dynamic(input_tensors[0], 2)
            compiled = torch.compile(permute_contig)
            result, _ = run_and_get_code(compiled, *input_tensors)

        expected_stride = get_padded_stride(
            result.shape, alignment_bytes, enable_pad, result.dtype.itemsize
        )
        self.assertEqual(result.stride(), expected_stride)

    @parametrize(
        "shape,alignment_bytes,enable_pad",
        [
            ((512, 1), 32, False),
            ((512, 1), 32, True),
            ((32, 30), 64, False),
            ((32, 30), 64, True),
            ((512, 100, 1), 32, False),
            ((512, 100, 1), 32, True),
            ((32, 50, 30), 64, False),
            ((32, 50, 30), 64, True),
        ],
    )
    def test_dynamic_shape_padding(self, shape, alignment_bytes, enable_pad):
        # [FUNCTION DOC] Parametrized over shape, alignment and pad flag. Compiles an add
        # with dynamic shapes everywhere, with dynamic padding enabled only when the pad
        # flag is on, and asserts result strides equal the expected strides.
        """
        When only the outermost dim is dynamic shape, the output can still be padded up
        based on padding configuration.
        """
        num_inputs = 2
        input_tensors = [
            torch.randn(shape, dtype=torch.float32) for _ in range(num_inputs)
        ]

        config_patches = {
            "comprehensive_padding": enable_pad,
            "pad_dynamic_shapes": enable_pad,
            "cpu_backend": "triton",
            "padding_alignment_bytes": alignment_bytes,
            "pad_outputs": True,
            "padding_stride_threshold": 0,
        }
        with config.patch(config_patches):
            compiled = torch.compile(torch.add, dynamic=True)
            result, _ = run_and_get_code(compiled, *input_tensors)

        expected_stride = get_padded_stride(
            result.shape, alignment_bytes, enable_pad, result.dtype.itemsize
        )
        self.assertEqual(result.stride(), expected_stride)

    def test_reduction_comprehensive_padding_stride(self):
        # [FUNCTION DOC] Regression test for an issue with strides of user-visible
        # reductions: with comprehensive padding on, a compiled adaptive average pool then
        # flatten and sum must equal the eager result.
        """Comprehensive padding should not cause stride mismatches for
        user-visible reductions.

        Regression test for https://github.com/pytorch/pytorch/issues/179931
        """

        def program(x):
            # [FUNCTION DOC] HELPER: Applies adaptive average pooling to 7 by 7, flattens
            # from dimension 1 and sums over the last dimension.
            y = torch.nn.functional.adaptive_avg_pool2d(x, 7)
            return y.flatten(1).sum(dim=-1)

        x = torch.randn(4, 2049, 8, 8, dtype=torch.float32, device=GPU_TYPE)
        eager = program(x.clone())

        with config.patch({"comprehensive_padding": True}):
            compiled = torch.compile(program, backend="inductor")(x.clone())

        self.assertEqual(eager, compiled)


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
