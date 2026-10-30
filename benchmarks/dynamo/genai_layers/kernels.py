# [MODULE DOC] Concrete benchmark kernels for the genai layer benchmarks: cross entropy,
# softmax, RMSNorm and LayerNorm, each in a forward and a backward variant, derived from the
# BenchmarkKernel base class in the utils module. Each class lists its input shapes, its memory
# traffic model, and builders for the eager, compiled, quack and liger backends (each builder
# returns a zero-argument callable that is timed). Imports cutlass at import time, and
# quack / liger_kernel lazily inside the backend builders. Also defines extra_shapes_for_norm,
# the extra large shapes appended to the norm kernels.
from typing import Any

import cutlass
import cutlass.torch as cutlass_torch
from utils import BenchmarkKernel

import torch
import torch.nn.functional as F


# more important shapes used by internal models
extra_shapes_for_norm = (
    (1152 * 500, 384),
    (1152 * 500, 512),
    (1152 * 1000, 384),
    (1152 * 1000, 512),
)


class CrossEntropyForward(BenchmarkKernel):
    # [CLASS DOC] Benchmark of the cross entropy loss forward pass (per-row loss, no
    # reduction) on bfloat16 logits and int64 targets. Backends: eager, compiled, quack, liger.
    def __init__(self, script_args):
        # [FUNCTION DOC] Initialize the base benchmark and set the backend list to eager,
        # compiled, quack and liger.
        super().__init__(script_args)
        self.available_backends = ["eager", "compiled", "quack", "liger"]

    def get_shapes(self) -> tuple[tuple[int, ...], ...]:
        # [FUNCTION DOC] Return the fixed tuple of (rows, classes) shapes: 32768 rows with
        # 256 up to 65536 classes, then (16384, 131072) and (8192, 262144).
        return (
            (32768, 256),
            (32768, 512),
            (32768, 1024),
            (32768, 2048),
            (32768, 4096),
            (32768, 8192),
            (32768, 16384),
            (32768, 32768),
            (32768, 65536),
            (16384, 131072),
            (8192, 262144),
        )

    def get_memory_bytes(self, args, kwargs) -> int:
        # [FUNCTION DOC] Estimate memory traffic in bytes from the (logits, target) arguments:
        # reading logits, reading targets and writing one loss per row, all counted at the
        # logits element size.
        # Read x (M*N elements) + read target (M elements) + write loss (M elements)
        x, target = args
        M, N = x.shape
        dtype = x.dtype
        return (M * N + M + M) * dtype.itemsize

    def eager(self, args, kwargs=None) -> Any:
        # [FUNCTION DOC] Build the eager backend: returns a callable computing unreduced cross
        # entropy on the captured inputs. Raises AssertionError if kwargs is not None.
        if kwargs is not None:
            raise AssertionError(f"Expected kwargs to be None, but got {kwargs}")
        x, target = args
        return lambda: F.cross_entropy(x, target, reduction="none")

    def compiled(self, args, kwargs=None) -> Any:
        # [FUNCTION DOC] Build the compiled backend: marks the batch dimension of both inputs
        # dynamic, compiles the loss in full-graph mode using the configured compile mode, and
        # returns a callable running it. A wrapping function is used so compile traces it.
        # Raises AssertionError if kwargs is not None.
        if kwargs is not None:
            raise AssertionError(f"Expected kwargs to be None, but got {kwargs}")
        x, target = args

        # Mark batch size as dynamic for realistic workload
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(target, 0)

        # Need `lambda` otherwise torch.compile will not trace the function.
        # More discussion: https://github.com/pytorch/pytorch/issues/158455
        compiled_cross_entropy = torch.compile(
            lambda x, target: F.cross_entropy(x, target, reduction="none"),
            mode=self.compile_mode,
            fullgraph=True,
        )
        return lambda: compiled_cross_entropy(x, target)

    def quack(self, args, kwargs=None) -> Any:
        # [FUNCTION DOC] Build the quack backend: lazily imports the quack cross entropy
        # kernel and returns a callable invoking it. Raises AssertionError if kwargs is set.
        if kwargs is not None:
            raise AssertionError(f"Expected kwargs to be None, but got {kwargs}")
        x, target = args
        from quack.cross_entropy import _cross_entropy

        return lambda: _cross_entropy(x, target)

    def liger(self, args, kwargs=None) -> Any:
        # [FUNCTION DOC] Build the liger backend: lazily imports and constructs the Liger
        # cross entropy loss (no reduction) and returns a callable applying it. Raises
        # AssertionError if kwargs is not None.
        if kwargs is not None:
            raise AssertionError(f"Expected kwargs to be None, but got {kwargs}")
        from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss

        x, target = args
        cross_entropy = LigerCrossEntropyLoss(reduction="none")
        return lambda: cross_entropy(x, target)

    def benchmark(self):
        # [FUNCTION DOC] For every shape, allocate random bfloat16 logits (scaled by 0.1) and
        # random int64 targets on the GPU, then benchmark them via the base class helper with a
        # setting label naming the shape. Prints each shape.
        for M, N in self.get_shapes():
            print(f"\n Tensor dimensions: [{M}, {N}]")
            # quack requires cutlass dtype
            torch_dtype = cutlass_torch.dtype(cutlass.BFloat16)
            x = 0.1 * torch.randn(M, N, device="cuda", dtype=torch_dtype)
            target = torch.randint(0, N, (M,), device="cuda", dtype=torch.int64)
            self.benchmark_single_shape((x, target), setting=f"shape: [{M}, {N}]")

    def check_accuracy(self, args, kwargs) -> None:
        # [FUNCTION DOC] Run every backend once on cloned inputs, then compare each non-eager
        # result to the eager result with assert_close, printing a success or failure line.
        # The quack result is first cast to the eager dtype since it returns float32. Failures
        # are only printed (no exit, no custom tolerance), unlike the base class version.
        res = {}
        for backend in self.available_backends:
            args_ref, kwargs_ref = self.clone_inputs(args, kwargs)
            res[backend] = getattr(self, backend)(args_ref, kwargs_ref)()
        gold = res["eager"]
        for backend in self.available_backends:
            if backend == "eager":
                continue
            if backend == "quack":
                # quack's cross_entropy only returns float32 loss output.
                # Need to convert it to the same dtype as gold for comparison.
                res[backend] = res[backend].to(gold.dtype)
            try:
                torch.testing.assert_close(res[backend], gold)
                print(
                    f"Accuracy check \033[92m✓ succeed\033[0m for {backend} backend on {self.name} kernel"
                )
            except Exception as e:
                print(
                    f"Accuracy check \033[91m✗ failed\033[0m for {backend} backend on {self.name} kernel. Error {e}"
                )


class CrossEntropyBackward(BenchmarkKernel):
    # [CLASS DOC] Benchmark of the cross entropy backward pass: the gradient with respect to
    # logits given a float32 per-row output gradient. Forward is run at build time; only the
    # gradient computation is timed. Backends: eager, compiled, quack, liger.
    def __init__(self, script_args):
        # [FUNCTION DOC] Initialize the base benchmark and set the backend list to eager,
        # compiled, quack and liger.
        super().__init__(script_args)
        self.available_backends = ["eager", "compiled", "quack", "liger"]

    def get_shapes(self) -> tuple[tuple[int, ...], ...]:
        # [FUNCTION DOC] Return the fixed tuple of (rows, classes) shapes: 32768 rows with
        # 256 up to 65536 classes, then (16384, 131072) and (8192, 262144).
        return (
            (32768, 256),
            (32768, 512),
            (32768, 1024),
            (32768, 2048),
            (32768, 4096),
            (32768, 8192),
            (32768, 16384),
            (32768, 32768),
            (32768, 65536),
            (16384, 131072),
            (8192, 262144),
        )

    def get_memory_bytes(self, args, kwargs) -> int:
        # [FUNCTION DOC] Estimate memory traffic in bytes from (logits, target, loss gradient):
        # logits read plus gradient written, plus target and loss-gradient reads.
        # KNOWN ISSUE: contains a stray truncated comment line ("Memory ba") with no meaning.
        # Read x (M*N elements) + read target (M elements) + read dloss (M elements) + write grad(M*N elements)
        x, target, dloss = args
        # Memory ba
        M, N = x.shape
        return (
            2 * M * N * x.dtype.itemsize
            + M * target.dtype.itemsize
            + M * dloss.dtype.itemsize
        )

    def eager(self, args, kwargs=None) -> Any:
        # [FUNCTION DOC] Build the eager backend: runs the forward loss now and returns a
        # callable that computes the gradient wrt logits, retaining the graph so it can be
        # called repeatedly. Raises AssertionError if kwargs is not None.
        if kwargs is not None:
            raise AssertionError(f"Expected kwargs to be None, but got {kwargs}")
        x, target, dloss = args
        loss = F.cross_entropy(x, target, reduction="none")
        return lambda: torch.autograd.grad(
            loss, x, grad_outputs=dloss, retain_graph=True
        )

    def compiled(self, args, kwargs=None) -> Any:
        # [FUNCTION DOC] Build the compiled backend: compiles the loss in full-graph mode, runs
        # the forward now, and returns a callable computing the logits gradient with the graph
        # retained. Raises AssertionError if kwargs is not None.
        if kwargs is not None:
            raise AssertionError(f"Expected kwargs to be None, but got {kwargs}")
        x, target, dloss = args

        compiled_cross_entropy = torch.compile(
            lambda x, target: F.cross_entropy(x, target, reduction="none"),
            mode=self.compile_mode,
            fullgraph=True,
        )
        loss = compiled_cross_entropy(x, target)
        return lambda: torch.autograd.grad(
            loss, x, grad_outputs=dloss, retain_graph=True
        )

    def quack(self, args, kwargs=None) -> Any:
        # [FUNCTION DOC] Build the quack backend: lazily imports the quack cross entropy,
        # runs forward now and returns a callable computing the logits gradient (graph
        # retained). Raises AssertionError if kwargs is not None.
        from quack.cross_entropy import cross_entropy

        if kwargs is not None:
            raise AssertionError(f"Expected kwargs to be None, but got {kwargs}")
        x, target, dloss = args
        loss = cross_entropy(x, target)
        return lambda: torch.autograd.grad(
            loss, x, grad_outputs=dloss, retain_graph=True
        )

    def liger(self, args, kwargs=None) -> Any:
        # [FUNCTION DOC] Build the liger backend: constructs the Liger cross entropy loss, runs
        # forward now and returns a callable computing the logits gradient (graph retained).
        # Raises AssertionError if kwargs is not None.
        if kwargs is not None:
            raise AssertionError(f"Expected kwargs to be None, but got {kwargs}")
        from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss

        x, target, dloss = args
        cross_entropy = LigerCrossEntropyLoss(reduction="none")
        loss = cross_entropy(x, target)
        return lambda: torch.autograd.grad(
            loss, x, grad_outputs=dloss, retain_graph=True
        )

    def benchmark(self):
        # [FUNCTION DOC] For every shape, allocate bfloat16 logits requiring grad, int64
        # targets and a float32 loss gradient on the GPU, and benchmark them via the base class.
        for M, N in self.get_shapes():
            print(f"Tensor dimensions: [{M}, {N}]")
            torch_dtype = cutlass_torch.dtype(cutlass.BFloat16)
            x = 0.1 * torch.randn(
                M, N, device="cuda", dtype=torch_dtype, requires_grad=True
            )
            target = torch.randint(0, N, (M,), device="cuda", dtype=torch.int64)
            dloss = torch.randn(M, device="cuda", dtype=torch.float32)
            self.benchmark_single_shape(
                (x, target, dloss), setting=f"shape: [{M}, {N}]"
            )


class SoftmaxForward(BenchmarkKernel):
    # [CLASS DOC] Benchmark of the softmax forward pass over the last dimension on bfloat16
    # input. Backends: eager, compiled, quack, liger.
    def __init__(self, script_args):
        # [FUNCTION DOC] Initialize the base benchmark and set the backend list to eager,
        # compiled, quack and liger.
        super().__init__(script_args)
        self.available_backends = ["eager", "compiled", "quack", "liger"]

    def get_shapes(self) -> tuple[tuple[int, ...], ...]:
        # [FUNCTION DOC] Return the fixed tuple of (rows, columns) shapes: 32768 rows with
        # 256 up to 65536 columns, then (16384, 131072) and (8192, 262144).
        return (
            (32768, 256),
            (32768, 512),
            (32768, 1024),
            (32768, 2048),
            (32768, 4096),
            (32768, 8192),
            (32768, 16384),
            (32768, 32768),
            (32768, 65536),
            (16384, 131072),
            (8192, 262144),
        )

    def get_memory_bytes(self, args, kwargs) -> int:
        # [FUNCTION DOC] Estimate memory traffic in bytes for the single input: one read and
        # one write of the full matrix.
        (x,) = args
        M, N = x.shape
        return 2 * M * N * x.dtype.itemsize

    def eager(self, args, kwargs=None) -> Any:
        # [FUNCTION DOC] Build the eager backend: a callable applying softmax over the last
        # dimension. Raises AssertionError if kwargs is not None.
        if kwargs is not None:
            raise AssertionError(f"Expected kwargs to be None, but got {kwargs}")
        (x,) = args
        return lambda: F.softmax(x, dim=-1)

    def compiled(self, args, kwargs=None) -> Any:
        # [FUNCTION DOC] Build the compiled backend: marks the batch dimension dynamic, compiles
        # softmax in full-graph mode and returns a callable running it. Raises AssertionError
        # if kwargs is not None.
        if kwargs is not None:
            raise AssertionError(f"Expected kwargs to be None, but got {kwargs}")
        (x,) = args

        # Mark batch size as dynamic for realistic workload
        torch._dynamo.mark_dynamic(x, 0)

        compiled_softmax = torch.compile(
            lambda x: F.softmax(x, dim=-1), mode=self.compile_mode, fullgraph=True
        )
        return lambda: compiled_softmax(x)

    def quack(self, args, kwargs=None) -> Any:
        # [FUNCTION DOC] Build the quack backend: lazily imports the quack softmax and returns
        # a callable applying it. Raises AssertionError if kwargs is not None.
        from quack.softmax import softmax

        if kwargs is not None:
            raise AssertionError(f"Expected kwargs to be None, but got {kwargs}")
        (x,) = args
        return lambda: softmax(x)

    def liger(self, args, kwargs=None) -> Any:
        # [FUNCTION DOC] Build the liger backend: lazily imports the Liger softmax module, moves
        # it to the GPU and returns a callable applying it. Raises AssertionError if kwargs
        # is not None.
        from liger_kernel.transformers.softmax import LigerSoftmax

        if kwargs is not None:
            raise AssertionError(f"Expected kwargs to be None, but got {kwargs}")
        (x,) = args
        softmax = LigerSoftmax().to("cuda")
        return lambda: softmax(x)

    def benchmark(self):
        # [FUNCTION DOC] For every shape, allocate scaled random bfloat16 input on the GPU and
        # benchmark it via the base class helper.
        for M, N in self.get_shapes():
            print(f"Tensor dimensions: [{M}, {N}]")
            torch_dtype = cutlass_torch.dtype(cutlass.BFloat16)
            x = 0.1 * torch.randn(M, N, device="cuda", dtype=torch_dtype)
            self.benchmark_single_shape((x,), setting=f"shape: [{M}, {N}]")


class SoftmaxBackward(BenchmarkKernel):
    # [CLASS DOC] Benchmark of the softmax backward pass: gradient wrt the input given an output
    # gradient; forward is run at build time and only the gradient call is timed. Backends:
    # eager, compiled, quack, liger.
    def __init__(self, script_args):
        # [FUNCTION DOC] Initialize the base benchmark and set the backend list to eager,
        # compiled, quack and liger.
        super().__init__(script_args)
        self.available_backends = ["eager", "compiled", "quack", "liger"]

    def get_shapes(self) -> tuple[tuple[int, ...], ...]:
        # [FUNCTION DOC] Return the fixed tuple of (rows, columns) shapes: 32768 rows with
        # 256 up to 65536 columns, then (16384, 131072) and (8192, 262144).
        return (
            (32768, 256),
            (32768, 512),
            (32768, 1024),
            (32768, 2048),
            (32768, 4096),
            (32768, 8192),
            (32768, 16384),
            (32768, 32768),
            (32768, 65536),
            (16384, 131072),
            (8192, 262144),
        )

    def get_memory_bytes(self, args, kwargs) -> int:
        # [FUNCTION DOC] Estimate memory traffic in bytes from (input, output gradient): three
        # full-matrix transfers (reads of the saved output and gradient, one write).
        # Memory: read dy and y, write ax backward
        x, dy = args
        M, N = x.shape
        return 3 * M * N * x.dtype.itemsize

    def eager(self, args, kwargs=None) -> Any:
        # [FUNCTION DOC] Build the eager backend: runs softmax forward now and returns a
        # callable computing the input gradient with the graph retained. Raises AssertionError
        # if kwargs is not None.
        if kwargs is not None:
            raise AssertionError(f"Expected kwargs to be None, but got {kwargs}")
        x, dy = args
        y = F.softmax(x, dim=-1)
        return lambda: torch.autograd.grad(y, x, grad_outputs=dy, retain_graph=True)

    def compiled(self, args, kwargs=None) -> Any:
        # [FUNCTION DOC] Build the compiled backend: compiles softmax in full-graph mode, runs
        # forward now and returns a callable computing the input gradient (graph retained).
        # Raises AssertionError if kwargs is not None.
        if kwargs is not None:
            raise AssertionError(f"Expected kwargs to be None, but got {kwargs}")
        x, dy = args
        compiled_softmax = torch.compile(
            lambda x: F.softmax(x, dim=-1), mode=self.compile_mode, fullgraph=True
        )
        y = compiled_softmax(x)
        return lambda: torch.autograd.grad(y, x, grad_outputs=dy, retain_graph=True)

    def quack(self, args, kwargs=None) -> Any:
        # [FUNCTION DOC] Build the quack backend: lazily imports quack softmax, runs forward now
        # and returns a callable computing the input gradient (graph retained). Raises
        # AssertionError if kwargs is not None.
        from quack.softmax import softmax

        if kwargs is not None:
            raise AssertionError(f"Expected kwargs to be None, but got {kwargs}")
        x, dy = args

        y = softmax(x)
        return lambda: torch.autograd.grad(y, x, grad_outputs=dy, retain_graph=True)

    def liger(self, args, kwargs=None) -> Any:
        # [FUNCTION DOC] Build the liger backend: lazily imports the Liger softmax, runs forward
        # now and returns a callable computing the input gradient (graph retained). Raises
        # AssertionError if kwargs is not None.
        from liger_kernel.transformers.softmax import LigerSoftmax

        if kwargs is not None:
            raise AssertionError(f"Expected kwargs to be None, but got {kwargs}")
        x, dy = args
        softmax = LigerSoftmax().to("cuda")
        y = softmax(x)
        return lambda: torch.autograd.grad(y, x, grad_outputs=dy, retain_graph=True)

    def benchmark(self):
        # [FUNCTION DOC] For every shape, allocate bfloat16 input requiring grad and a bfloat16
        # output gradient on the GPU, and benchmark them via the base class helper.
        for M, N in self.get_shapes():
            print(f"Tensor dimensions: [{M}, {N}]")
            torch_dtype = cutlass_torch.dtype(cutlass.BFloat16)
            x = 0.1 * torch.randn(
                M, N, device="cuda", dtype=torch_dtype, requires_grad=True
            )
            dy = torch.randn(M, N, device="cuda", dtype=torch_dtype)
            self.benchmark_single_shape((x, dy), setting=f"shape: [{M}, {N}]")


class RMSNormForward(BenchmarkKernel):
    # [CLASS DOC] Benchmark of the RMSNorm forward pass (epsilon 1e-6) on bfloat16 input with a
    # float32 weight vector. Backends: eager, compiled, quack, liger.
    def __init__(self, script_args):
        # [FUNCTION DOC] Initialize the base benchmark and set the backend list to eager,
        # compiled, quack and liger.
        super().__init__(script_args)
        self.available_backends = ["eager", "compiled", "quack", "liger"]

    def get_shapes(self) -> tuple[tuple[int, ...], ...]:
        # [FUNCTION DOC] Return the softmax-style shape tuple (32768 rows with 256 to 65536
        # columns, then two very wide ones) with the extra norm shapes appended.
        return (
            (32768, 256),
            (32768, 512),
            (32768, 1024),
            (32768, 2048),
            (32768, 4096),
            (32768, 8192),
            (32768, 16384),
            (32768, 32768),
            (32768, 65536),
            (16384, 131072),
            (8192, 262144),
        ) + extra_shapes_for_norm

    def get_memory_bytes(self, args, kwargs) -> int:
        # [FUNCTION DOC] Estimate memory traffic in bytes from (input, weight): read and write of
        # the matrix plus one read of the weight vector.
        x, w = args
        M, N = x.shape
        return 2 * M * N * x.dtype.itemsize + N * w.dtype.itemsize

    def rms_norm_ref(self, x, w):
        # [FUNCTION DOC] Reference implementation: computes in float32 the input scaled by the
        # reciprocal root mean square over the last dimension (epsilon 1e-6) and by the weight,
        # then casts back to the input dtype.
        x_f32 = x.float()
        return (
            x_f32
            * torch.rsqrt(torch.mean(x_f32.square(), dim=-1, keepdim=True) + 1e-6)
            * w
        ).to(x.dtype)

    def eager(self, args, kwargs=None) -> Any:
        # [FUNCTION DOC] Build the eager backend: a callable running the reference
        # implementation. Raises AssertionError if kwargs is not None.
        if kwargs is not None:
            raise AssertionError(f"Expected kwargs to be None, but got {kwargs}")
        x, w = args
        return lambda: self.rms_norm_ref(x, w)

    def compiled(self, args, kwargs=None) -> Any:
        # [FUNCTION DOC] Build the compiled backend: marks the batch dimension dynamic, compiles
        # the reference in full-graph mode and returns a callable running it. Raises
        # AssertionError if kwargs is not None.
        if kwargs is not None:
            raise AssertionError(f"Expected kwargs to be None, but got {kwargs}")
        x, w = args

        # Mark batch size as dynamic for realistic workload
        torch._dynamo.mark_dynamic(x, 0)

        compiled_rms_norm = torch.compile(
            self.rms_norm_ref, mode=self.compile_mode, fullgraph=True
        )
        return lambda: compiled_rms_norm(x, w)

    def quack(self, args, kwargs=None) -> Any:
        # [FUNCTION DOC] Build the quack backend: lazily imports the quack forward kernel,
        # preallocates an output buffer and returns a callable that fills and returns it.
        # kwargs is not checked. Only a float32 weight is supported.
        # Note: only supper weight with float32 dtype
        from quack.rmsnorm import _rmsnorm_fwd

        x, w = args
        y = torch.empty_like(x)

        def quack_fwd():
            # [FUNCTION DOC] HELPER: Run the quack forward kernel into the preallocated output
            # (no bias, residual or saved reciprocal std) and return that output buffer.
            _rmsnorm_fwd(
                x,
                w,
                out=y,
                bias=None,
                rstd=None,
                residual=None,
                residual_out=None,
                eps=1e-6,
            )
            return y

        return quack_fwd

    def liger(self, args, kwargs) -> Any:
        # [FUNCTION DOC] Build the liger backend: constructs a Liger RMSNorm on the GPU, copies
        # the given weight into it and returns a callable applying it. kwargs is unused.
        from liger_kernel.transformers.rms_norm import LigerRMSNorm

        x, w = args
        M, N = x.shape
        liger_rmsnorm = LigerRMSNorm(hidden_size=N, eps=1e-6).cuda()
        liger_rmsnorm.weight.data.copy_(w)
        return lambda: liger_rmsnorm(x)

    def benchmark(self):
        # [FUNCTION DOC] For every shape, allocate random bfloat16 input and float32 weight on
        # the GPU and benchmark them via the base class helper.
        for M, N in self.get_shapes():
            print(f"Tensor dimensions: [{M}, {N}]")
            torch_dtype = cutlass_torch.dtype(cutlass.BFloat16)
            x = torch.randn(M, N, device="cuda", dtype=torch_dtype)
            w = torch.randn(N, device="cuda", dtype=torch.float32)
            self.benchmark_single_shape((x, w), setting=f"shape: [{M}, {N}]")


class RMSNormBackward(BenchmarkKernel):
    # [CLASS DOC] Benchmark of the RMSNorm backward pass: gradients wrt input and weight given
    # an output gradient; forward is run at build time where applicable. Backends: eager,
    # compiled, quack, liger.
    def __init__(self, script_args):
        # [FUNCTION DOC] Initialize the base benchmark and set the backend list to eager,
        # compiled, quack and liger.
        super().__init__(script_args)
        self.available_backends = [
            "eager",
            "compiled",
            "quack",
            "liger",
        ]

    def get_shapes(self) -> tuple[tuple[int, ...], ...]:
        # [FUNCTION DOC] Return shapes: 32768 rows with 256 to 16384 columns, plus the extra norm
        # shapes. Larger shapes are omitted because the widest one ran out of memory on H100.
        # TODO: OOM for (32768, 65536) on h100
        return (
            (32768, 256),
            (32768, 512),
            (32768, 1024),
            (32768, 2048),
            (32768, 4096),
            (32768, 8192),
            (32768, 16384),
        ) + extra_shapes_for_norm

    def get_memory_bytes(self, args, kwargs) -> int:
        # [FUNCTION DOC] Estimate memory traffic in bytes from (input, weight, output gradient):
        # three full-matrix transfers plus two weight-sized transfers.
        x, w, dy = args
        # x, dy: [M, N], w: [N]
        M, N = x.shape
        # Read x, w, dy, write dx, dw
        return 3 * M * N * x.dtype.itemsize + 2 * N * w.dtype.itemsize

    def rms_norm_ref(self, x, w):
        # [FUNCTION DOC] Reference implementation: computes in float32 the input scaled by the
        # reciprocal root mean square over the last dimension (epsilon 1e-6) and by the weight,
        # then casts back to the input dtype.
        x_f32 = x.float()
        return (
            x_f32
            * torch.rsqrt(torch.mean(x_f32.square(), dim=-1, keepdim=True) + 1e-6)
            * w
        ).to(x.dtype)

    def eager(self, args, kwargs=None) -> Any:
        # [FUNCTION DOC] Build the eager backend: runs the reference forward now and returns a
        # callable computing gradients wrt input and weight (graph retained). Raises
        # AssertionError if kwargs is not None.
        if kwargs is not None:
            raise AssertionError(f"Expected kwargs to be None, but got {kwargs}")
        x, w, dy = args
        y = self.rms_norm_ref(x, w)
        return lambda: torch.autograd.grad(
            y, [x, w], grad_outputs=dy, retain_graph=True
        )

    def compiled(self, args, kwargs=None) -> Any:
        # [FUNCTION DOC] Build the compiled backend: compiles the reference in full-graph mode,
        # runs forward now and returns a callable computing gradients wrt input and weight
        # (graph retained). Raises AssertionError if kwargs is not None.
        if kwargs is not None:
            raise AssertionError(f"Expected kwargs to be None, but got {kwargs}")
        x, w, dy = args
        y = torch.compile(self.rms_norm_ref, mode=self.compile_mode, fullgraph=True)(
            x, w
        )
        return lambda: torch.autograd.grad(
            y, [x, w], grad_outputs=dy, retain_graph=True
        )

    def compute_rstd(self, x, eps):
        # [FUNCTION DOC] Compute the float32 reciprocal root mean square of the input over the
        # last dimension (kept as a dimension) with the given epsilon.
        return torch.rsqrt(torch.mean(x.float().square(), dim=-1, keepdim=True) + eps)

    def quack(self, args, kwargs=None) -> Any:
        # [FUNCTION DOC] Build the quack backend: lazily imports the quack backward kernel and
        # SM count helper, precomputes the reciprocal std and allocates the input gradient and a
        # per-SM partial weight gradient buffer. Returns a callable running the kernel and
        # returning the input gradient plus the partial weight gradients summed and cast to the
        # weight dtype. Raises AssertionError if kwargs is not None.
        if kwargs is not None:
            raise AssertionError(f"Expected kwargs to be None, but got {kwargs}")
        from quack.rmsnorm import _get_sm_count, _rmsnorm_bwd

        (
            x,
            w,
            dy,
        ) = args
        M, N = x.shape

        rstd = self.compute_rstd(x, eps=1e-6)
        dx = torch.empty_like(x)
        sm_count = _get_sm_count(x.size(1), x.device)
        dw_partial = torch.empty(
            sm_count, x.size(1), device=x.device, dtype=torch.float32
        )

        def quack_bwd():
            # [FUNCTION DOC] HELPER: Run the quack backward kernel (no bias or residual terms),
            # reduce the partial weight gradients over SMs, and return the gradient pair.
            _rmsnorm_bwd(
                x,
                w,
                dy,
                rstd,
                dx,
                dw_partial,
                db_partial=None,
                dresidual_out=None,
                dresidual=None,
                sm_count=sm_count,
            )
            dw = dw_partial.sum(dim=0).to(w.dtype)
            return dx, dw

        return quack_bwd

    def liger(self, args, kwargs=None) -> Any:
        # [FUNCTION DOC] Build the liger backend: constructs a Liger RMSNorm with gemma casting
        # on the GPU, copies in the weight, runs forward now and returns a callable computing
        # gradients wrt the input and the Liger weight (graph retained). Raises AssertionError
        # if kwargs is not None.
        if kwargs is not None:
            raise AssertionError(f"Expected kwargs to be None, but got {kwargs}")
        from liger_kernel.transformers.rms_norm import LigerRMSNorm

        x, w, dy = args
        M, N = x.shape
        liger_rmsnorm = LigerRMSNorm(
            hidden_size=N, eps=1e-6, casting_mode="gemma"
        ).cuda()
        liger_rmsnorm.weight.data.copy_(w)
        y = liger_rmsnorm(x)
        return lambda: torch.autograd.grad(
            y, [x, liger_rmsnorm.weight], grad_outputs=dy, retain_graph=True
        )

    def benchmark(self):
        # [FUNCTION DOC] For every shape, allocate bfloat16 input and float32 weight both
        # requiring grad, plus a bfloat16 output gradient, on the GPU and benchmark them.
        for M, N in self.get_shapes():
            print(f"Tensor dimensions: [{M}, {N}]")
            torch_dtype = cutlass_torch.dtype(cutlass.BFloat16)
            x = torch.randn(M, N, device="cuda", dtype=torch_dtype, requires_grad=True)
            w = torch.randn(N, device="cuda", dtype=torch.float32, requires_grad=True)
            dy = torch.randn(M, N, device="cuda", dtype=torch_dtype)
            self.benchmark_single_shape((x, w, dy), setting=f"shape: [{M}, {N}]")


class LayerNormForward(BenchmarkKernel):
    # [CLASS DOC] Benchmark of the LayerNorm forward pass (epsilon 1e-6, weight but no bias) on
    # bfloat16 input with a float32 weight. Backends: eager, compiled, quack, liger.
    def __init__(self, script_args):
        # [FUNCTION DOC] Initialize the base benchmark and set the backend list to eager,
        # compiled, quack and liger.
        super().__init__(script_args)
        self.available_backends = ["eager", "compiled", "quack", "liger"]

    def get_shapes(self) -> tuple[tuple[int, ...], ...]:
        # [FUNCTION DOC] Return shapes: 32768 rows with 256 to 65536 columns plus the extra norm
        # shapes. The widest softmax shapes are excluded because they ran out of memory on H100.
        # OOM for (16384, 131072) on h100
        return (
            (32768, 256),
            (32768, 512),
            (32768, 1024),
            (32768, 2048),
            (32768, 4096),
            (32768, 8192),
            (32768, 16384),
            (32768, 32768),
            (32768, 65536),
        ) + extra_shapes_for_norm

    def get_memory_bytes(self, args, kwargs) -> int:
        # [FUNCTION DOC] Estimate memory traffic in bytes from (input, weight): read and write of
        # the matrix plus one weight read.
        x, w = args
        M, N = x.shape
        # Read x ([M, N]), w ([N]), write y ([M, N])
        return 2 * M * N * x.dtype.itemsize + N * w.dtype.itemsize

    def layernorm_ref(self, x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6):
        # [FUNCTION DOC] Reference implementation: upcasts the input to float32, applies layer
        # normalization over the weight shape using the weight and no bias, and casts back to the
        # input dtype. The epsilon argument defaults to 1e-6.
        x_f32 = x.float()
        return F.layer_norm(x_f32, w.shape, w, None, eps).to(x.dtype)

    def eager(self, args, kwargs=None) -> Any:
        # [FUNCTION DOC] Build the eager backend: a callable running the reference. Raises
        # AssertionError if kwargs is not None.
        if kwargs is not None:
            raise AssertionError(f"Expected kwargs to be None, but got {kwargs}")
        x, w = args
        return lambda: self.layernorm_ref(x, w)

    def compiled(self, args, kwargs=None) -> Any:
        # [FUNCTION DOC] Build the compiled backend: marks the batch dimension dynamic, compiles
        # the reference in full-graph mode and returns a callable running it with epsilon 1e-6.
        # Raises AssertionError if kwargs is not None.
        if kwargs is not None:
            raise AssertionError(f"Expected kwargs to be None, but got {kwargs}")
        x, w = args

        # Mark batch size as dynamic for realistic workload
        torch._dynamo.mark_dynamic(x, 0)

        compiled_layernorm = torch.compile(
            self.layernorm_ref, mode=self.compile_mode, fullgraph=True
        )
        return lambda: compiled_layernorm(x, w, eps=1e-6)

    def quack(self, args, kwargs) -> Any:
        # [FUNCTION DOC] Build the quack backend: lazily imports the quack layernorm (which has
        # no bias support) and returns a callable applying it with epsilon 1e-6. kwargs unused.
        # Note: quack layernorm does not support bias
        from quack.layernorm import layernorm

        x, w = args
        return lambda: layernorm(x, w, eps=1e-6)

    def liger(self, args, kwargs) -> Any:
        # [FUNCTION DOC] Build the liger backend: constructs a Liger LayerNorm on the GPU, copies
        # in the weight, zeroes the bias, and returns a callable applying it. kwargs is unused.
        from liger_kernel.transformers.layer_norm import LigerLayerNorm

        x, w = args
        M, N = x.shape
        liger_layernorm = LigerLayerNorm(hidden_size=N, eps=1e-6).cuda()
        liger_layernorm.weight.data.copy_(w)
        liger_layernorm.bias.data.copy_(
            torch.zeros(N, device="cuda", dtype=torch.float32)
        )
        return lambda: liger_layernorm(x)

    def benchmark(self):
        # [FUNCTION DOC] For every shape, allocate random bfloat16 input and float32 weight on
        # the GPU and benchmark them via the base class helper.
        for M, N in self.get_shapes():
            print(f"Tensor dimensions: [{M}, {N}]")
            torch_dtype = cutlass_torch.dtype(cutlass.BFloat16)
            x = torch.randn(M, N, device="cuda", dtype=torch_dtype)
            w = torch.randn(N, device="cuda", dtype=torch.float32)
            self.benchmark_single_shape((x, w), setting=f"shape: [{M}, {N}]")


class LayerNormBackward(BenchmarkKernel):
    # [CLASS DOC] Benchmark of the LayerNorm backward pass: gradients wrt input and weight given
    # an output gradient. Backends: eager, compiled, liger (no quack variant).
    def __init__(self, script_args):
        # [FUNCTION DOC] Initialize the base benchmark and set the backend list to eager,
        # compiled and liger.
        super().__init__(script_args)
        self.available_backends = ["eager", "compiled", "liger"]

    def get_shapes(self) -> tuple[tuple[int, ...], ...]:
        # [FUNCTION DOC] Return shapes: 32768 rows with 256 to 65536 columns plus the extra norm
        # shapes. Wider shapes are excluded because they ran out of memory.
        # OOM for (16384, 131072), (8192, 262144)
        return (
            (32768, 256),
            (32768, 512),
            (32768, 1024),
            (32768, 2048),
            (32768, 4096),
            (32768, 8192),
            (32768, 16384),
            (32768, 32768),
            (32768, 65536),
        ) + extra_shapes_for_norm

    def get_memory_bytes(self, args, kwargs) -> int:
        # [FUNCTION DOC] Estimate memory traffic in bytes from (input, weight, output gradient):
        # two full-matrix transfers at the input size, two weight-sized transfers, and one
        # full-matrix transfer at the output gradient size.
        x, w, dy = args
        M, N = x.shape
        # Read x ([M, N]), w ([N]), dy ([M, N]), write dx ([M, N]), dw ([N])
        return (
            2 * M * N * x.dtype.itemsize
            + 2 * N * w.dtype.itemsize
            + M * N * dy.dtype.itemsize
        )

    def layernorm_ref(self, x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6):
        # [FUNCTION DOC] Reference implementation: upcasts the input to float32, applies layer
        # normalization using the weight and no bias, and casts back to the input dtype.
        x_f32 = x.float()
        return F.layer_norm(x_f32, w.shape, w, None, eps).to(x.dtype)

    def eager(self, args, kwargs=None) -> Any:
        # [FUNCTION DOC] Build the eager backend: runs the reference forward now and returns a
        # callable computing gradients wrt input and weight (graph retained). Raises
        # AssertionError if kwargs is not None.
        if kwargs is not None:
            raise AssertionError(f"Expected kwargs to be None, but got {kwargs}")
        x, w, dy = args
        y = self.layernorm_ref(x, w)
        return lambda: torch.autograd.grad(
            y, [x, w], grad_outputs=dy, retain_graph=True
        )

    def compiled(self, args, kwargs=None) -> Any:
        # [FUNCTION DOC] Build the compiled backend: compiles the reference in full-graph mode,
        # runs forward now and returns a callable computing gradients wrt input and weight
        # (graph retained). Raises AssertionError if kwargs is not None.
        if kwargs is not None:
            raise AssertionError(f"Expected kwargs to be None, but got {kwargs}")
        x, w, dy = args
        compiled_layernorm = torch.compile(
            self.layernorm_ref, mode=self.compile_mode, fullgraph=True
        )
        y = compiled_layernorm(x, w)
        return lambda: torch.autograd.grad(
            y, [x, w], grad_outputs=dy, retain_graph=True
        )

    def compute_mean_rstd(self, x, eps):
        # [FUNCTION DOC] Compute float32 per-row mean and reciprocal standard deviation over the
        # last dimension, using the biased variance and the given epsilon. Returns both.
        x = x.float()

        var, mean = torch.var_mean(x, dim=-1, keepdim=True, correction=0)
        rstd = torch.rsqrt(var + eps)
        return mean, rstd

    def liger(self, args, kwargs) -> Any:
        # [FUNCTION DOC] Build the liger backend: lazily imports the Liger layer norm backward
        # function, precomputes float32 mean and reciprocal std, and returns a callable calling
        # that backward directly and keeping only its first two outputs (input and weight
        # gradients). kwargs is unused.
        """
        Call layer_norm_backward directly rather than calling
        liger_kernel.transformers.layer_norm.LigerLayerNorm and
        torch.autograd.grad.

        The latter fashion saves mean/rstd in x.dtype which can fail
        accuracy test. We call layer_norm_backward with fp32 mean and
        rstd.
        """
        from liger_kernel.ops.layer_norm import layer_norm_backward

        x, w, dy = args
        eps = 1e-6
        mean, rstd = self.compute_mean_rstd(x, eps)
        M, N = x.shape

        return lambda: layer_norm_backward(dy, x, w, None, mean, rstd)[0:2]

    def benchmark(self):
        # [FUNCTION DOC] For every shape, allocate bfloat16 input and float32 weight both
        # requiring grad, plus a bfloat16 output gradient, on the GPU and benchmark them.
        for M, N in self.get_shapes():
            print(f"Tensor dimensions: [{M}, {N}]")
            torch_dtype = cutlass_torch.dtype(cutlass.BFloat16)
            x = torch.randn(M, N, device="cuda", dtype=torch_dtype, requires_grad=True)
            w = torch.randn(N, device="cuda", dtype=torch.float32, requires_grad=True)
            dy = torch.randn(M, N, device="cuda", dtype=torch_dtype)
            self.benchmark_single_shape((x, w, dy), setting=f"shape: [{M}, {N}]")
