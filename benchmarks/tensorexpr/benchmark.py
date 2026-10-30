# [MODULE DOC] Core of the tensor expression micro-benchmark harness. Provides the Benchmark
# base class (setup, warmup and timed loop, optional tracing with a fuser, optional backward,
# bandwidth and throughput reporting), a context manager for cuda pointwise fuser tuning, the
# DynamicShape mixin that rotates through randomly shaped inputs, and the global registry of
# benchmark classes with its registration function. Subclasses supply the computation,
# configuration and workload size.
import contextlib
import json
import os
import time

import numpy as np

import torch

from . import tensor_engine


class Benchmark:
    # [CLASS DOC] Base class for harness benchmarks. A subclass must set inputs (a list of
    # tensors) and implement forward, config, module, memory_workload and default_configs, and
    # may provide reference for result checking and compute_workload. State: mode (fwd or both),
    # device, dtype, requires_grad, the engine, the tensors needing gradients, the gradient seed
    # and output options. Public engine methods are forwarded onto the instance. The run and
    # run_impl methods warm up, time and report; the jit_mode attribute must be set by the
    # caller before running.
    def __init__(self, mode, device, dtype):
        # [FUNCTION DOC] Stores mode, device and dtype and default options (non-deterministic,
        # stdout output, no IR or kernel printing). Mode fwd means no gradients, mode both means
        # gradients are required, any other value raises a value error. Fetches the global
        # tensor engine (an error if none is set), resets it for the device, and copies every
        # public callable of the engine onto this instance unless the instance already has an
        # attribute of that name.
        self.mode = mode
        self.deterministic = False
        self.device = device
        self.dtype = dtype
        self.output_type = "stdout"
        self.print_ir = False
        self.print_kernel = False
        if mode == "both":
            self.requires_grad = True
        elif mode == "fwd":
            self.requires_grad = False
        else:
            raise ValueError(f"invalid mode: {mode}")
        self.result_grad = None
        self.grad_variables = []
        self.engine = tensor_engine.get_engine()
        self.engine.reset(device)

        # forward all member functions in self.engine to self
        for method in dir(self.engine):
            if not callable(getattr(self.engine, method)):
                continue
            # don't forward if this function is overridden here
            if hasattr(self, method):
                continue
            # don't forward if it is a internal function
            if method.startswith("_"):
                continue
            method_engine = getattr(self.engine, method)
            setattr(self, method, method_engine)

    def forward(self):
        # [FUNCTION DOC] Abstract: performs one step of computation. Subclasses must override;
        # the base version raises a value error.
        """do one step worth of computation"""
        raise ValueError("this method should be reimplemented by subclass")

    def check(self):
        # [FUNCTION DOC] Verifies correctness only when the benchmark is flagged
        # deterministic: compares the reference result with the computed result converted to
        # numpy, with an absolute tolerance of 0.01, raising an assertion error on mismatch.
        # Does nothing otherwise.
        if not self.deterministic:
            return
        np.testing.assert_allclose(
            self.reference(), self.numpy(self.compute()), atol=1e-2
        )

    def config(self):
        # [FUNCTION DOC] Abstract: returns the list of configuration values of this instance.
        # Subclasses must override; the base version raises a value error.
        """returns an array for the current benchmark configs"""
        raise ValueError("this method should be reimplemented by subclass")

    def desc(self):
        # [FUNCTION DOC] Builds the description string: engine mode, module name, run mode,
        # device and the configuration values joined by underscores. When the thread-count
        # environment variable is set, its value is appended to the device text.
        """return the description of the current benchmark"""
        config = self.config()
        config_str = "_".join([str(x) for x in config])
        device = self.device
        if "NNC_NUM_THREADS" in os.environ:
            num_threads_str = os.environ["NNC_NUM_THREADS"]
            device += num_threads_str
        return f"{self.engine.mode}: {self.module()}_{self.mode}_{device}_{config_str}"

    @staticmethod
    def module():
        # [FUNCTION DOC] Abstract: returns the benchmark module name. Subclasses must override;
        # the base version raises a value error.
        raise ValueError("this method should be reimplemented by subclass")

    def memory_workload(self):
        # [FUNCTION DOC] Abstract: returns a mapping with entries sol and algorithmic
        # giving data movement amounts. Subclasses must override; the base version raises a
        # value error.
        raise ValueError("this method should be reimplemented by subclass")

    def compute_workload(self):
        # [FUNCTION DOC] Optional hook: number of scalar operations per step. The base version
        # returns none, meaning no compute throughput is reported.
        """return the number of scalar operations it takes to finish the tensor op"""
        return None

    @staticmethod
    def input_iterable():
        # [FUNCTION DOC] Returns false; subclasses that accept a sweep of input dimensions
        # override it to return true.
        """A benchmark child class should return true if it utilizes the input iter arg"""
        return False

    def dtype_to_bytes(self):
        # [FUNCTION DOC] Returns the size in bytes of one element of this benchmark's dtype.
        return torch.tensor(0, dtype=self.dtype).element_size()

    @staticmethod
    def default_configs():
        # [FUNCTION DOC] Abstract: returns the list of default configs, each a list of
        # constructor arguments. Subclasses must override; the base version raises a
        # value error.
        """return a list of default configs for this benchmark"""
        raise ValueError("this method should be reimplemented by subclass")

    def is_supported(self):
        # [FUNCTION DOC] Returns true; subclasses override it to skip configurations they
        # cannot run.
        return True

    def rand(self, shape, device=None, dtype=None, requires_grad=False):
        # [FUNCTION DOC] Creates a random tensor through the engine and, if gradients are
        # required, records it in the list of gradient variables. Returns the tensor.
        v = self.engine.rand(
            shape, device=device, dtype=dtype, requires_grad=requires_grad
        )
        if requires_grad:
            self.grad_variables.append(v)
        return v

    def nchw_rand(self, shape, device=None, requires_grad=False):
        # [FUNCTION DOC] Same as rand but uses the engine channels-first random generator for an
        # image-like shape; records the tensor for gradients when required. Returns the tensor.
        v = self.engine.nchw_rand(shape, device=device, requires_grad=requires_grad)
        if requires_grad:
            self.grad_variables.append(v)
        return v

    def compute(self):
        # [FUNCTION DOC] Runs one step on the stored inputs: through the traced module if one
        # was created, otherwise through forward. Returns the result.
        if self.bm_jit:
            return self.bm_jit(*self.inputs)
        else:
            return self.forward(*self.inputs)

    def run(self, args):
        # [FUNCTION DOC] Runs the benchmark with the fuser configured from the parsed arguments,
        # then delegates to the timed run. Remembers the print-IR option. For the old fuser
        # option it allows gpu fusion (and sets a debug environment variable if kernel printing
        # is on); for the te option it enables the tensor expression fuser and applies the cuda
        # pointwise tuning values for the duration; for the nvf option it configures the
        # profiling executor, enables nvfuser, disables fusion elsewhere, sets a bailout depth
        # of 20 (and a debug environment variable if kernel printing is on); for any other value
        # it runs without a fuser. Mutates global jit and environment state. Returns the result
        # of the timed run.
        self.print_ir = args.print_ir
        if args.cuda_fuser == "old":
            torch._C._jit_override_can_fuse_on_gpu(True)
            if args.print_kernel:
                os.environ["PYTORCH_FUSION_DEBUG"] = "1"
            return self.run_impl(True)
        elif args.cuda_fuser == "te":
            torch._C._jit_set_texpr_fuser_enabled(True)
            with cuda_pointwise_context(
                args.cuda_pointwise_loop_levels,
                args.cuda_pointwise_block_count,
                args.cuda_pointwise_block_size,
            ):
                return self.run_impl(True)
        elif args.cuda_fuser == "nvf":
            torch._C._jit_set_nvfuser_enabled(True)
            torch._C._jit_set_profiling_executor(True)
            torch._C._jit_set_profiling_mode(True)
            torch._C._jit_override_can_fuse_on_cpu(False)
            torch._C._jit_override_can_fuse_on_gpu(False)
            torch._C._jit_set_bailout_depth(20)
            if args.print_kernel:
                os.environ["PYTORCH_CUDA_FUSER_DEBUG"] = "1"
            return self.run_impl(True)
        else:
            return self.run_impl(False)

    def run_impl(self, use_fuser):
        # [FUNCTION DOC] Warms up for 10 steps and then times 1000 steps on cuda or 10 steps on
        # other devices, synchronizing cuda before starting and stopping the clock. On the first
        # step optionally traces forward (when jit_mode is trace and a fuser is in use), and
        # checks the result against the reference if one exists, otherwise prints a warning; on
        # the second step optionally prints the fused graph. In mode both, also backpropagates a
        # random gradient (created once) through all recorded variables each step. Computes the
        # mean step time, bandwidth for the sol and algorithmic workloads (scaled by the dtype
        # size) and optional compute throughput, and reports them through dump_result. Requires
        # the jit_mode attribute to have been set, otherwise an attribute error is raised.
        warmups = 10
        if self.device == "cuda":
            iters = 1000
        else:
            iters = 10
        engine = tensor_engine.get_engine()

        self.bm_jit = None
        for i in range(warmups + iters):
            if i == warmups:
                if self.device == "cuda":
                    engine.sync_cuda()
                time_start = time.time()

            if i == 0:
                if self.jit_mode == "trace" and use_fuser:
                    self.bm_jit = torch.jit.trace(
                        self.forward, example_inputs=self.inputs, check_trace=False
                    )
                if callable(getattr(self, "reference", None)):
                    self.check()
                else:
                    print("Warning: no reference result for ", self.module())
            elif i == 1:
                # The fusion graph is visible after the first iter is executed
                if self.jit_mode == "trace" and use_fuser and self.print_ir:
                    print(self.bm_jit.graph_for(*self.inputs))
            z = self.compute()
            if self.mode == "both":
                if self.result_grad is None:
                    self.result_grad = engine.rand_like(z)
                engine.backward([z], [self.result_grad], self.grad_variables)

        if self.device == "cuda":
            engine.sync_cuda()

        duration = time.time() - time_start
        iter_time = duration / iters
        memory_workload = self.memory_workload()
        compute_workload = self.compute_workload()

        result_dict = {
            "desc": self.desc(),
            "us": iter_time * 1e6,
            "sol": memory_workload["sol"] * self.dtype_to_bytes() / iter_time / 1e9,
            "algorithmic": memory_workload["algorithmic"]
            * self.dtype_to_bytes()
            / iter_time
            / 1e9,
        }
        if compute_workload:
            result_dict["compute_workload"] = compute_workload / iter_time / 1e9
        self.dump_result(result_dict)

    def dump_result(self, result_dict):
        # [FUNCTION DOC] Emits the result mapping: as a JSON line when the output type is json;
        # as a human-readable line with microseconds, bandwidth for both workloads and optional
        # compute throughput when stdout; raises a generic exception for any other output type.
        if self.output_type == "json":
            print(json.dumps(result_dict))
        elif self.output_type == "stdout":
            msg = "{}: {:.2f} us, SOL {:.2f} GB/s, algorithmic {:.2f} GB/s".format(
                result_dict["desc"],
                result_dict["us"],
                result_dict["sol"],
                result_dict["algorithmic"],
            )
            if "compute_workload" in result_dict:
                msg += f", compute {result_dict['compute_workload']:.2f} Gops/s"
            print(msg)
        else:
            raise Exception("Unknown output_type " + self.output_type)  # noqa: TRY002


@contextlib.contextmanager
def cuda_pointwise_context(loop_levels, block_count, block_size):
    # [FUNCTION DOC] Context manager that temporarily overrides the tensor expression
    # cuda pointwise tuning values (loop levels, block count, block size). Only values
    # that are truthy are changed, and their previous values are restored on exit even if
    # the body raises.
    if loop_levels:
        old_loop_levels = torch._C._jit_get_te_cuda_pointwise_loop_levels()
        torch._C._jit_set_te_cuda_pointwise_loop_levels(loop_levels)
    if block_count:
        old_block_count = torch._C._jit_get_te_cuda_pointwise_block_count()
        torch._C._jit_set_te_cuda_pointwise_block_count(block_count)
    if block_size:
        old_block_size = torch._C._jit_get_te_cuda_pointwise_block_size()
        torch._C._jit_set_te_cuda_pointwise_block_size(block_size)

    try:
        yield
    finally:
        if loop_levels:
            torch._C._jit_set_te_cuda_pointwise_loop_levels(old_loop_levels)
        if block_count:
            torch._C._jit_set_te_cuda_pointwise_block_count(old_block_count)
        if block_size:
            torch._C._jit_set_te_cuda_pointwise_block_size(old_block_size)


# Auxiliary class to facilitate dynamic input shape
class DynamicShape:
    # [CLASS DOC] Mixin for benchmarks with dynamic shapes. It collects a fixed number of input
    # sets (the sample size constant) with randomly reduced shapes, and exposes the current set
    # as inputs, advancing to the next set each time compute is called. It relies on cooperating
    # base classes for run and compute (it calls the parent versions) and requires subclasses to
    # implement instantiate_input.
    r"""
    An Auxiliary class for dynamic shape benchmarks

    Pre-computes input with random shapes and also
    modifies the compute method so in each call the
    fuser sees a different input tensor shape
    """

    # Number of random inputs in an instance
    SAMPLE_SIZE = 100

    def __init__(self, dynamic_range=1.2):
        # [FUNCTION DOC] Initializes the empty sample list, the current sample index, the
        # shape-shrinking ratio (the inverse if the given range exceeds one, so it ends up at
        # most one) and enables dynamic shapes.
        self._input_samples = []
        self._input_sample_index = 0
        self._dynamic_range = (
            1.0 / dynamic_range if dynamic_range > 1.0 else dynamic_range
        )
        self._enable_dynamic_shapes = True

    # Returns the input test case that current index points to
    @property
    def inputs(self):
        # [FUNCTION DOC] Property: returns the input set at the current sample index.
        return self._input_samples[self._input_sample_index]

    # An inputs assignment actually adds a test case in the class buffer
    @inputs.setter
    def inputs(self, val):
        # [FUNCTION DOC] Property setter: does not replace anything; appends the assigned value
        # as a new sample input set.
        self._input_samples.append(val)

    # Runs normal compute while increment test case index
    def compute(self):
        # [FUNCTION DOC] Runs the parent compute and then advances the sample index, wrapping
        # around at the sample size.
        # KNOWN ISSUE: the result of the parent compute is dropped and none is returned, so
        #     callers that use the computed value (the backward step of the timed loop) get
        #     no result.
        super().compute()
        self._input_sample_index = (self._input_sample_index + 1) % self.SAMPLE_SIZE

    # Defined by benchmark, the benchmark needs to specify the input
    # tensor construction in this method, essentially the same way
    # a benchmark creates the inputs list in the initializer
    def instantiate_input(self):
        # [FUNCTION DOC] Abstract hook: subclasses must create one more random-shaped input set
        # (and assign it to inputs). The base version raises a not-implemented error.
        raise NotImplementedError

    # Instantiate random shaped inputs and start the benchmark run
    def run(self, args):
        # [FUNCTION DOC] Disables shape randomization when the parsed arguments ask for it,
        # precomputes the extra input samples, and then runs the parent run method.
        # force disable dynamic shape from command line
        if args.no_dynamic_shape:
            self._enable_dynamic_shapes = False
        self.load_inputs()
        super().run(args)

    # pre-compute inputs so the creations of random tensors
    # do not add to the compute time
    def load_inputs(self):
        # [FUNCTION DOC] Creates the remaining samples (sample size minus one, because
        # the constructor is expected to have created the first) by calling
        # instantiate_input repeatedly.
        for i in range(self.SAMPLE_SIZE - 1):
            self.instantiate_input()

    # returns a randomized shape
    def rand_shape(self, shape):
        # [FUNCTION DOC] Returns a shape with each dimension multiplied by an independent random
        # ratio drawn between the shrink ratio and one and truncated to an integer, or the shape
        # unchanged when dynamic shapes are disabled. Returns a list.
        if not self._enable_dynamic_shapes:
            return shape
        ratios = np.random.uniform(self._dynamic_range, 1.0, len(shape))
        dyn_shape = list(np.multiply(shape, ratios).astype(int))
        return dyn_shape


benchmark_classes = []


def register_benchmark_class(benchmark_cls):
    # [FUNCTION DOC] Adds a benchmark class to the global registry list used by the
    # command-line driver.
    benchmark_classes.append(benchmark_cls)
