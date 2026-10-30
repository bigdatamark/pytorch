# [MODULE DOC] Helper to configure the TorchScript fuser and graph executor for benchmarks.
# Provides set_fuser, which flips global JIT flags through the private C extension.
import torch


def set_fuser(fuser_name, executor_name):
    # [FUNCTION DOC] Configure global JIT state. fuser_name must be te, old, none or default
    # (anything else raises AssertionError): te enables the profiling executor, executor
    # optimization, GPU fusion and the tensor-expression fuser (CPU fusion off); old uses the
    # legacy executor with optimization off and GPU fusion on, TE off; none disables fusion on
    # both devices and TE; default changes nothing. Then executor_name overrides: profiling,
    # simple, legacy set the profiling flag and optimize flag accordingly (simple only turns
    # optimization off); default or other values change nothing. Side effects: mutates process
    # wide JIT settings; returns None.
    if fuser_name not in ["te", "old", "none", "default"]:
        raise AssertionError(
            f"fuser_name must be one of 'te', 'old', 'none', 'default', but got '{fuser_name}'"
        )
    if fuser_name == "te":
        torch._C._jit_set_profiling_executor(True)
        torch._C._get_graph_executor_optimize(True)
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(True)
        torch._C._jit_set_texpr_fuser_enabled(True)
    elif fuser_name == "old":
        torch._C._jit_set_profiling_executor(False)
        torch._C._get_graph_executor_optimize(False)
        torch._C._jit_override_can_fuse_on_gpu(True)
        torch._C._jit_set_texpr_fuser_enabled(False)
    elif fuser_name == "none":
        torch._C._jit_set_profiling_executor(False)
        torch._C._get_graph_executor_optimize(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
    elif fuser_name == "default":
        pass

    # --executor overrides settings of --fuser
    if executor_name == "profiling":
        torch._C._jit_set_profiling_executor(True)
        torch._C._get_graph_executor_optimize(True)
    elif executor_name == "simple":
        torch._C._get_graph_executor_optimize(False)
    elif executor_name == "legacy":
        torch._C._jit_set_profiling_executor(False)
        torch._C._get_graph_executor_optimize(True)
    elif executor_name == "default":
        pass
