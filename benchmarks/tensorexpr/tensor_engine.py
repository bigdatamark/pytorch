# [MODULE DOC] Holds the globally selected tensor engine for the benchmark harness and helpers
# to select, fetch and annotate engine methods. The module global starts empty until an engine
# mode is set. Engine modules are imported lazily when selected.
tensor_engine = None


def unsupported(func):
    # [FUNCTION DOC] Decorator marking an engine method as unsupported: it wraps the method in
    # a function taking only the instance, sets a not-supported attribute on the wrapper and
    # returns the wrapper.
    # KNOWN ISSUE: the wrapper accepts only the instance, so the wrapped method cannot be
    # called with extra arguments.
    def wrapper(self):
        # [FUNCTION DOC] HELPER: Calls the wrapped method with only the instance and returns
        # its result.
        return func(self)

    wrapper.is_supported = False
    return wrapper


def is_supported(method):
    # [FUNCTION DOC] Returns the value of the support attribute if the method carries one
    # (false for methods marked unsupported), otherwise True.
    if hasattr(method, "is_supported"):
        return method.is_supported
    return True


def set_engine_mode(mode):
    # [FUNCTION DOC] Selects the tensor engine by mode string: tf, pt, topi, relay or nnc,
    # importing the corresponding engine module lazily, creating its engine object and storing
    # it in the module global. Also records the mode string on the engine object. Raises
    # ValueError for any other mode. Mutates global state.
    global tensor_engine
    if mode == "tf":
        from . import tf_engine

        tensor_engine = tf_engine.TensorFlowEngine()
    elif mode == "pt":
        from . import pt_engine

        tensor_engine = pt_engine.TorchTensorEngine()
    elif mode == "topi":
        from . import topi_engine

        tensor_engine = topi_engine.TopiEngine()
    elif mode == "relay":
        from . import relay_engine

        tensor_engine = relay_engine.RelayEngine()
    elif mode == "nnc":
        from . import nnc_engine

        tensor_engine = nnc_engine.NncEngine()
    else:
        raise ValueError(f"invalid tensor engine mode: {mode}")
    tensor_engine.mode = mode


def get_engine():
    # [FUNCTION DOC] Returns the current global tensor engine. Raises ValueError if no engine
    # mode has been set yet.
    if tensor_engine is None:
        raise ValueError("use of get_engine, before calling set_engine_mode is illegal")
    return tensor_engine
