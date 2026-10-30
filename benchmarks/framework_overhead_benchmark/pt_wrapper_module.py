# [MODULE DOC] Wrapper that prepares a module for the framework overhead benchmark: builds the
# module, creates random inputs, optionally traces and saves it, and runs its forward repeatedly.
import torch


class WrapperModule:
    # [CLASS DOC] Holds the wrapped module instance (module, traced in graph mode), its name, and
    # the list of random one element input tensors (tensor_inputs). The forward method runs the
    # module a given number of times without gradients, used for timing.
    """Wraps the instance of wrapped_type.
    For graph_mode traces the instance of wrapped_type.
    Randomaly initializes num_params tensors with single float element.
    Args:
        wrapped_type:
            - Object type to be wrapped.
                Expects the wrapped_type to:
                   - be constructed with pt_fn specified in module_config.
                   - provide forward method that takes module_config.num_params args.
        module_config:
            - Specified pt_fn to construct wrapped_type with, whether graph_mode
              is enabled, and number of parameters wrapped_type's forward method
              takes.
        debug:
            - Whether debug mode is enabled.
        save:
            - In graph mode, whether graph is to be saved.
    """

    def __init__(self, wrapped_type, module_config, debug, save=False):
        # [FUNCTION DOC] Instantiate wrapped_type with the function from module_config, create
        # num_params random tensors with a single element, and in graph mode trace the module
        # with them. If save is set the traced module is saved to a file named after the type and
        # function in the current directory and a message is printed. Prints a benchmarking
        # banner; when debug is set and the module is traced, prints its graph and code.
        pt_fn = module_config.pt_fn
        self.module = wrapped_type(pt_fn)
        self.tensor_inputs = []
        self.module_name = wrapped_type.__name__
        for _ in range(module_config.num_params):
            self.tensor_inputs.append(torch.randn(1))
        if module_config.graph_mode:
            self.module = torch.jit.trace(self.module, self.tensor_inputs)
            if save:
                file_name = self.module_name + "_" + pt_fn.__name__ + ".pt"
                torch.jit.save(self.module, file_name)
                print(f"Generated graph is saved in {file_name}")
        print(
            f"Benchmarking module {self.module_name} with fn {pt_fn.__name__}: Graph mode:{module_config.graph_mode}"
        )
        if debug and isinstance(self.module, torch.jit.ScriptModule):
            print(self.module.graph)
            print(self.module.code)

    def forward(self, niters):
        # [FUNCTION DOC] Run the module niters times on the stored inputs under no-grad mode.
        # Returns None.
        with torch.no_grad():
            for _ in range(niters):
                self.module.forward(*self.tensor_inputs)
