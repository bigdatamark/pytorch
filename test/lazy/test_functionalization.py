# Owner(s): ["oncall: jit"]
# [MODULE DOC] Tests for functionalization behavior of lazy tensors on the TorchScript lazy
# backend. Import-time side effects: initializes the TS lazy backend and compiles a regular
# expression that strips node type suffixes from lazy IR text. Compares lazy results with CPU
# results for a module created on the lazy device, and checks IR text after assigning to the
# data attribute.

import re

import torch
import torch._lazy.metrics as metrics
import torch._lazy.ts_backend
from torch.testing._internal.common_utils import run_tests, TestCase


torch._lazy.ts_backend.init()

NODE_TYPE_PATTERN = re.compile(r", NodeType=[^\n]+")


class LazyFuncionalizationTest(TestCase):
    # [CLASS DOC] Test case for lazy tensors with functionalization (class name keeps the original
    # misspelling). Contains a test for initializing a module under the lazy device context and a
    # test for reassigning the data of a lazy tensor.
    def test_lazy_init_with_view(self):
        # [FUNCTION DOC] Runs a helper that builds and applies a small bias-free linear layer on a
        # device and compares the CPU output against two lazy runs, one with the functional storage
        # left as is and one with it reset; outputs must be equal after moving to CPU.
        def f(device, reset_storage=False):
            # [FUNCTION DOC] HELPER: Seeds the generator, resets lazy metrics when the device is
            # lazy, creates the model under the device context, and on lazy marks a step and checks
            # the synced-tensor counter: it must be 1 if the weight storage was reset, else 2 (an
            # extra tensor is synced when the functional storage is kept); otherwise raises an
            # assertion error. Then applies the model to a ones vector, marks a step on lazy, and
            # returns the output.
            torch.manual_seed(2023)

            if device == "lazy":
                metrics.reset()

            class Model(torch.nn.Module):
                # [CLASS DOC] HELPER: Module with one bias-free linear layer from 4 to 2 features.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Creates the linear layer.
                    super().__init__()
                    self.fc1 = torch.nn.Linear(4, 2, bias=False)

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Multiplies the input by the transposed weight of
                    # the layer.
                    return x @ self.fc1.weight.transpose(0, 1)

            with torch.device(device):
                model = Model()

                if device == "lazy":
                    if reset_storage:
                        torch._C._unsafe_reset_storage(model.fc1.weight)

                    torch._lazy.mark_step()

                    sync_tensors = metrics.counter_value("SyncedTensorsWithIR")
                    if reset_storage:
                        if sync_tensors != 1:
                            raise AssertionError(
                                f"Expected 1 synced tensor, got {sync_tensors}"
                            )
                    else:
                        # There is an extra tensor being unnecessarily synced if
                        # the functional storage is not reset.
                        if sync_tensors != 2:
                            raise AssertionError(
                                f"Expected 2 synced tensors, got {sync_tensors}"
                            )

                x = torch.ones(4)
                out = model(x)

                if device == "lazy":
                    torch._lazy.mark_step()

                return out

        cpu_out = f("cpu")
        lazy_out_1 = f("lazy", reset_storage=False)
        lazy_out_2 = f("lazy", reset_storage=True)

        self.assertEqual(cpu_out, lazy_out_1.to("cpu"))
        self.assertEqual(cpu_out, lazy_out_2.to("cpu"))

    def test_data_assign(self):
        # [FUNCTION DOC] Creates a lazy float tensor and checks its IR text equals the expected
        # single device-data node. Then converts it to bfloat16 and assigns that to its data
        # attribute and checks the IR text now shows the original device data followed by a
        # type-conversion copy node as root, i.e. assigning data updates the wrapper's inner tensor
        # and IR node.
        def text(lazyt):
            # [FUNCTION DOC] HELPER: Returns the lazy IR text of the given tensor with the node
            # type suffixes removed by the pattern, so the comparison is stable.
            raw = torch._C._lazy._get_tensors_text([lazyt])
            return NODE_TYPE_PATTERN.sub("", raw)

        origin = torch.rand(3, dtype=torch.float32)
        tensor = origin.to("lazy")

        self.assertExpectedInline(
            text(tensor),
            """\
IR {
  %0 = [Float[3]] lazy_tensors::device_data(), device=CPU0, ROOT=0
}
""",
        )

        # Modify the data-type of tensor, and assign it to 'data'.
        # This should update the inner tensor of FunctionalTensorWrapper,
        # changing the corresponding IR node.
        modified_tensor = tensor.to(torch.bfloat16)
        tensor.data = modified_tensor

        self.assertExpectedInline(
            text(tensor),
            """\
IR {
  %0 = [Float[3]] lazy_tensors::device_data(), device=CPU0
  %1 = [BFloat16[3]] aten::_to_copy(%0), dtype=BFloat16, layout=null, device=null, pin_memory=null, non_blocking=0, memory_format=null, ROOT=0
}
""",
        )


if __name__ == "__main__":
    run_tests()
