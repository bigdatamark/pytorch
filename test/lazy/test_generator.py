# Owner(s): ["oncall: jit"]
# [MODULE DOC] Tests for random number generators with lazy tensors on the TorchScript lazy
# backend. Import-time side effect: initializes the TS lazy backend. Checks that generator state is
# captured in the lazy graph and that different seeds cause recompilation.

import torch
import torch._lazy.metrics as metrics
import torch._lazy.ts_backend
from torch.testing._internal.common_utils import run_tests, TestCase


torch._lazy.ts_backend.init()


class LazyGeneratorTest(TestCase):
    # [CLASS DOC] Test case for generator handling in the lazy backend.
    def test_generator(self):
        # [FUNCTION DOC] Generates a pair of random tensors with seeded generators on CPU and on
        # the lazy device (global seeds differ between runs), marks a step, and raises an assertion
        # error unless both tensors from the lazy run match the CPU ones, showing seeds are part of
        # the traced graph.
        """
        Test that generators are being inserted into the TorchScript
        graph by setting different seeds before each call to
        generate_tensor but the resulting tensor is the same
        """

        def generate_tensor():
            # [FUNCTION DOC] HELPER: Creates two generators seeded 2023 and 2024, fills one scalar
            # tensor with a uniform draw from the first and another with a normal draw from the
            # second, and returns both.
            g1 = torch.Generator()
            g1.manual_seed(2023)
            t1 = torch.tensor(1.0)
            t1.uniform_(generator=g1)

            g2 = torch.Generator()
            g2.manual_seed(2024)
            t2 = torch.tensor(1.0)
            t2.normal_(generator=g2)

            return t1, t2

        torch.manual_seed(1)

        with torch.device("cpu"):
            cpu_t1, cpu_t2 = generate_tensor()

        torch.manual_seed(2)

        with torch.device("lazy"):
            lazy_t1, lazy_t2 = generate_tensor()

        torch._lazy.mark_step()

        if not torch.allclose(cpu_t1, lazy_t1.to("cpu")):
            raise AssertionError(f"Expected {cpu_t1}, got {lazy_t1.to('cpu')}")
        if not torch.allclose(cpu_t2, lazy_t2.to("cpu")):
            raise AssertionError(f"Expected {cpu_t2}, got {lazy_t2.to('cpu')}")

    def test_generator_causes_multiple_compiles(self):
        # [FUNCTION DOC] Resets metrics and runs the seeded generation on the lazy device three
        # times (seeds 1, 2, 1), checking after each step the compile counters: one uncached
        # compile after the first, two after the second, and after the third still two uncached
        # plus one cached. Then resets metrics again and checks the latest computation graph
        # text contains the generator with seed 1 and a uniform op. Raises assertion errors on
        # any mismatch.
        """
        Test that inserting generators with different seed caused recompile
        """

        def generate_tensor(seed):
            # [FUNCTION DOC] HELPER: Returns a scalar tensor filled with a uniform draw in the
            # range from -1 to 1 using a generator seeded with the given seed.
            t = torch.tensor(1.0)
            g = torch.Generator()
            g.manual_seed(seed)
            t.uniform_(-1, 1, generator=g)
            return t

        metrics.reset()

        with torch.device("lazy"):
            t = generate_tensor(1)
            torch._lazy.mark_step()

            uncached_compile = metrics.counter_value("UncachedCompile")
            if uncached_compile != 1:
                raise AssertionError(
                    f"Expected 1 uncached compiles, got {uncached_compile}"
                )

            t = generate_tensor(2)
            torch._lazy.mark_step()

            uncached_compile = metrics.counter_value("UncachedCompile")
            if uncached_compile != 2:
                raise AssertionError(
                    f"Expected 2 uncached compiles, got {uncached_compile}"
                )

            t = generate_tensor(1)  # noqa: F841
            torch._lazy.mark_step()

            uncached_compile = metrics.counter_value("UncachedCompile")
            if uncached_compile != 2:
                raise AssertionError(
                    f"Expected 2 uncached compiles, got {uncached_compile}"
                )
            cached_compile = metrics.counter_value("CachedCompile")
            if cached_compile != 1:
                raise AssertionError(f"Expected 1 cached compile, got {cached_compile}")

        metrics.reset()

        latest_graph = torch._C._lazy_ts_backend._get_latest_computation_graph()
        if 'torch.Generator(device="cpu", seed=1)' not in latest_graph:
            raise AssertionError("Expected generator in graph")
        if "aten::uniform" not in latest_graph:
            raise AssertionError("Expected aten::uniform in graph")


if __name__ == "__main__":
    run_tests()
