# Owner(s): ["oncall: jit"]
# [MODULE DOC] Tests for torch.Generator support in TorchScript: generators created and seeded
# inside scripted code, the default generator, generators passed as arguments, and save/load of
# modules that use generators. Eager and scripted results are compared under identical seeding.
# Not runnable directly: running the file as a script raises an error that points to the
# umbrella test runner.

import io
import math
import unittest

import torch
from torch.nn import init
from torch.testing._internal.common_utils import (
    raise_on_run_directly,
    skipIfLegacyJitExecutor,
)
from torch.testing._internal.jit_utils import JitTestCase


class TestGenerator(JitTestCase):
    # [CLASS DOC] JitTestCase with tests comparing eager and TorchScript random number
    # generation through explicit Generator objects.
    # torch.jit.trace does not properly capture the generator manual seed
    # and thus is non deterministic even if the generator is manually seeded
    @skipIfLegacyJitExecutor("legacy JIT executor does not support Generator type")
    @unittest.expectedFailure
    def test_trace(self):
        # [FUNCTION DOC] Traces a function that creates a generator, seeds it, reads its initial
        # seed and fills a tensor with uniform samples drawn from it, then three times compares
        # the eager result under one global seed with the traced result under a different global
        # seed. Decorated as an expected failure and skipped for the legacy JIT executor.
        # KNOWN ISSUE: expected to fail because tracing does not capture the manual seeding of
        # the generator, so the traced output is not deterministic.
        def f():
            # [FUNCTION DOC] HELPER: Creates a new generator, seeds it randomly then with a
            # fixed value, reads the initial seed, and returns a 2 by 2 tensor filled with
            # uniform samples from that generator.
            generator = torch.Generator()
            generator.seed()
            generator.manual_seed(2023)
            generator.initial_seed()
            tensor = torch.empty(2, 2)
            tensor.uniform_(0, 1, generator=generator)
            return tensor

        traced_f = torch.jit.trace(f, ())

        # Run this 3 times to ensure that the generator is being manually seeded
        # each time the traced function is run
        for _ in range(3):
            torch.manual_seed(1)

            eager_tensor = f()

            # Change the seed of the default generator to
            # check that we're using the generator from the
            # trace
            torch.manual_seed(2)
            traced_tensor = traced_f()

            self.assertEqual(eager_tensor, traced_tensor)

    def test_script(self):
        # [FUNCTION DOC] Scripts a function that builds and manually seeds its own generator and
        # returns normal samples from it. Three times, with the global seed set differently
        # before the eager and scripted calls, asserts the eager and scripted tensors are equal.
        def f():
            # [FUNCTION DOC] HELPER: Creates a new generator, seeds it randomly then with a
            # fixed value, reads the initial seed, and returns a 2 by 2 tensor of normal samples
            # (mean -1, std 1) drawn from it.
            generator = torch.Generator()
            generator.seed()
            generator.manual_seed(2023)
            generator.initial_seed()
            tensor = torch.empty(2, 2)
            tensor.normal_(-1.0, 1.0, generator=generator)
            return tensor

        script_f = torch.jit.script(f, ())

        # Run this 3 times to ensure that the generator is being manually seeded
        # each time the traced function is run
        for _ in range(3):
            torch.manual_seed(1)

            eager_tensor = f()

            # Change the seed of the default generator to
            # check that we're using the generator from the
            # trace
            torch.manual_seed(2)

            script_tensor = script_f()

            self.assertEqual(eager_tensor, script_tensor)

    def test_default_generator(self):
        # [FUNCTION DOC] Seeds the global generator, runs the eager function, reseeds it
        # differently, runs the scripted version, and asserts both tensors are equal since the
        # function reseeds the default generator itself.
        def f():
            # [FUNCTION DOC] HELPER: Seeds the default generator with a fixed value and returns
            # a 2 by 2 tensor of normal samples drawn from it.
            # check that calling manual seed for the default generator works
            torch.manual_seed(2023)
            tensor = torch.empty(2, 2)
            tensor.normal_(-1.0, 1.0)
            return tensor

        torch.manual_seed(1)

        eager_tensor = f()

        torch.manual_seed(2)

        script_f = torch.jit.script(f, ())
        script_tensor = script_f()

        self.assertEqual(eager_tensor, script_tensor)

    def test_generator_arg(self):
        # [FUNCTION DOC] Scripts a function taking a Generator argument using example inputs.
        # For three iterations, builds identically seeded generators for the eager and scripted
        # calls, sets the global seed to the same value before each, and asserts equal outputs.
        def f(generator: torch.Generator):
            # [FUNCTION DOC] HELPER: Returns a 2 by 2 tensor of normal samples (mean -1, std 1)
            # drawn from the given generator.
            tensor = torch.empty(2, 2)
            tensor.normal_(-1.0, 1.0, generator=generator)
            return tensor

        generator = torch.Generator()
        generator.manual_seed(2023)

        script_f = torch.jit.script(f, (generator,))

        for i in range(3):
            generator = torch.Generator()
            generator.manual_seed(2023 + i)

            torch.manual_seed(1 + i)

            eager_tensor = f(generator)

            generator = torch.Generator()
            generator.manual_seed(2023 + i)

            torch.manual_seed(1 + i)

            script_tensor = script_f(generator)

            self.assertEqual(eager_tensor, script_tensor)

    def test_save_load(self):
        # [FUNCTION DOC] Scripts a module that initializes its two linear layers from seeded
        # generators, saves it to a memory buffer and loads it back. Asserts the loaded layer
        # weights equal the eager module's weights, and for three iterations asserts that
        # outputs and the generator-drawn random tensor of loaded and eager modules agree. On a
        # mismatch it prints diagnostic values (and the loaded forward code) before re-raising.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two bias-free 2 by 2 linear layers whose weights
            # are set at construction from generators with fixed seeds. Forward applies both
            # layers and also returns normal noise from a generator seeded inside forward.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Calls the base initializer, creates the two linear
                # layers foo and bar, and runs the parameter reset.
                super().__init__()
                self.foo = torch.nn.Linear(2, 2, bias=False)
                self.bar = torch.nn.Linear(2, 2, bias=False)

                self.reset_parameters()

            def reset_linear(self, module, generator):
                # [FUNCTION DOC] HELPER: Re-initializes the weight of the given module with
                # Kaiming uniform initialization (gain parameter square root of five) using the
                # supplied generator.
                init.kaiming_uniform_(
                    module.weight, a=math.sqrt(5), generator=generator
                )

            def reset_parameters(self):
                # [FUNCTION DOC] HELPER: Resets foo using a generator seeded with 1 and bar
                # using a fresh generator seeded with 2, so initial weights are reproducible.
                generator = torch.Generator()
                generator.manual_seed(1)
                self.reset_linear(self.foo, generator)

                generator = torch.Generator()
                generator.manual_seed(2)
                self.reset_linear(self.bar, generator)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the foo layer then the bar layer, draws normal
                # noise shaped like the result from a fresh generator seeded with 3, and returns
                # the pair of output and noise.
                x = self.foo(x)
                x = self.bar(x)

                generator = torch.Generator()
                generator.manual_seed(3)
                r = torch.empty_like(x)
                r.normal_(0.0, 1.0, generator=generator)

                return x, r

        eager_foo = Foo()

        script_module = torch.jit.script(Foo())
        saved_module = io.BytesIO()
        torch.jit.save(script_module, saved_module)
        saved_module.seek(0)

        loaded_module = torch.jit.load(saved_module)

        self.assertEqual(eager_foo.foo.weight, loaded_module.foo.weight)
        self.assertEqual(eager_foo.bar.weight, loaded_module.bar.weight)

        try:
            # Run this 3 times so make sure that the generator seed is being set
            # every time forward is called
            for i in range(3):
                x = torch.ones(2, 2)
                out1, r1 = eager_foo(x)
                out2, r2 = loaded_module(x)

                try:
                    self.assertEqual(out1, out2)
                except:
                    print(f"Iteration {i}:\n{out1=}\n{out2=}")
                    raise

                try:
                    self.assertEqual(r1, r2)
                except:
                    print(f"Iteration {i}:\n{r1=}\n{r2=}")
                    raise
        except:
            print(loaded_module.forward.code)
            raise


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
