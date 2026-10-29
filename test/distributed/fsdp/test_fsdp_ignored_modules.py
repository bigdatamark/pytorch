# Owner(s): ["oncall: distributed"]

import functools
import math
import sys

import torch
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy, transformer_auto_wrap_policy
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    DEVICEInitMode,
    FSDPInitMode,
    FSDPTestContinuous,
    TransformerWithSharedParams,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
)


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class Model(torch.nn.Module):
    # [CLASS DOC] HELPER: Small feed-forward test model: layer0 (3 to 5), layer1 (a sequential
    # of three linear layers, 5 to 4 to 4 to 4), layer2 (4 to 2), layer3 (2 to 2), with a
    # shared relu after each stage. Also offers get_input, get_loss and run_backward hooks
    # in the interface the test helpers expect.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates layer0, layer1 (three linears), layer2, layer3 and relu.
        super().__init__()
        self.layer0 = torch.nn.Linear(3, 5)
        layer1_modules = [
            torch.nn.Linear(5, 4),
            torch.nn.Linear(4, 4),
            torch.nn.Linear(4, 4),
        ]
        self.layer1 = torch.nn.Sequential(*layer1_modules)
        self.layer2 = torch.nn.Linear(4, 2)
        self.layer3 = torch.nn.Linear(2, 2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies layer0 to layer3 in order, with relu after each, and
        # returns the final activation.
        z = self.relu(self.layer0(x))
        z = self.relu(self.layer1(z))
        z = self.relu(self.layer2(z))
        z = self.relu(self.layer3(z))
        return z

    def get_input(self, device):
        # [FUNCTION DOC] HELPER: Returns a one-element tuple with a random 8x3 tensor on device.
        return (torch.randn((8, 3)).to(device),)

    def get_loss(self, input, output):
        # [FUNCTION DOC] HELPER: Returns the sum of all elements of output; input is unused.
        return output.sum()

    def run_backward(self, loss):
        # [FUNCTION DOC] HELPER: Runs backward on the loss.
        loss.backward()


class IgnoredModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Module holding a single random in_dim by out_dim weight parameter
    # and no layers; its forward is a matrix product. Used as a module to be ignored by FSDP.
    def __init__(self, in_dim: int, out_dim: int) -> None:
        # [FUNCTION DOC] HELPER: Creates the random weight parameter of shape (in_dim, out_dim).
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn((in_dim, out_dim)))

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the matrix product of x and the weight.
        return x @ self.weight


class ModelWithIgnoredModules(Model):
    # [CLASS DOC] HELPER: Model whose layer1 is rebuilt as two linear layers, then
    # num_ignored IgnoredModule(4, 4) instances, then a final linear layer, so the number of
    # parameters in layer1 varies with num_ignored.
    """Adds a variable number of :class:`IgnoredModule` to ``self.layer1``."""

    def __init__(self, num_ignored: int) -> None:
        # [FUNCTION DOC] HELPER: Builds the base Model, then replaces layer1 as described in
        # the class. Raises AssertionError if num_ignored is negative.
        if not (num_ignored >= 0):
            raise AssertionError(f"Expected num_ignored >= 0, got {num_ignored}")
        super().__init__()
        layer1_modules = (
            [torch.nn.Linear(5, 4), torch.nn.Linear(4, 4)]
            + [IgnoredModule(4, 4) for _ in range(num_ignored)]
            + [torch.nn.Linear(4, 4)]
        )
        self.layer1 = torch.nn.Sequential(*layer1_modules)


class TestFSDPIgnoredModules(FSDPTestContinuous):
    # [CLASS DOC] Multi-process test case (up to 2 ranks) for FSDP ignored_modules and
    # ignored_states. Each scenario has a public test_ entry that sweeps options through
    # run_subtests and a private _test_ method that does the work.
    @property
    def world_size(self):
        # [FUNCTION DOC] HELPER: Property giving min(device count, 2) as the number of ranks.
        return min(torch.accelerator.device_count(), 2)

    def _train_model(self, model, optim, num_iters, device=torch.device(device_type)):
        # [FUNCTION DOC] HELPER: Runs num_iters training steps: gets an input from the
        # (unwrapped, if FSDP) model, runs forward, computes the loss on device, runs the
        # model's backward helper and steps the optimizer. Gradients are not zeroed.
        for _ in range(num_iters):
            module = model.module if isinstance(model, FSDP) else model
            inp = module.get_input(device)
            output = model(*inp)
            loss = module.get_loss(inp, output).to(device)
            module.run_backward(loss)
            optim.step()

    @skip_if_lt_x_gpu(2)
    def test_ignored_modules_transformer(self):
        # [FUNCTION DOC] Test driver (2+ GPUs): runs the transformer scenario over use_orig_params,
        # ignore via modules vs. via states, and auto wrap on/off.
        """Tests that ignored modules' parameters are not flattened for a
        transformer model with shared parameters."""
        self.run_subtests(
            {
                "use_orig_params": [False, True],
                "ignore_modules": [True, False],
                "use_auto_wrap": [False, True],
            },
            self._test_ignored_modules_transformer,
        )

    def _test_ignored_modules_transformer(
        self,
        use_orig_params: bool,
        ignore_modules: bool,  # as opposed to `ignored_states`
        use_auto_wrap: bool,
    ):
        # [FUNCTION DOC] HELPER: Wraps a transformer test model in FSDP ignoring its
        # transformer submodule (given as modules or as that module's parameters; optionally
        # with a linear-layer auto wrap policy after unsharing the output projection weight).
        # Sums the sizes of all flat parameters (minus alignment padding when
        # use_orig_params) with full params summoned and asserts it equals the parameter count
        # of the model minus the transformer's. Finally trains 3 steps with Adam.
        # Initialize an FSDP-wrapped transformer model that has FSDP ignore
        # the `nn.Transformer` module's parameters
        model: nn.Module = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.NO_FSDP,
            DEVICEInitMode.DEVICE_BEFORE,
            deterministic=True,
        )
        fsdp_kwargs = {"process_group": self.process_group}
        if use_auto_wrap:
            # Unshare the output projection weight and embedding weight to be
            # able to auto wrap every linear correctly
            model.output_proj.weight = nn.Parameter(model.output_proj.weight.clone())
            fsdp_kwargs["auto_wrap_policy"] = ModuleWrapPolicy({nn.Linear})
        if ignore_modules:
            fsdp_kwargs["ignored_modules"] = [model.transformer]
        else:
            fsdp_kwargs["ignored_states"] = list(model.transformer.parameters())
        wrapper_cls = FSDP
        wrapped_model = wrapper_cls(model, **fsdp_kwargs)
        # Check that the wrapped model's flattened parameter does not include
        # the ignored transformer module's parameters
        nonwrapped_model: nn.Module = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.NO_FSDP,
            DEVICEInitMode.DEVICE_BEFORE,
            deterministic=True,
        )
        if use_auto_wrap:
            nonwrapped_model.output_proj.weight = nn.Parameter(
                nonwrapped_model.output_proj.weight.clone()
            )
        total_numel = sum(p.numel() for p in nonwrapped_model.parameters())
        ignored_numel = sum(
            p.numel() for p in nonwrapped_model.transformer.parameters()
        )
        nonignored_numel = total_numel - ignored_numel
        fsdp_managed_numel = 0
        with FSDP.summon_full_params(wrapped_model):
            for handle in traversal_utils._get_fsdp_handles(wrapped_model):
                flat_param = handle.flat_param
                flat_param_numel = flat_param.numel()
                if use_orig_params:
                    # Subtract the numel contributed from alignment padding
                    padding_numel = sum(
                        numel
                        for (numel, is_padding) in zip(
                            flat_param._numels_with_padding, flat_param._is_padding_mask
                        )
                        if is_padding
                    )
                    flat_param_numel -= padding_numel
                fsdp_managed_numel += flat_param_numel
        self.assertEqual(fsdp_managed_numel, nonignored_numel)
        # Check that we can run a few iterations
        optim = torch.optim.Adam(wrapped_model.parameters(), lr=1e-3)
        self._train_model(wrapped_model, optim, 3)

    @skip_if_lt_x_gpu(2)
    def test_ignored_modules_nested(self):
        # [FUNCTION DOC] Test driver (2+ GPUs): runs the nested scenario over use_orig_params
        # and ignoring via modules vs. via states.
        """Tests that passing a module with nested FSDP modules does not
        error and still ignores non-FSDP modules' parameters."""
        self.run_subtests(
            {
                "use_orig_params": [False, True],
                "ignore_modules": [True, False],
            },
            self._test_ignored_modules_nested,
        )

    def _test_ignored_modules_nested(self, use_orig_params: bool, ignore_modules: bool):
        # [FUNCTION DOC] HELPER: Wraps one linear inside layer1 with FSDP, then wraps the whole
        # Model ignoring layer1 (as module or as its parameters). Asserts the root's single
        # flat parameter holds exactly the non-layer1 element count (padding removed when
        # use_orig_params), then trains 3 steps with Adam.
        # KNOWN ISSUE: the numel assertion is duplicated inside and after the padding branch.
        # Initialize an FSDP-wrapped nested model that first wraps the nested
        # sequential's second linear layer (`layer1[1]`) and then wraps the
        # overall model while ignoring the nested sequential (`layer1`)
        model = Model().to(device_type)
        fsdp_fn = functools.partial(FSDP, use_orig_params=use_orig_params)
        model.layer1[1] = fsdp_fn(model.layer1[1])
        if ignore_modules:
            wrapped_model = fsdp_fn(model, ignored_modules=[model.layer1])
        else:
            wrapped_model = fsdp_fn(
                model, ignored_states=list(model.layer1.parameters())
            )
        # Check that the wrapped model's flattened parameter does not include
        # the ignored nested sequential's parameters
        nonwrapped_model = Model()
        total_numel = sum(p.numel() for p in nonwrapped_model.parameters())
        ignored_numel = sum(p.numel() for p in nonwrapped_model.layer1.parameters())
        nonignored_numel = total_numel - ignored_numel
        with FSDP.summon_full_params(wrapped_model):
            flat_param = wrapped_model.params[0]
            flat_param_numel = flat_param.numel()
            if use_orig_params:
                # Subtract the numel contributed from alignment padding
                padding_numel = sum(
                    numel
                    for (numel, is_padding) in zip(
                        flat_param._numels_with_padding, flat_param._is_padding_mask
                    )
                    if is_padding
                )
                flat_param_numel -= padding_numel
                self.assertEqual(flat_param_numel, nonignored_numel)
            self.assertEqual(flat_param_numel, nonignored_numel)
        # Check that we can run a few iterations
        optim = torch.optim.Adam(wrapped_model.parameters(), lr=1e-3)
        self._train_model(wrapped_model, optim, 3)

    @skip_if_lt_x_gpu(2)
    def test_ignored_states_auto_wrap(self):
        # [FUNCTION DOC] Test driver (2+ GPUs): runs the auto wrap scenario with two policies
        # (transformer policy and module policy, both wrapping Sequential) and with or
        # without ignoring the bias of the middle layer1 linear in addition to its weight.
        transformer_policy = functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls={nn.Sequential}
        )
        self.run_subtests(
            {
                "policy": [transformer_policy, ModuleWrapPolicy((nn.Sequential,))],
                "ignore_bias": [True, False],
            },
            self._test_ignored_states_auto_wrap,
        )

    def _test_ignored_states_auto_wrap(self, policy, ignore_bias: bool):
        # [FUNCTION DOC] HELPER: Wraps Model with the policy and ignored parameters (weight, and
        # bias if ignore_bias) of layer1's middle linear, without orig params. Computes the
        # expected per-rank flat parameter sizes for layer1 and for the rest of the root
        # (ceil of unsharded count over world size) and asserts the actual flat parameters
        # are no larger.
        model = Model().to(device_type)
        ignored_states = [model.layer1[1].weight]
        if ignore_bias:
            ignored_states.append(model.layer1[1].bias)
        # Construct 2 flat parameters: one for `layer1` and one for the model
        fsdp_model = FSDP(
            model,
            # Use `False` to avoid complexity of intra-flat-parameter padding
            use_orig_params=False,
            auto_wrap_policy=policy,
            ignored_states=ignored_states,
        )
        ref_model = Model()
        expected_layer1_unsharded_numel = (
            sum(p.numel() for p in ref_model.layer1.parameters())
            - ref_model.layer1[1].weight.numel()
        )
        if ignore_bias:
            expected_layer1_unsharded_numel -= ref_model.layer1[1].bias.numel()
        expected_model_unsharded_numel = sum(
            p.numel() for p in ref_model.parameters()
        ) - sum(p.numel() for p in ref_model.layer1.parameters())
        expected_layer1_sharded_numel = math.ceil(
            expected_layer1_unsharded_numel / self.world_size
        )
        expected_model_sharded_numel = math.ceil(
            expected_model_unsharded_numel / self.world_size
        )
        self.assertLessEqual(
            fsdp_model.layer1.module._flat_param.numel(), expected_layer1_sharded_numel
        )
        self.assertLessEqual(
            fsdp_model.module._flat_param.numel(), expected_model_sharded_numel
        )

    @skip_if_lt_x_gpu(2)
    def test_ignored_modules_invalid(self):
        # [FUNCTION DOC] Test (2+ GPUs): ignoring an already FSDP-wrapped submodule is expected
        # to raise ValueError, and ignoring the top-level module itself must emit a
        # UserWarning that all parameters will be ignored (using a fresh model).
        # KNOWN ISSUE: the msg argument given to the error assertion is only the failure
        # message, it is not matched against the raised error text.
        """Tests that passing an FSDP module as an ignored module or the
        top-level module itself errors."""
        model = Model().to(device_type)
        wrap_cls = FSDP
        model.layer1 = wrap_cls(model.layer1)
        # Passing an FSDP module as an ignored module should error
        with self.assertRaises(
            ValueError,
            msg="`ignored_modules` should not include FSDP modules",
        ):
            wrap_cls(model, ignored_modules=[model.layer1])
        with self.assertWarnsRegex(
            expected_warning=UserWarning,
            expected_regex="Trying to ignore the top-level module passed into "
            "the FSDP constructor itself will result in all parameters being "
            "ignored",
        ):
            # FSDP does not allow to wrap the same model twice, so create
            # a new local model here.
            new_model = Model().to(device_type)
            wrap_cls(new_model, ignored_modules=[new_model])

    @skip_if_lt_x_gpu(2)
    def test_diff_ignored_modules_across_ranks(self):
        # [FUNCTION DOC] Test driver (2+ GPUs): runs the cross-rank scenario over whether the
        # root receives the ignored modules and whether they are given as modules or states.
        """
        Tests ignoring different modules across ranks.

        Args:
            pass_ignored_modules_to_root (bool): If ``False``, does not pass
                any ignored modules (including those already ignored in child
                FSDP instances) to the root FSDP instance; if ``True``, passes
                all ignored modules (representing a superset of the children's
                ignored modules) to the root FSDP instance.
        """
        self.run_subtests(
            {
                "pass_ignored_modules_to_root": [False, True],
                "ignore_modules": [True, False],
            },
            self._test_diff_ignored_modules_across_ranks,
        )

    def _test_diff_ignored_modules_across_ranks(
        self,
        pass_ignored_modules_to_root: bool,
        ignore_modules: bool,
    ):
        # [FUNCTION DOC] HELPER: Each rank builds a model with rank+1 ignored modules in layer1
        # so flat parameter layouts differ across ranks. Wraps layer1 (ignoring those modules)
        # and layer3 with FSDP, then the root (optionally also ignoring all of them), and
        # trains 3 Adam steps to check nothing hangs or errors.
        # To exercise different `FlatParameter` enumerations across ranks,
        # we wrap `layer3` with FSDP, where `layer3` is registered as a module
        # after `layer1`, which has the variable number of ignored modules
        wrap_cls = FSDP
        model = ModelWithIgnoredModules(num_ignored=self.rank + 1).to(device_type)
        layer1_ignored_modules = [
            m for m in model.layer1.modules() if isinstance(m, IgnoredModule)
        ]
        ignore_kwargs = (
            {"ignored_modules": layer1_ignored_modules}
            if ignore_modules
            else {
                "ignored_states": (
                    p for m in layer1_ignored_modules for p in m.parameters()
                )
            }
        )
        model.layer1 = wrap_cls(model.layer1, **ignore_kwargs)
        model.layer3 = wrap_cls(model.layer3)
        model_ignored_modules = (
            [m for m in model.modules() if isinstance(m, IgnoredModule)]
            if pass_ignored_modules_to_root
            else []
        )
        ignore_kwargs_top = (
            {"ignored_modules": model_ignored_modules}
            if ignore_modules
            else {
                "ignored_states": {
                    p for m in model_ignored_modules for p in m.parameters()
                }
            }
        )
        wrapped_model = wrap_cls(model, **ignore_kwargs_top)
        optim = torch.optim.Adam(wrapped_model.parameters(), lr=1e-3)
        self._train_model(wrapped_model, optim, 3)

    @skip_if_lt_x_gpu(2)
    @parametrize("ignore_modules", [True, False])
    def test_ignored_modules_not_under_wrapped_root(self, ignore_modules: bool):
        # [FUNCTION DOC] Test (2+ GPUs): takes layer1's children after the first as ignored
        # modules (or their parameters) and passes the same set to FSDP wrappers of both
        # layer1 and layer3, so for layer3 the ignored items are not under its root. Trains 3
        # Adam steps and expects no error.
        model = Model().to(device_type)
        ignored_modules = list(model.layer1.children())[1:]

        ignore_kwargs = (
            {"ignored_modules": ignored_modules}
            if ignore_modules
            else {
                "ignored_states": {p for m in ignored_modules for p in m.parameters()}
            }
        )

        wrap_cls = FSDP

        model.layer1 = wrap_cls(
            model.layer1,
            **ignore_kwargs,
        )
        model.layer3 = wrap_cls(
            model.layer3,
            # the ignored modules/parameters contains submodule under model.layer1, which
            # is out of the local root model.layer3.
            **ignore_kwargs,
        )

        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        self._train_model(model, optim, 3)

    @skip_if_lt_x_gpu(1)
    def test_ignored_states_check(self):
        # [FUNCTION DOC] Test driver (1+ GPU): runs the argument validation scenario for both
        # ignored_modules and ignored_states.
        """
        Tests that passing invalid ``ignored_modules`` or ``ignored_states``
        raises an appropriate error.
        """
        self.run_subtests(
            {"ignore_modules": [True, False]},
            self._test_ignored_states_check,
        )

    def _test_ignored_states_check(self, ignore_modules: bool):
        # [FUNCTION DOC] HELPER: Asserts ValueError with specific messages for invalid
        # arguments: ignored_modules given as parameters; both ignored_modules and
        # ignored_states together; and ignored_states mixing modules and parameters.
        model = Model().to(device_type)
        ignored_modules = list(model.layer1.children())[1:]
        ignored_params = {p for m in ignored_modules for p in m.parameters()}
        ignored_states = ignored_params.union(set(ignored_modules))
        if ignore_modules:
            # Check that passing `ignored_modules` not as uniformly `nn.Module`
            # raises an error
            with self.assertRaisesRegex(
                ValueError,
                "ignored_modules expects nn.Module list elements but got types "
                r"\[<class 'torch.nn.parameter.Parameter'>\]",
            ):
                FSDP(model, ignored_modules=ignored_params)
            # Check that passing both `ignored_modules` and `ignored_states`
            # raises an error (and fold this only into `ignore_modules=True`)
            with self.assertRaisesRegex(
                ValueError,
                "Cannot pass both ignored_modules and ignored_states at the same time",
            ):
                FSDP(
                    model,
                    ignored_modules=ignored_modules,
                    ignored_states=ignored_params,
                )
        else:
            # Check that passing `ignored_states` not as uniformly
            # `nn.Parameter` or uniformly `nn.Module` raises an error
            with self.assertRaisesRegex(
                ValueError,
                "ignored_states expects all nn.Parameter or all nn.Module list "
                r"elements but got types \[<class 'torch.nn.modules.linear.Linear'>, "
                r"<class 'torch.nn.parameter.Parameter'>\]",
            ):
                FSDP(model, ignored_states=ignored_states)


instantiate_parametrized_tests(TestFSDPIgnoredModules)

if __name__ == "__main__":
    run_tests()
