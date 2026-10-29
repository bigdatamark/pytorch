# Owner(s): ["module: sparse"]
# [MODULE DOC]
# Tests for the experimental structured pruners in torch.ao.pruning._experimental.pruner:
# BaseStructuredSparsifier (via small test subclasses), SaliencyPruner, LSTMSaliencyPruner
# and FPGMPruner. Covers prepare, mask updates, and prune() across Linear, Conv2d, pooling,
# activation and LSTM model patterns from common_pruning. Device-generic test classes are
# instantiated by the caller (test/test_ao_sparsity.py); running this file directly raises.
import copy
import random

import torch
from torch import nn
from torch.ao.pruning._experimental.pruner import (
    BaseStructuredSparsifier,
    FakeStructuredSparsity,
    FPGMPruner,
    LSTMSaliencyPruner,
    SaliencyPruner,
)
from torch.nn.utils import parametrize
from torch.testing._internal.common_device_type import dtypes, onlyCPU
from torch.testing._internal.common_pruning import (
    Conv2dActivation,
    Conv2dBias,
    Conv2dPadBias,
    Conv2dPool,
    Conv2dPoolFlatten,
    Conv2dPoolFlattenFunctional,
    LinearActivation,
    LinearActivationFunctional,
    LinearBias,
    LSTMLayerNormLinearModel,
    LSTMLinearModel,
    rows_are_subset,
    SimpleConv2d,
    SimpleLinear,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    raise_on_run_directly,
    skipIfTorchDynamo,
    TestCase,
)


class SimplePruner(BaseStructuredSparsifier):
    # [CLASS DOC] HELPER: minimal concrete structured pruner that prunes exactly the second
    # output channel (row 1) of every prepared weight; keeps no state of its own.
    def update_mask(self, module, tensor_name, **kwargs):
        # [FUNCTION DOC] HELPER: sets the mask entry at index 1 of the module's weight
        # parametrization to False, in place. Extra keyword arguments are ignored.
        getattr(module.parametrizations, tensor_name)[0].mask[1] = False


class ImplementedPruner(BaseStructuredSparsifier):
    # [CLASS DOC] HELPER: structured pruner that randomly masks out one third of the output
    # channels of each prepared weight; keeps no state of its own.
    def update_mask(self, module, tensor_name, **kwargs):
        # [FUNCTION DOC] HELPER: randomly samples one third (integer division) of the mask
        # row indices and sets them to False in the weight parametrization, in place.
        # Uses the global random module, so results are not seeded here.
        """Prunes 1/3 of the weight output channels, so resulting module has 33.3% pruning"""
        num_rows = len(module.parametrizations[tensor_name][0].mask)
        prune = random.sample(list(range(num_rows)), num_rows // 3)
        module.parametrizations[tensor_name][0].mask[prune] = False


class BottomHalfLSTMPruner(BaseStructuredSparsifier):
    # [CLASS DOC] HELPER: deterministic structured pruner for LSTM weights; masks the bottom
    # half of each of the four gate blocks. Keeps no state of its own.
    """
    Pruner that will remove the bottom half of the rows.
    This is primarily meant for testing purposes
    """

    def update_mask(self, module, tensor_name, **kwargs):
        # [FUNCTION DOC] HELPER: for the FakeStructuredSparsity parametrization of the named
        # tensor, splits the mask into four equal gate chunks, sets the second half of each
        # chunk to False, and writes the concatenated mask back into the original mask.
        for p in getattr(module.parametrizations, tensor_name):
            if isinstance(p, FakeStructuredSparsity):
                mask = p.mask
                masks = torch.split(mask, len(mask) // 4)
                for small in masks:
                    num = len(small)
                    small[num // 2 :] = False
                new_mask = torch.cat(masks)
                mask.data = new_mask.data


class TestSaliencyPruner(TestCase):
    # [CLASS DOC] Device-generic tests for SaliencyPruner and LSTMSaliencyPruner; stateless.
    @dtypes(torch.float)
    def test_saliency_pruner_update_mask(self, device, dtype):
        # [FUNCTION DOC] Sets linear1 rows to constant values 1..4, prunes at sparsity 0.5,
        # and asserts the two lowest-saliency rows are removed, leaving rows 3 and 4.
        """Test that we prune out the row with the lowest saliency (first row)"""
        model = SimpleLinear().to(device)
        with torch.no_grad():
            model.linear1.weight = nn.Parameter(
                torch.tensor(
                    [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
                    dtype=dtype,
                    device=device,
                )
            )
        pruning_config = [{"tensor_fqn": "linear1.weight", "sparsity_level": 0.5}]
        pruner = SaliencyPruner({})

        pruner.prepare(model, pruning_config)
        pruner.enable_mask_update = True
        pruner.step()
        pruned_model = pruner.prune()

        expected = torch.tensor(
            [[3, 3, 3, 3], [4, 4, 4, 4]], dtype=dtype, device=device
        )
        pruned = pruned_model.linear1.weight

        if expected.shape != pruned.shape:
            raise AssertionError(f"Expected shape {expected.shape}, got {pruned.shape}")
        if not torch.isclose(expected, pruned, rtol=1e-05, atol=1e-07).all():
            raise AssertionError("Expected and pruned tensors are not close")

    @dtypes(torch.float)
    def test_lstm_saliency_pruner_update_mask(self, device, dtype):
        # [FUNCTION DOC] Installs hand-picked LSTM weights and biases, prunes half of the
        # hidden units with LSTMSaliencyPruner, runs both models, and asserts the kept
        # rows of weight_ih, weight_hh and both biases are the highest-saliency ones.
        # Note the expected values are compared against the original model's parameters,
        # which prune() has already modified.
        model = LSTMLinearModel(
            input_dim=2,
            hidden_dim=2,
            output_dim=2,
            num_layers=1,
        ).to(device)

        manual_weights = torch.tensor(
            [[1, 1], [2, 2], [2, 2], [1, 1], [-1, -1], [-2, -2], [-2, -2], [-1, -1]],
            dtype=dtype,
            device=device,
        )

        with torch.no_grad():
            model.lstm.weight_ih_l0 = nn.Parameter(manual_weights)
            model.lstm.weight_hh_l0 = nn.Parameter(manual_weights.clone())
            model.lstm.bias_ih_l0 = nn.Parameter(manual_weights[:, 0])
            model.lstm.bias_hh_l0 = nn.Parameter(manual_weights[:, 0])

        config = [
            {"tensor_fqn": "lstm.weight_ih_l0"},
            {"tensor_fqn": "lstm.weight_hh_l0"},
        ]
        lstm_input = torch.ones((1, 2), device=device)
        fx_pruner = LSTMSaliencyPruner({"sparsity_level": 0.5})
        fx_pruner.prepare(model, config)
        fx_pruner.enable_mask_update = True
        fx_pruner.step()

        model.eval()
        pruned_model = fx_pruner.prune()
        pruned_model.eval()

        # make sure both models run
        model(lstm_input)
        pruned_model(lstm_input)

        # make sure lowest saliency rows are pruned
        expected = torch.tensor(
            [[2, 2], [2, 2], [-2, -2], [-2, -2]], dtype=dtype, device=device
        )
        pruned = model.lstm.weight_ih_l0
        if expected.shape != pruned.shape:
            raise AssertionError(f"Expected shape {expected.shape}, got {pruned.shape}")
        if not torch.isclose(expected, pruned, rtol=1e-05, atol=1e-07).all():
            raise AssertionError("Expected and pruned tensors are not close")

        expected = torch.tensor([[2], [2], [-2], [-2]], dtype=dtype, device=device)
        pruned = model.lstm.weight_hh_l0
        if expected.shape != pruned.shape:
            raise AssertionError(f"Expected shape {expected.shape}, got {pruned.shape}")
        if not torch.isclose(expected, pruned, rtol=1e-05, atol=1e-07).all():
            raise AssertionError("Expected and pruned tensors are not close")

        expected = torch.tensor([2, 2, -2, -2], dtype=dtype, device=device)
        for pruned in [model.lstm.bias_ih_l0, model.lstm.bias_hh_l0]:
            if expected.shape != pruned.shape:
                raise AssertionError(
                    f"Expected shape {expected.shape}, got {pruned.shape}"
                )
            if not torch.isclose(expected, pruned, rtol=1e-05, atol=1e-07).all():
                raise AssertionError("Expected and pruned tensors are not close")


class TestBaseStructuredSparsifierCPU(TestCase):
    # [CLASS DOC] CPU-only tests of structured pruning on LSTM models using
    # BottomHalfLSTMPruner; tagged with the CPU hardware classification; stateless.
    hw_classification = HardwareClassification.CPU

    def test_prune_lstm_linear_multiple_layer(self):
        # [FUNCTION DOC] Prunes a 2-layer LSTM followed by Linear, runs original and pruned
        # models, then checks each parameter's rows are a subset of the expected ones.
        # KNOWN ISSUE: expected_params is built from the original model itself, and the
        # loop iterates that same model, so the subset check is trivially true.
        """
        Test fusion support for LSTM(multi-layer) -> Linear
        """
        device = "cpu"

        model = LSTMLinearModel(
            input_dim=8,
            hidden_dim=8,
            output_dim=8,
            num_layers=2,
        ).to(device)

        config = [
            {"tensor_fqn": "lstm.weight_ih_l0"},
            {"tensor_fqn": "lstm.weight_hh_l0"},
            {"tensor_fqn": "lstm.weight_ih_l1"},
            {"tensor_fqn": "lstm.weight_hh_l1"},
        ]

        lstm_input = torch.ones((1, 8), device=device)
        fx_pruner = BottomHalfLSTMPruner({"sparsity_level": 0.5})
        fx_pruner.prepare(model, config)

        fx_pruner.enable_mask_update = True
        fx_pruner.step()

        model.eval()
        _, _ = model(lstm_input)
        pruned_model = fx_pruner.prune()
        pruned_model.eval()
        _, _ = pruned_model(lstm_input)

        expected_params = dict(model.named_parameters())
        for name, param in model.named_parameters():
            if name not in expected_params:
                raise AssertionError(f"Expected parameter '{name}' in expected_params")
            # We cannot compare y_expected == y_pruned, as the 0 elements mess up the numerics
            # Instead we check that the weights of the new LSTM are a subset of the weights of
            # the old LSTM
            if not rows_are_subset(param, expected_params[name]):
                raise AssertionError(f"Parameter '{name}' rows are not a subset")
            del expected_params[name]

        # assert we haven't deleted any keys
        if len(expected_params) != 0:
            raise AssertionError(
                f"Expected all params deleted, but {len(expected_params)} remain"
            )

    def test_prune_lstm_linear_single_layer(self):
        # [FUNCTION DOC] Prunes a 1-layer LSTM followed by Linear; asserts the pruned LSTM
        # output equals the first half of the original output and the final Linear output
        # keeps the same shape (its input columns were resized).
        """
        Test fusion support for LSTM (single-layer) -> Linear
        """
        device = "cpu"

        model = LSTMLinearModel(
            input_dim=8,
            hidden_dim=8,
            output_dim=8,
            num_layers=1,
        ).to(device)

        config = [
            {"tensor_fqn": "lstm.weight_ih_l0"},
            {"tensor_fqn": "lstm.weight_hh_l0"},
        ]

        lstm_input = torch.ones((1, 8), device=device)
        fx_pruner = BottomHalfLSTMPruner({"sparsity_level": 0.5})
        fx_pruner.prepare(model, config)
        fx_pruner.enable_mask_update = True
        fx_pruner.step()
        model.eval()

        out_expected, lstm_out_expected = model(lstm_input)
        pruned_model = fx_pruner.prune()
        pruned_model.eval()
        out_pruned, lstm_out_pruned = pruned_model(lstm_input)
        _, c = lstm_out_expected.size()

        # We cannot check that y_expected == y_pruned as usual because
        # zeros vs. missing elements yield different numerical results.
        # Instead that we check that the pruned elements are the first half of the results
        # since we are using a BottomHalfLSTMPruner
        if not torch.isclose(
            lstm_out_expected[:, : c // 2], lstm_out_pruned, rtol=1e-05, atol=1e-07
        ).all():
            raise AssertionError("LSTM outputs are not close")
        # also check that output of linear is the same shape, this means we've resized
        # linear columns correctly.
        if out_expected.shape != out_pruned.shape:
            raise AssertionError(
                f"Expected shape {out_expected.shape}, got {out_pruned.shape}"
            )

    def test_prune_lstm_layernorm_linear_multiple_layer(self):
        # [FUNCTION DOC] Same as the multi-layer LSTM-Linear test but with the
        # LSTM-LayerNorm-Linear model.
        # KNOWN ISSUE: the expected_params comparison uses the original model against
        # itself, so the subset check is trivially true.
        """
        Test fusion support for LSTM(multi-layer) -> Linear
        """
        device = "cpu"

        model = LSTMLayerNormLinearModel(
            input_dim=8,
            output_dim=8,
            hidden_dim=8,
            num_layers=2,
        ).to(device)

        config = [
            {"tensor_fqn": "lstm.weight_ih_l0"},
            {"tensor_fqn": "lstm.weight_hh_l0"},
            {"tensor_fqn": "lstm.weight_ih_l1"},
            {"tensor_fqn": "lstm.weight_hh_l1"},
        ]

        lstm_input = torch.ones((1, 8), device=device)
        fx_pruner = BottomHalfLSTMPruner({"sparsity_level": 0.5})
        fx_pruner.prepare(model, config)

        fx_pruner.enable_mask_update = True
        fx_pruner.step()

        model.eval()
        _, _ = model(lstm_input)
        pruned_model = fx_pruner.prune()
        pruned_model.eval()
        _, _ = pruned_model(lstm_input)

        expected_params = dict(model.named_parameters())
        for name, param in model.named_parameters():
            if name not in expected_params:
                raise AssertionError(f"Expected parameter '{name}' in expected_params")
            # We cannot compare y_expected == y_pruned, as the 0 elements mess up the numerics
            # Instead we check that the weights of the new LSTM are a subset of the weights of
            # the old LSTM
            if not rows_are_subset(param, expected_params[name]):
                raise AssertionError(f"Parameter '{name}' rows are not a subset")
            del expected_params[name]

        # assert we haven't deleted any keys
        if len(expected_params) != 0:
            raise AssertionError(
                f"Expected all params deleted, but {len(expected_params)} remain"
            )

    def test_prune_lstm_layernorm_linear_single_layer(self):
        # [FUNCTION DOC] Single-layer variant; checks pruned LSTM output equals the first
        # half of the original and the final output shape is unchanged.
        # KNOWN ISSUE: despite the name it builds LSTMLinearModel, not
        # LSTMLayerNormLinearModel, so it duplicates the plain LSTM-Linear test.
        """
        Test fusion support for LSTM (single-layer) -> Linear
        """
        device = "cpu"

        model = LSTMLinearModel(
            input_dim=8,
            hidden_dim=8,
            output_dim=8,
            num_layers=1,
        ).to(device)

        config = [
            {"tensor_fqn": "lstm.weight_ih_l0"},
            {"tensor_fqn": "lstm.weight_hh_l0"},
        ]

        lstm_input = torch.ones((1, 8), device=device)
        fx_pruner = BottomHalfLSTMPruner({"sparsity_level": 0.5})
        fx_pruner.prepare(model, config)
        fx_pruner.enable_mask_update = True
        fx_pruner.step()
        model.eval()

        out_expected, lstm_out_expected = model(lstm_input)
        pruned_model = fx_pruner.prune()
        pruned_model.eval()
        out_pruned, lstm_out_pruned = pruned_model(lstm_input)
        _, c = lstm_out_expected.size()

        # We cannot check that y_expected == y_pruned as usual because
        # zeros vs. missing elements yield different numerical results.
        # Instead that we check that the pruned elements are the first half of the results
        # since we are using a BottomHalfLSTMPruner
        if not torch.isclose(
            lstm_out_expected[:, : c // 2], lstm_out_pruned, rtol=1e-05, atol=1e-07
        ).all():
            raise AssertionError("LSTM outputs are not close")
        # also check that output of linear is the same shape, this means we've resized
        # linear columns correctly.
        if out_expected.shape != out_pruned.shape:
            raise AssertionError(
                f"Expected shape {out_expected.shape}, got {out_pruned.shape}"
            )


class TestBaseStructuredSparsifierDevice(TestCase):
    # [CLASS DOC] Device-generic tests of BaseStructuredSparsifier on Linear and Conv2d
    # models (constructor, prepare, step, prune); tagged with the accelerator hardware
    # classification. Private helper methods perform the per-device checks; stateless.
    hw_classification = HardwareClassification.ACCELERATOR

    def _check_pruner_prepared(self, model, pruner, device):
        # [FUNCTION DOC] HELPER: for each pruner group asserts the module weight is on the
        # device type, its fqn is in pruner.state, and the module is parametrized with a
        # FakeStructuredSparsity as first weight parametrization.
        for config in pruner.groups:
            module = config["module"]
            if module.weight.device.type != device.type:
                raise AssertionError(
                    f"Expected device {device.type}, got {module.weight.device.type}"
                )
            # Check mask exists
            if config["tensor_fqn"] not in pruner.state:
                raise AssertionError("Expected tensor_fqn in pruner.state")
            # Check parametrization exists and is correct
            if not parametrize.is_parametrized(module):
                raise AssertionError("Expected module to be parametrized")
            if not hasattr(module, "parametrizations"):
                raise AssertionError(
                    "Expected module to have parametrizations attribute"
                )
            # Assume that this is the 1st/only parametrization
            if type(module.parametrizations.weight[0]) is not FakeStructuredSparsity:
                raise AssertionError(
                    f"Expected FakeStructuredSparsity, got {type(module.parametrizations.weight[0])}"
                )

    def _check_pruner_valid_before_step(self, model, pruner, device):
        # [FUNCTION DOC] HELPER: for each group (module or tuple of modules) asserts the
        # weight is on the device type and the weight mask has boolean dtype.
        for config in pruner.groups:
            modules = []
            if type(config["module"]) is tuple:
                modules.extend(config["module"])
            else:
                module = config["module"]
                modules.append(module)
            for module in modules:
                if module.weight.device.type != device.type:
                    raise AssertionError(
                        f"Expected device {device.type}, got {module.weight.device.type}"
                    )
                if module.parametrizations.weight[0].mask.dtype != torch.bool:
                    raise AssertionError(
                        f"Expected mask dtype torch.bool, got {module.parametrizations.weight[0].mask.dtype}"
                    )

    def _check_pruner_valid_after_step(self, model, pruner, mask, device):
        # [FUNCTION DOC] HELPER: for each group module asserts device placement and that the
        # count of nonzero mask entries equals the mask size minus the argument named mask
        # (the expected number of pruned entries).
        for config in pruner.groups:
            modules = []
            if type(config["module"]) is tuple:
                modules.extend(config["module"])
            else:
                module = config["module"]
                modules.append(module)
            for module in modules:
                if module.weight.device.type != device.type:
                    raise AssertionError(
                        f"Expected device {device.type}, got {module.weight.device.type}"
                    )
                total = module.parametrizations.weight[0].mask.numel()
                actual_nonzero = module.parametrizations.weight[0].mask.count_nonzero()
                expected_nonzero = total - mask
                if actual_nonzero != expected_nonzero:
                    raise AssertionError(
                        f"Expected {expected_nonzero} nonzero elements, got {actual_nonzero}"
                    )

    def _test_constructor_on_device(self, model, device):
        # [FUNCTION DOC] HELPER: asserts the abstract BaseStructuredSparsifier cannot be
        # built (TypeError about update_mask); prepares copies of the model on the device
        # with no config (5 groups) and with one config entry (1 group carrying defaults).
        self.assertRaisesRegex(
            TypeError,
            "BaseStructuredSparsifier.*update_mask",
            BaseStructuredSparsifier,
        )
        model1 = copy.deepcopy(model).to(device)
        pruner = SimplePruner(None)
        pruner.prepare(model1, None)
        pruner.enable_mask_update = True
        for g in pruner.groups:
            module = g["module"]
            if module.weight.device.type != device.type:
                raise AssertionError(
                    f"Expected device {device.type}, got {module.weight.device.type}"
                )
        if len(pruner.groups) != 5:
            raise AssertionError(f"Expected 5 groups, got {len(pruner.groups)}")
        pruner.step()
        # Can instantiate the model with configs
        model2 = copy.deepcopy(model).to(device)
        pruner = SimplePruner({"test": 3})
        pruner.prepare(model2, [{"tensor_fqn": "seq.0.weight"}])
        if len(pruner.groups) != 1:
            raise AssertionError(f"Expected 1 group, got {len(pruner.groups)}")
        if pruner.groups[0]["module_fqn"] != "seq.0":
            raise AssertionError(
                f"Expected module_fqn 'seq.0', got {pruner.groups[0]['module_fqn']}"
            )
        if "test" not in pruner.groups[0]:
            raise AssertionError("Expected 'test' key in pruner.groups[0]")
        if pruner.groups[0]["test"] != 3:
            raise AssertionError(
                f"Expected test value 3, got {pruner.groups[0]['test']}"
            )

    def test_constructor(self, device):
        # [FUNCTION DOC] Runs the constructor checks on SimpleLinear for the given device.
        model = SimpleLinear()
        self._test_constructor_on_device(model, torch.device(device))

    def _test_prepare_linear_on_device(self, model, device):
        # [FUNCTION DOC] HELPER: prepares a copy of the model with SimplePruner, checks the
        # prepared state, and asserts a forward pass on a 128x7 input gives shape (128, 10).
        model = copy.deepcopy(model).to(device)
        x = torch.ones(128, 7, device=device)
        pruner = SimplePruner(None)
        pruner.prepare(model, None)
        self._check_pruner_prepared(model, pruner, device)
        if model(x).shape != (128, 10):
            raise AssertionError(f"Expected shape (128, 10), got {model(x).shape}")

    def test_prepare_linear(self, device):
        # [FUNCTION DOC] Runs the prepare check over four Linear model variants (with and
        # without bias, activation module and functional activation).
        models = [
            SimpleLinear(),
            LinearBias(),
            LinearActivation(),
            LinearActivationFunctional(),
        ]  # without and with bias
        for model in models:
            self._test_prepare_linear_on_device(model, torch.device(device))

    def _test_prepare_conv2d_on_device(self, model, expected_shape, config, device):
        # [FUNCTION DOC] HELPER: prepares the model with SimplePruner and the given config,
        # checks prepared state, and asserts the output shape for a 1x1x28x28 input.
        x = torch.ones((1, 1, 28, 28), device=device)
        pruner = SimplePruner(None)
        pruner.prepare(model, config)
        self._check_pruner_prepared(model, pruner, device)
        if model(x).shape != expected_shape:
            raise AssertionError(
                f"Expected shape {expected_shape}, got {model(x).shape}"
            )

    def test_prepare_conv2d(self, device):
        # [FUNCTION DOC] Runs the conv prepare check over five Conv2d model variants with
        # their expected output shapes and no explicit config.
        models = [
            SimpleConv2d(),
            Conv2dBias(),
            Conv2dActivation(),
            Conv2dPadBias(),
            Conv2dPool(),
        ]
        shapes = [
            (1, 52, 20, 20),
            (1, 52, 18, 18),
            (1, 52, 18, 18),
            (1, 52, 24, 24),
            (1, 52, 3, 3),
        ]
        configs = [None, None, None, None, None]
        for model, shape, config in zip(models, shapes, configs):
            model = model.to(device)
            self._test_prepare_conv2d_on_device(
                model, shape, config, torch.device(device)
            )

    def _test_step_linear_on_device(self, model, device):
        # [FUNCTION DOC] HELPER: prepares with SimplePruner, enables mask updates, checks
        # pre-step validity, steps, then checks exactly one mask entry was pruned.
        model = model.to(device)
        pruner = SimplePruner(None)
        pruner.prepare(model, None)
        pruner.enable_mask_update = True
        self._check_pruner_valid_before_step(model, pruner, device)
        pruner.step()
        self._check_pruner_valid_after_step(model, pruner, 1, device)

    def test_step_linear(self, device):
        # [FUNCTION DOC] Runs the step check over the four Linear model variants.
        models = [
            SimpleLinear(),
            LinearBias(),
            LinearActivation(),
            LinearActivationFunctional(),
        ]
        for model in models:
            self._test_step_linear_on_device(model, torch.device(device))

    def _test_step_conv2d_on_device(self, model, expected_shape, config, device):
        # [FUNCTION DOC] HELPER: like the linear step check but for conv models; also
        # asserts the (unpruned, masked) model output shape on a 1x1x28x28 input.
        model = model.to(device)
        x = torch.ones((1, 1, 28, 28), device=device)
        pruner = SimplePruner(None)
        pruner.prepare(model, config)
        pruner.enable_mask_update = True
        self._check_pruner_valid_before_step(model, pruner, device)
        pruner.step()
        self._check_pruner_valid_after_step(model, pruner, 1, device)
        if model(x).shape != expected_shape:
            raise AssertionError(
                f"Expected shape {expected_shape}, got {model(x).shape}"
            )

    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    def test_step_conv2d(self, device):
        # [FUNCTION DOC] Runs the conv step check over five Conv2d variants with expected
        # output shapes; skipped under TorchDynamo.
        models = [
            SimpleConv2d(),
            Conv2dBias(),
            Conv2dActivation(),
            Conv2dPadBias(),
            Conv2dPool(),
        ]
        shapes = [
            (1, 52, 20, 20),
            (1, 52, 18, 18),
            (1, 52, 18, 18),
            (1, 52, 24, 24),
            (1, 52, 3, 3),
        ]
        configs = [None, None, None, None, None]
        for model, shape, config in zip(models, shapes, configs):
            self._test_step_conv2d_on_device(model, shape, config, torch.device(device))

    def _check_pruner_pruned(self, model, pruner, device):
        # [FUNCTION DOC] HELPER: asserts no group module still has parametrizations or a
        # mask attribute after prune().
        for config in pruner.groups:
            module = config["module"]
            if hasattr(module, "parametrizations"):
                raise AssertionError(
                    "Module should not have parametrizations after pruning"
                )
            if hasattr(module, "mask"):
                raise AssertionError("Module should not have mask after pruning")

    def _test_linear_on_device(
        self, model, config, expected_shape, device, also_prune_bias
    ):
        # [FUNCTION DOC] HELPER: runs ImplementedPruner (optionally pruning biases) on a
        # Linear model: prepare, step, record the masked output, prune, then assert pruned
        # output shape, that parametrizations are gone, that outputs are close when shapes
        # match, and that the parameter count strictly decreased.
        model = model.to(device)
        model.eval()
        num_original_params = sum(p.numel() for p in model.parameters())
        x = torch.ones(128, 7, device=device)

        pruner = ImplementedPruner({"prune_bias": also_prune_bias})
        pruner.prepare(model, config)
        pruner.enable_mask_update = True
        pruner.step()

        y_expected = model(x)

        if y_expected.shape != (128, 10):
            raise AssertionError(f"Expected shape (128, 10), got {y_expected.shape}")
        self._check_pruner_prepared(model, pruner, device)

        # Pruning step
        pruned = pruner.prune()
        y_pruned = pruned(x)
        num_pruned_params = sum(p.numel() for p in pruned.parameters())

        if y_pruned.shape != expected_shape:
            raise AssertionError(
                f"Expected shape {expected_shape}, got {y_pruned.shape}"
            )
        self._check_pruner_pruned(model, pruner, device)
        if y_pruned.shape == y_expected.shape:
            if not torch.isclose(y_expected, y_pruned, rtol=1e-05, atol=1e-07).all():
                raise AssertionError("Expected and pruned outputs are not close")
            if num_pruned_params >= num_original_params:
                raise AssertionError(
                    f"Expected pruned params ({num_pruned_params}) < original ({num_original_params})"
                )

    def test_prune_linear_linear(self, device):
        # [FUNCTION DOC] Prunes chained Linear layers in SimpleLinear using three config
        # subsets, with and without bias pruning, expecting output shape (128, 10).
        r"""test pruning linear-> linear modules"""
        configs, shapes = [], []
        configs.append(
            [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.1.weight"},
                {"tensor_fqn": "seq.2.weight"},
            ]
        )
        shapes.append((128, 10))

        configs.append(
            [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.1.weight"},
                {"tensor_fqn": "seq.2.weight"},
                {"tensor_fqn": "linear1.weight"},
            ]
        )
        shapes.append((128, 10))

        configs.append(
            [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.2.weight"},
            ]
        )
        shapes.append((128, 10))
        for also_prune_bias in [True, False]:
            for config, shape in zip(configs, shapes):
                self._test_linear_on_device(
                    SimpleLinear(),
                    config,
                    shape,
                    torch.device(device),
                    also_prune_bias,
                )

    def test_prune_linear_bias_linear(self, device):
        # [FUNCTION DOC] Prunes LinearBias across bias/no-bias layer pairings (three config
        # cases), with and without bias pruning, expecting output shape (128, 10).
        # linear(bias) -> linear(no bias)
        configs, shapes = [], []
        configs.append(
            [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.1.weight"},
            ]
        )
        shapes.append((128, 10))

        # linear(bias) -> linear(bias)
        configs.append(
            [
                {"tensor_fqn": "seq.2.weight"},
                {"tensor_fqn": "seq.3.weight"},
            ]
        )
        shapes.append((128, 10))

        # linear(no bias) -> linear(bias)
        configs.append(
            [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.1.weight"},
                {"tensor_fqn": "seq.2.weight"},
            ]
        )
        shapes.append((128, 10))

        for also_prune_bias in [True, False]:
            for config, shape in zip(configs, shapes):
                self._test_linear_on_device(
                    LinearBias(),
                    config,
                    shape,
                    torch.device(device),
                    also_prune_bias,
                )

    def test_prune_linear_activation_linear(self, device):
        # [FUNCTION DOC] Prunes Linear-activation-Linear models, both module and functional
        # activation variants, with and without bias pruning; output shape (128, 10).
        config = [
            {"tensor_fqn": "seq.0.weight"},
            {"tensor_fqn": "seq.2.weight"},
            {"tensor_fqn": "seq.4.weight"},
            {"tensor_fqn": "linear1.weight"},
        ]
        shape = (128, 10)

        for also_prune_bias in [True, False]:
            # test version with nn.Modules
            self._test_linear_on_device(
                LinearActivation(),
                config,
                shape,
                torch.device(device),
                also_prune_bias,
            )
            # test functional version
            self._test_linear_on_device(
                LinearActivationFunctional(),
                config,
                shape,
                torch.device(device),
                also_prune_bias,
            )

    def _test_conv2d_on_device(
        self, model, config, x, expected_shape, device, also_prune_bias
    ):
        # [FUNCTION DOC] HELPER: runs ImplementedPruner on a conv model: prepare, step,
        # check the masked output shape, prune, then assert the pruned output shape,
        # removed parametrizations, closeness within 1e-3 and a non-increasing parameter
        # count (equal is allowed when padding blocks pruning).
        model = model.to(device)
        num_original_params = sum(p.numel() for p in model.parameters())
        model.eval()

        pruner = ImplementedPruner({"prune_bias": also_prune_bias})
        pruner.prepare(model, config)
        pruner.enable_mask_update = True
        pruner.step()

        y_expected = model(x)
        if y_expected.shape != expected_shape:
            raise AssertionError(
                f"Expected shape {expected_shape}, got {y_expected.shape}"
            )

        self._check_pruner_prepared(model, pruner, device)

        # Fusion step
        pruned = pruner.prune()
        y_pruned = pruned(x)
        num_pruned_params = sum(p.numel() for p in pruned.parameters())

        if y_pruned.shape != expected_shape:
            raise AssertionError(
                f"Expected shape {expected_shape}, got {y_pruned.shape}"
            )
        self._check_pruner_pruned(model, pruner, device)
        if y_pruned.shape == y_expected.shape:
            # TODO This rtol is a little high, need to double check if something specific is causing this to fail
            if not torch.isclose(y_expected, y_pruned, rtol=1e-3, atol=1e-3).all():
                raise AssertionError(f"fail for {type(model)}")
            # only time this should be equal is when all layers have padding and we can't prune
            if num_pruned_params > num_original_params:
                raise AssertionError(
                    f"Expected pruned params ({num_pruned_params}) <= original ({num_original_params})"
                )

    def test_prune_conv2d_conv2d(self, device):
        # [FUNCTION DOC] Prunes SimpleConv2d within and across sequential blocks, with and
        # without bias pruning, expecting output shape (1, 52, 20, 20).
        configs, shapes = [], []
        # all within sequential blocks
        configs.append(
            [
                {"tensor_fqn": "seq.0.weight"},
            ]
        )
        shapes.append((1, 52, 20, 20))
        # prune across sequential blocks
        configs.append(
            [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.1.weight"},
                {"tensor_fqn": "conv2d1.weight"},
            ]
        )
        shapes.append((1, 52, 20, 20))

        x = torch.ones((1, 1, 28, 28), device=device)
        for also_prune_bias in [True, False]:
            for config, shape in zip(configs, shapes):
                self._test_conv2d_on_device(
                    SimpleConv2d(),
                    config,
                    x,
                    shape,
                    torch.device(device),
                    also_prune_bias,
                )

    def test_prune_conv2d_bias_conv2d(self, device):
        # [FUNCTION DOC] Prunes Conv2dBias across bias/no-bias layer pairings, with and
        # without bias pruning, expecting output shape (1, 52, 18, 18).
        # Conv2d with Bias and no Activation
        configs, shapes = [], []
        # conv2d(bias) -> conv2d(bias)
        configs.append(
            [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.1.weight"},
            ]
        )
        shapes.append((1, 52, 18, 18))

        # conv2d(no bias) -> conv2d(bias)
        configs.append(
            [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.1.weight"},
                {"tensor_fqn": "conv2d1.weight"},
            ]
        )
        shapes.append((1, 52, 18, 18))

        # conv2d(bias) -> conv2d(no bias)
        configs.append(
            [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.1.weight"},
                {"tensor_fqn": "seq.2.weight"},
            ]
        )
        shapes.append((1, 52, 18, 18))

        x = torch.ones((1, 1, 28, 28), device=device)
        for also_prune_bias in [True, False]:
            for config, shape in zip(configs, shapes):
                self._test_conv2d_on_device(
                    Conv2dBias(),
                    config,
                    x,
                    shape,
                    torch.device(device),
                    also_prune_bias,
                )

    def test_prune_conv2d_activation_conv2d(self, device):
        # [FUNCTION DOC] Prunes Conv2dActivation across four bias/activation pairings, with
        # and without bias pruning, expecting output shape (1, 52, 18, 18).
        # Conv2d with Activation and no Bias
        configs, shapes = [], []

        # conv2d(no bias) -> activation -> conv2d(no bias)
        configs.append(
            [
                {"tensor_fqn": "seq.4.weight"},
            ]
        )
        shapes.append((1, 52, 18, 18))

        # conv2d(bias) -> activation -> conv2d(bias)
        configs.append(
            [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.2.weight"},
            ]
        )
        shapes.append((1, 52, 18, 18))

        # conv2d(bias) -> activation -> conv2d(no bias)
        configs.append(
            [
                {"tensor_fqn": "seq.2.weight"},
                {"tensor_fqn": "seq.4.weight"},
            ]
        )
        shapes.append((1, 52, 18, 18))

        # conv2d(no bias) -> activation -> conv2d(bias)
        configs.append(
            [
                {"tensor_fqn": "conv2d1.weight"},
            ]
        )
        shapes.append((1, 52, 18, 18))

        x = torch.ones((1, 1, 28, 28), device=device)
        for also_prune_bias in [True, False]:
            for config, shape in zip(configs, shapes):
                self._test_conv2d_on_device(
                    Conv2dActivation(),
                    config,
                    x,
                    shape,
                    torch.device(device),
                    also_prune_bias,
                )

    def test_prune_conv2d_padding_conv2d(self, device):
        # [FUNCTION DOC] Prunes Conv2dPadBias for five padded/unpadded layer pairings, with
        # and without bias pruning, expecting output shape (1, 52, 24, 24).
        # Conv2d with Padded layers after Bias layers
        configs, shapes = [], []

        # conv(padded, bias) -> conv(padded, bias)
        configs.append(
            [
                {"tensor_fqn": "seq.4.weight"},
            ]
        )
        shapes.append((1, 52, 24, 24))

        # conv(no bias, no pad) -> conv(padded, bias)
        configs.append(
            [
                {"tensor_fqn": "seq.2.weight"},
            ]
        )
        shapes.append((1, 52, 24, 24))

        # conv(padded, bias) -> conv ( no bias ,no pad)
        configs.append(
            [
                {"tensor_fqn": "seq.0.weight"},
            ]
        )
        shapes.append((1, 52, 24, 24))
        # conv(pad, bias) -> conv(no pad, bias)
        configs.append(
            [
                {"tensor_fqn": "seq.6.weight"},
            ]
        )
        shapes.append((1, 52, 24, 24))
        # conv(no pad, bias) -> conv(pad, bias)
        configs.append(
            [
                {"tensor_fqn": "seq.8.weight"},
            ]
        )
        shapes.append((1, 52, 24, 24))

        x = torch.ones((1, 1, 28, 28), device=device)
        for also_prune_bias in [True, False]:
            for config, shape in zip(configs, shapes):
                self._test_conv2d_on_device(
                    Conv2dPadBias(),
                    config,
                    x,
                    shape,
                    torch.device(device),
                    also_prune_bias,
                )

    def test_prune_conv2d_pool_conv2d(self, device):
        # [FUNCTION DOC] Prunes Conv2dPool (convs separated by pooling), with and without
        # bias pruning, expecting output shape (1, 52, 3, 3).
        # Conv2d with Pooling layers
        config = [
            {"tensor_fqn": "seq.0.weight"},
            {"tensor_fqn": "seq.3.weight"},
            {"tensor_fqn": "conv2d1.weight"},
            {"tensor_fqn": "conv2d2.weight"},
        ]
        shape = (1, 52, 3, 3)

        x = torch.ones((1, 1, 28, 28), device=device)
        for also_prune_bias in [True, False]:
            self._test_conv2d_on_device(
                Conv2dPool(),
                config,
                x,
                shape,
                torch.device(device),
                also_prune_bias,
            )

    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    def test_complex_conv2d(self, device):
        # [FUNCTION DOC] Prunes Conv2d-Pool-Flatten-Linear models (functional and module
        # flatten variants, with a skip-add) with and without bias pruning, expecting
        # output shape (1, 13); skipped under TorchDynamo.
        """Test fusion for models that contain Conv2d & Linear modules.
        Currently supports: Conv2d-Pool2d-Flatten-Linear, Skip-add"""
        config = [
            {"tensor_fqn": "seq.0.weight"},
            {"tensor_fqn": "seq.3.weight"},
            {"tensor_fqn": "conv2d1.weight"},
            {"tensor_fqn": "conv2d2.weight"},
        ]
        shape = (1, 13)

        x = torch.ones((1, 1, 28, 28), device=device)
        for also_prune_bias in [True, False]:
            self._test_conv2d_on_device(
                Conv2dPoolFlattenFunctional(),
                config,
                x,
                shape,
                torch.device(device),
                also_prune_bias,
            )
            self._test_conv2d_on_device(
                Conv2dPoolFlatten(),
                config,
                x,
                shape,
                torch.device(device),
                also_prune_bias,
            )


class TestFPGMPruner(TestCase):
    # [CLASS DOC] Tests FPGMPruner (filter pruning via geometric median) on a hand-weighted
    # two-conv model; checks the pairwise-distance helper and mask/prune results.
    """
    Test case for the implementation of paper:
    `Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration <https://arxiv.org/abs/1811.00250>`_.
    """

    class SimpleConvFPGM(nn.Module):
        # [CLASS DOC] HELPER: two stacked bias-free 3x3 convs (1->3 and 3->4 channels) whose
        # filters are filled with constant values (3.0, 2.0, 0.1 and 6.0, 7.0, 0.4, 0.5) so
        # geometric-median pruning differs from norm-based pruning.
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: builds conv2d1 and conv2d2 and overwrites their
            # weights in place with per-filter constants.
            super().__init__()
            self.conv2d1 = nn.Conv2d(
                in_channels=1, out_channels=3, kernel_size=3, padding=1, bias=False
            )
            # Manually set the filter weights for demonstration purposes
            """
            Three filters' weight are manually set to values 3.0, 2.0, and 0.1.
            Different from the norm-based decision that prunes filter with value 0.1,
            FPGM will prune the one with value 2.0.
            """
            weights = torch.tensor([3.0, 2.0, 0.1])  # Weight weights for each filter
            weights = weights[:, None, None, None]  # broadcasting
            self.conv2d1.weight.data.copy_(
                torch.ones(self.conv2d1.weight.shape) * weights
            )

            # Second Convolutional Layer
            self.conv2d2 = nn.Conv2d(
                in_channels=3, out_channels=4, kernel_size=3, padding=1, bias=False
            )
            weights = torch.tensor([6.0, 7.0, 0.4, 0.5])
            weights = weights[:, None, None, None]
            self.conv2d2.weight.data.copy_(
                torch.ones(self.conv2d2.weight.shape) * weights
            )

        def forward(self, x):
            # [FUNCTION DOC] HELPER: applies conv2d1 then conv2d2 and returns the result.
            x = self.conv2d1(x)
            x = self.conv2d2(x)
            return x

    @onlyCPU
    def test_compute_distance(self, device):
        # [FUNCTION DOC] CPU only. Asserts FPGMPruner's private distance computation for the
        # first conv equals, per filter, the sum of L2 distances to all filters (from cdist).
        """Test the distance computation function"""
        model = TestFPGMPruner.SimpleConvFPGM().to(device)
        pruner = FPGMPruner(0.3)
        dist_conv1 = pruner._compute_distance(model.conv2d1.weight)

        # compute the distance matrix using torch.cdist
        flattened_filters = torch.tensor(
            [
                [
                    3.0000,
                    3.0000,
                    3.0000,
                    3.0000,
                    3.0000,
                    3.0000,
                    3.0000,
                    3.0000,
                    3.0000,
                ],
                [
                    2.0000,
                    2.0000,
                    2.0000,
                    2.0000,
                    2.0000,
                    2.0000,
                    2.0000,
                    2.0000,
                    2.0000,
                ],
                [
                    0.1000,
                    0.1000,
                    0.1000,
                    0.1000,
                    0.1000,
                    0.1000,
                    0.1000,
                    0.1000,
                    0.1000,
                ],
            ],
            device=device,
        )

        """
        Expected distance matrix should have the following values:
            [0.0000, 3.0000, 8.7000],
            [3.0000, 0.0000, 5.7000],
            [8.7000, 5.7000, 0.0000],
        the distance should therefore be:
            [11.7000, 8.7000, 14.4000]
        """
        expected_dist_matrix_conv1 = torch.cdist(
            flattened_filters, flattened_filters, p=2
        )
        expected_dist_conv1 = torch.sum(torch.abs(expected_dist_matrix_conv1), 1)
        if not torch.isclose(
            dist_conv1, expected_dist_conv1, rtol=1e-05, atol=1e-07
        ).all():
            raise AssertionError("Distance computation does not match expected")

    def _test_update_mask_on_single_layer(self, expected_conv1, device):
        # [FUNCTION DOC] HELPER: prunes only conv2d1 at sparsity 0.3, asserts the least-norm
        # filter is not the one masked, then after prune() asserts output shape
        # (1, 4, 32, 32), conv2d1 weights match expected_conv1, and conv2d2 lost an input
        # channel (shape (4, 2, 3, 3)).
        # KNOWN ISSUE: the mask check compares an item to False with an identity test, which
        # is only meaningful because item returns a Python bool.
        """Test that pruning is conducted based on the pair-wise distance measurement instead of absolute norm value"""
        # test pruning with one layer of conv2d
        model = TestFPGMPruner.SimpleConvFPGM().to(device)
        x = torch.ones((1, 1, 32, 32), device=device)
        pruner = FPGMPruner(0.3)
        config = [{"tensor_fqn": "conv2d1.weight"}]
        pruner.prepare(model, config)
        pruner.enable_mask_update = True
        pruner.step()
        if (
            pruner.groups[0]["module"].parametrizations.weight[0].mask[-1].item()
            is False
        ):
            raise AssertionError("do not prune the least-norm filter")

        # fusion step
        pruned_model = pruner.prune()

        pruned_y = pruned_model(x)
        # assert shapes
        expected_conv1 = expected_conv1.to(device)
        if pruned_y.shape != (1, 4, 32, 32):
            raise AssertionError(f"Expected shape (1, 4, 32, 32), got {pruned_y.shape}")
        if pruned_model.conv2d1.weight.shape != expected_conv1.shape:
            raise AssertionError(
                f"Expected conv2d1 shape {expected_conv1.shape}, got {pruned_model.conv2d1.weight.shape}"
            )
        if pruned_model.conv2d2.weight.shape != (4, 2, 3, 3):
            raise AssertionError(
                f"conv2d2 should have input channel pruned, got {pruned_model.conv2d2.weight.shape}"
            )
        # assert value
        if not torch.isclose(
            pruned_model.conv2d1.weight, expected_conv1, rtol=1e-05, atol=1e-07
        ).all():
            raise AssertionError("conv2d1 weight does not match expected")

    def _test_update_mask_on_multiple_layer(
        self, expected_conv1, expected_conv2, device
    ):
        # [FUNCTION DOC] HELPER: prunes conv2d1 (default level 0.3) and conv2d2 (level 0.5),
        # then after prune() asserts output shape (1, 2, 32, 32) and that both conv weights
        # match the expected shapes and values.
        # the second setting
        model = TestFPGMPruner.SimpleConvFPGM().to(device)
        x = torch.ones((1, 1, 32, 32), device=device)
        pruner = FPGMPruner(0.3)
        config = [
            {"tensor_fqn": "conv2d1.weight"},
            {"tensor_fqn": "conv2d2.weight", "sparsity_level": 0.5},
        ]
        pruner.prepare(model, config)
        pruner.enable_mask_update = True
        pruner.step()
        # Get the masks for the two least-norm filters
        mask1 = pruner.groups[0]["module"].parametrizations.weight[0].mask[-1]
        mask2 = pruner.groups[0]["module"].parametrizations.weight[0].mask[-2]
        # Check if either of the least-norm filters is not pruned
        if not (mask1.item() is not False or mask2.item() is not False):
            raise AssertionError("Do not prune all least-norm filters")

        # fusion step
        pruned_model = pruner.prune()
        pruned_y = pruned_model(x)
        # assert shapes
        expected_conv1 = expected_conv1.to(device)
        expected_conv2 = expected_conv2.to(device)
        if pruned_y.shape != (1, 2, 32, 32):
            raise AssertionError(f"Expected shape (1, 2, 32, 32), got {pruned_y.shape}")
        if pruned_model.conv2d1.weight.shape != expected_conv1.shape:
            raise AssertionError(
                f"Expected conv2d1 shape {expected_conv1.shape}, got {pruned_model.conv2d1.weight.shape}"
            )
        if pruned_model.conv2d2.weight.shape != expected_conv2.shape:
            raise AssertionError(
                f"Expected conv2d2 shape {expected_conv2.shape}, got {pruned_model.conv2d2.weight.shape}"
            )
        # assert values
        if not torch.isclose(
            pruned_model.conv2d1.weight, expected_conv1, rtol=1e-05, atol=1e-07
        ).all():
            raise AssertionError("conv2d1 weight does not match expected")
        if not torch.isclose(
            pruned_model.conv2d2.weight, expected_conv2, rtol=1e-05, atol=1e-07
        ).all():
            raise AssertionError("conv2d2 weight does not match expected")

    def test_update_mask(self, device):
        # [FUNCTION DOC] Builds expected pruned weights (filters 3.0 and 0.1 kept for conv1,
        # 7.0 and 0.4 kept for conv2) and runs the single-layer and multi-layer checks.
        weights = torch.tensor([3.0, 0.1])
        expected_conv1 = torch.ones((2, 1, 3, 3)) * weights[:, None, None, None]

        weights = torch.tensor([7.0, 0.4])
        expected_conv2 = torch.ones((2, 2, 3, 3)) * weights[:, None, None, None]

        self._test_update_mask_on_single_layer(expected_conv1, device)
        self._test_update_mask_on_multiple_layer(expected_conv1, expected_conv2, device)


if __name__ == "__main__":
    raise_on_run_directly("test/test_ao_sparsity.py")
