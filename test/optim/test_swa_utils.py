# Owner(s): ["module: optimizer"]
# [MODULE DOC] Tests for the weight-averaging utilities (averaged model wrapper with mean or
# exponential averaging, and batch-norm statistics refresh) in torch.optim.swa_utils. The test
# class is instantiated per device by the main optimizer test file; running this file directly
# only prints a note saying so.

import itertools
import pickle

import torch
from torch.optim.swa_utils import (
    AveragedModel,
    get_ema_multi_avg_fn,
    get_swa_multi_avg_fn,
    update_bn,
)
from torch.testing._internal.common_device_type import onlyAccelerator, onlyCPU
from torch.testing._internal.common_utils import (
    HardwareClassification,
    parametrize,
    TestCase,
)


class TestSWAUtils(TestCase):
    # [CLASS DOC] Device-generic tests for averaged models and the batch-norm update helper.
    # Class attribute marks hardware classification as accelerator. Holds two small nested
    # network classes and helpers for averaging runs and batch-norm checks.
    # Instantiated via instantiate_device_type_tests in test/test_optim.py.
    hw_classification = HardwareClassification.ACCELERATOR

    class SWATestDNN(torch.nn.Module):
        # [CLASS DOC] HELPER: Small fully connected network with a linear layer of 100 features
        # followed by a one-dimensional batch norm; exposes the pre-activation computation.
        def __init__(self, input_features):
            # [FUNCTION DOC] HELPER: Records the feature count (100) and creates the linear
            # layer and batch norm.
            super().__init__()
            self.n_features = 100
            self.fc1 = torch.nn.Linear(input_features, self.n_features)
            self.bn = torch.nn.BatchNorm1d(self.n_features)

        def compute_preactivation(self, x):
            # [FUNCTION DOC] HELPER: Returns the linear layer output, i.e. the input of the
            # batch norm.
            return self.fc1(x)

        def forward(self, x):
            # [FUNCTION DOC] HELPER: Applies the linear layer and then the batch norm.
            x = self.fc1(x)
            x = self.bn(x)
            return x

    class SWATestCNN(torch.nn.Module):
        # [CLASS DOC] HELPER: Small convolutional network with a 3x3 convolution of 10 features
        # followed by a two-dimensional batch norm with momentum 0.3; exposes
        # the pre-activation.
        def __init__(self, input_channels):
            # [FUNCTION DOC] HELPER: Records the feature count (10) and creates the convolution
            # (padding one) and the batch norm.
            super().__init__()
            self.n_features = 10
            self.conv1 = torch.nn.Conv2d(
                input_channels, self.n_features, kernel_size=3, padding=1
            )
            self.bn = torch.nn.BatchNorm2d(self.n_features, momentum=0.3)

        def compute_preactivation(self, x):
            # [FUNCTION DOC] HELPER: Returns the convolution output, i.e. the input of the
            # batch norm.
            return self.conv1(x)

        def forward(self, x):
            # [FUNCTION DOC] HELPER: Applies the convolution and then the batch norm.
            x = self.conv1(x)
            x = self.bn(x)
            return x

    def _test_averaged_model(self, net_device, swa_device, ema):
        # [FUNCTION DOC] HELPER: Builds a small convolutional and linear network on the net
        # device, runs the averaging helper with the swa device and the ema flag, then checks
        # each averaged model parameter equals the manually averaged one, that the averaged
        # model parameters live on the swa device while the reference ones live on the net
        # device, and that the update counter lives on the swa device.
        net_device = torch.device(net_device)
        swa_device = torch.device(swa_device)
        dnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 5, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.BatchNorm2d(5, momentum=0.3),
            torch.nn.Conv2d(5, 2, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 10),
        ).to(net_device)

        averaged_params, averaged_dnn = self._run_averaged_steps(dnn, swa_device, ema)

        for p_avg, p_swa in zip(averaged_params, averaged_dnn.parameters()):
            self.assertEqual(p_avg, p_swa)
            # Check that AveragedModel is on the correct device
            self.assertTrue(p_swa.device == swa_device)
            self.assertTrue(p_avg.device == net_device)
        self.assertTrue(averaged_dnn.n_averaged.device == swa_device)

    def _run_averaged_steps(self, dnn, swa_device, ema):
        # [FUNCTION DOC] HELPER: Wraps the network in an averaged model with a multi-tensor
        # average function (exponential with decay 0.999 if the flag is set, otherwise
        # equal-weight mean). Then for ten updates perturbs each parameter in place with random
        # noise, accumulates the expected running average by hand (with decay weights for the
        # exponential case) and updates the averaged model. Returns the hand-computed averages
        # and the averaged model.
        ema_decay = 0.999
        if ema:
            averaged_dnn = AveragedModel(
                dnn, device=swa_device, multi_avg_fn=get_ema_multi_avg_fn(ema_decay)
            )
        else:
            averaged_dnn = AveragedModel(
                dnn, device=swa_device, multi_avg_fn=get_swa_multi_avg_fn()
            )

        averaged_params = [torch.zeros_like(param) for param in dnn.parameters()]

        n_updates = 10
        for i in range(n_updates):
            for p, p_avg in zip(dnn.parameters(), averaged_params):
                p.detach().add_(torch.randn_like(p))
                if ema:
                    p_avg += (
                        p.detach()
                        * ema_decay ** (n_updates - i - 1)
                        * ((1 - ema_decay) if i > 0 else 1.0)
                    )
                else:
                    p_avg += p.detach() / n_updates
            averaged_dnn.update_parameters(dnn)

        return averaged_params, averaged_dnn

    @parametrize("ema", [True, False])
    def test_averaged_model_same_device(self, device, ema):
        # [FUNCTION DOC] Runs the averaged model check with the network and averaged copy both
        # on the test device, for exponential and equal-weight averaging.
        self._test_averaged_model(device, device, ema)

    @onlyAccelerator
    @parametrize("ema", [True, False])
    def test_averaged_model_cross_device(self, device, ema):
        # [FUNCTION DOC] Accelerator-only. Runs the averaged model check with the network on the
        # device and the averaged copy on the CPU, then the reverse, for both averaging modes.
        cpu = torch.device("cpu")
        self._test_averaged_model(device, cpu, ema)
        self._test_averaged_model(cpu, device, ema)

    @onlyAccelerator
    @parametrize("ema", [True, False])
    def test_averaged_model_mixed_device(self, device, ema):
        # [FUNCTION DOC] Accelerator-only. Builds a network whose first layer is on the device
        # and second on the CPU, runs the averaging helper with no target device, and checks
        # each averaged parameter equals the reference and stays on the same device as it.
        dnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 5, kernel_size=3), torch.nn.Linear(5, 10)
        )
        dnn[0].to(device)
        dnn[1].cpu()

        averaged_params, averaged_dnn = self._run_averaged_steps(dnn, None, ema)

        for p_avg, p_swa in zip(averaged_params, averaged_dnn.parameters()):
            self.assertEqual(p_avg, p_swa)
            # Check that AveragedModel is on the correct device
            self.assertTrue(p_avg.device == p_swa.device)

    @onlyCPU
    def test_averaged_model_state_dict(self, device):
        # [FUNCTION DOC] CPU-only. Updates one averaged model ten times with perturbed
        # parameters, loads its state dict into a second averaged model, and checks parameters
        # and the update counter match.
        dnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 5, kernel_size=3), torch.nn.Linear(5, 10)
        )
        averaged_dnn = AveragedModel(dnn)
        averaged_dnn2 = AveragedModel(dnn)
        n_updates = 10
        for _ in range(n_updates):
            for p in dnn.parameters():
                p.detach().add_(torch.randn_like(p))
            averaged_dnn.update_parameters(dnn)
        averaged_dnn2.load_state_dict(averaged_dnn.state_dict())
        for p_swa, p_swa2 in zip(averaged_dnn.parameters(), averaged_dnn2.parameters()):
            self.assertEqual(p_swa, p_swa2)
        self.assertTrue(averaged_dnn.n_averaged == averaged_dnn2.n_averaged)

    @onlyCPU
    def test_averaged_model_default_avg_fn_picklable(self, device):
        # [FUNCTION DOC] CPU-only. Checks that an averaged model with the default average
        # function can be pickled without error.
        dnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 5, kernel_size=3),
            torch.nn.BatchNorm2d(5),
            torch.nn.Linear(5, 5),
        )
        averaged_dnn = AveragedModel(dnn)
        pickle.dumps(averaged_dnn)

    @onlyCPU
    @parametrize("use_multi_avg_fn", [True, False])
    @parametrize("use_buffers", [True, False])
    def test_averaged_model_exponential(self, device, use_multi_avg_fn, use_buffers):
        # [FUNCTION DOC] CPU-only, parametrized over multi-tensor versus single-tensor average
        # function and over whether buffers are averaged. Uses an exponential average with decay
        # 0.9, tracks the expected averages by hand (skipping scalar tensors) over ten perturbed
        # updates, and compares with the averaged model's parameters, plus its buffers (or,
        # without buffer averaging, checks buffers match the source network's buffers).
        # Test AveragedModel with EMA as avg_fn and use_buffers as True.
        dnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 5, kernel_size=3),
            torch.nn.BatchNorm2d(5, momentum=0.3),
            torch.nn.Linear(5, 10),
        )
        decay = 0.9

        if use_multi_avg_fn:
            averaged_dnn = AveragedModel(
                dnn, multi_avg_fn=get_ema_multi_avg_fn(decay), use_buffers=use_buffers
            )
        else:

            def avg_fn(p_avg, p, n_avg):
                # [FUNCTION DOC] HELPER: Exponential moving average of one tensor: decay times
                # the running average plus one minus decay times the new value. The count
                # argument is ignored.
                return decay * p_avg + (1 - decay) * p

            averaged_dnn = AveragedModel(dnn, avg_fn=avg_fn, use_buffers=use_buffers)

        if use_buffers:
            dnn_params = list(itertools.chain(dnn.parameters(), dnn.buffers()))
        else:
            dnn_params = list(dnn.parameters())

        averaged_params = [
            torch.zeros_like(param)
            for param in dnn_params
            if param.size() != torch.Size([])
        ]

        n_updates = 10
        for i in range(n_updates):
            updated_averaged_params = []
            for p, p_avg in zip(dnn_params, averaged_params):
                if p.size() == torch.Size([]):
                    continue
                p.detach().add_(torch.randn_like(p))
                if i == 0:
                    updated_averaged_params.append(p.clone())
                else:
                    updated_averaged_params.append(
                        (p_avg * decay + p * (1 - decay)).clone()
                    )
            averaged_dnn.update_parameters(dnn)
            averaged_params = updated_averaged_params

        if use_buffers:
            for p_avg, p_swa in zip(
                averaged_params,
                itertools.chain(
                    averaged_dnn.module.parameters(), averaged_dnn.module.buffers()
                ),
            ):
                self.assertEqual(p_avg, p_swa)
        else:
            for p_avg, p_swa in zip(averaged_params, averaged_dnn.parameters()):
                self.assertEqual(p_avg, p_swa)
            for b_avg, b_swa in zip(dnn.buffers(), averaged_dnn.module.buffers()):
                self.assertEqual(b_avg, b_swa)

    def _test_update_bn(self, dnn, dl_x, dl_xy, device):
        # [FUNCTION DOC] HELPER: Computes the mean and biased variance of the network's
        # pre-batch-norm activations over all batches of the plain loader. Runs the batch-norm
        # update on the loader of input-target pairs and checks running mean and variance
        # (variance with loose tolerance) match. Then resets running statistics and repeats with
        # the pair loader, and again with the plain loader.
        preactivation_sum = torch.zeros(dnn.n_features, device=device)
        preactivation_squared_sum = torch.zeros(dnn.n_features, device=device)
        total_num = 0
        for x in dl_x:
            x = x[0].to(device)

            dnn.forward(x)
            preactivations = dnn.compute_preactivation(x)
            if len(preactivations.shape) == 4:
                preactivations = preactivations.transpose(1, 3)
            preactivations = preactivations.contiguous().view(-1, dnn.n_features)
            total_num += preactivations.shape[0]

            preactivation_sum += torch.sum(preactivations, dim=0)
            preactivation_squared_sum += torch.sum(preactivations**2, dim=0)

        preactivation_mean = preactivation_sum / total_num
        preactivation_var = preactivation_squared_sum / total_num
        preactivation_var = preactivation_var - preactivation_mean**2

        update_bn(dl_xy, dnn, device=device)
        self.assertEqual(preactivation_mean, dnn.bn.running_mean)
        self.assertEqual(preactivation_var, dnn.bn.running_var, atol=1e-1, rtol=0)

        def _reset_bn(module):
            # [FUNCTION DOC] HELPER: For a batch-norm module, resets running mean to zeros and
            # running variance to ones by assigning new tensors; other modules are
            # left untouched.
            if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
                module.running_mean = torch.zeros_like(module.running_mean)
                module.running_var = torch.ones_like(module.running_var)

        # reset batch norm and run update_bn again
        dnn.apply(_reset_bn)
        update_bn(dl_xy, dnn, device=device)
        self.assertEqual(preactivation_mean, dnn.bn.running_mean)
        self.assertEqual(preactivation_var, dnn.bn.running_var, atol=1e-1, rtol=0)
        # using the dl_x loader instead of dl_xy
        dnn.apply(_reset_bn)
        update_bn(dl_x, dnn, device=device)
        self.assertEqual(preactivation_mean, dnn.bn.running_mean)
        self.assertEqual(preactivation_var, dnn.bn.running_var, atol=1e-1, rtol=0)

    def test_update_bn_dnn(self, device):
        # [FUNCTION DOC] Builds random data and loaders (with and without targets), a fully
        # connected network with batch norm on the device in training mode, runs the shared
        # batch-norm check, and asserts the network is still in training mode afterwards.
        # Test update_bn for a fully-connected network with BatchNorm1d
        objects, input_features = 100, 5
        x = torch.rand(objects, input_features)
        y = torch.rand(objects)
        ds_x = torch.utils.data.TensorDataset(x)
        ds_xy = torch.utils.data.TensorDataset(x, y)
        dl_x = torch.utils.data.DataLoader(ds_x, batch_size=5, shuffle=True)
        dl_xy = torch.utils.data.DataLoader(ds_xy, batch_size=5, shuffle=True)
        dnn = self.SWATestDNN(input_features=input_features).to(device)
        dnn.train()
        self._test_update_bn(dnn, dl_x, dl_xy, device)
        self.assertTrue(dnn.training)

    def test_update_bn_cnn(self, device):
        # [FUNCTION DOC] Same as the fully connected variant but with random image-shaped data
        # and the convolutional network; also asserts training mode is preserved.
        # Test update_bn for convolutional network and BatchNorm2d
        objects = 100
        input_channels = 3
        height, width = 5, 5
        x = torch.rand(objects, input_channels, height, width)
        y = torch.rand(objects)
        ds_x = torch.utils.data.TensorDataset(x)
        ds_xy = torch.utils.data.TensorDataset(x, y)
        dl_x = torch.utils.data.DataLoader(ds_x, batch_size=5, shuffle=True)
        dl_xy = torch.utils.data.DataLoader(ds_xy, batch_size=5, shuffle=True)
        cnn = self.SWATestCNN(input_channels=input_channels).to(device)
        cnn.train()
        self._test_update_bn(cnn, dl_x, dl_xy, device)
        self.assertTrue(cnn.training)

    @onlyCPU
    def test_bn_update_eval_momentum(self, device):
        # [FUNCTION DOC] CPU-only. Puts the convolutional network in evaluation mode, runs the
        # batch-norm update with no device argument, and checks the network is still in
        # evaluation mode and the batch norm momentum is still 0.3.
        # check that update_bn preserves eval mode
        objects = 100
        input_channels = 3
        height, width = 5, 5
        x = torch.rand(objects, input_channels, height, width)
        ds_x = torch.utils.data.TensorDataset(x)
        dl_x = torch.utils.data.DataLoader(ds_x, batch_size=5, shuffle=True)
        cnn = self.SWATestCNN(input_channels=input_channels)
        cnn.eval()
        update_bn(dl_x, cnn)
        self.assertFalse(cnn.training)

        # check that momentum is preserved
        self.assertEqual(cnn.bn.momentum, 0.3)


if __name__ == "__main__":
    print("These tests should be run through test/test_optim.py instead")
