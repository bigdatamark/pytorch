# Owner(s): ["module: optimizer", "module: LrScheduler" ]
# ruff: noqa: F841
# [MODULE DOC] Test module for learning rate schedulers (StepLR, LinearLR, SequentialLR,
# ChainedScheduler, ReduceLROnPlateau, CyclicLR, OneCycleLR, warm restarts, SWALR and others).
# One large TestCase compares per-epoch lrs with expected target lists, covers state dict
# round trips, warnings and error cases. Importing it only defines the class and expands
# parametrized tests; running it directly just prints a hint to use test/test_optim.py.
import copy
import math
import pickle
import tempfile
import types
import warnings
from functools import partial
from unittest import expectedFailure

import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch.optim import Adam, Rprop, SGD
from torch.optim.lr_scheduler import (
    ChainedScheduler,
    ConstantLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    EPOCH_DEPRECATION_WARNING,
    ExponentialLR,
    LambdaLR,
    LinearLR,
    LRScheduler,
    MultiplicativeLR,
    MultiStepLR,
    OneCycleLR,
    PolynomialLR,
    ReduceLROnPlateau,
    SequentialLR,
    StepLR,
)
from torch.optim.swa_utils import SWALR
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    load_tests,
    parametrize,
    skipIfTorchDynamo,
    TestCase,
)


# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests  # noqa: PLW0127


class TestLRScheduler(TestCase):
    # [CLASS DOC] Test case for the learning rate scheduler classes in torch.optim.lr_scheduler
    # plus SWALR. Each test builds scheduler objects over the optimizer self.opt created in setUp
    # (two SGD param groups) and compares the per-epoch learning rates, and for cyclic schedulers
    # the momentum, with hand computed target lists via the private _test helpers. Class
    # attributes: hw_classification is generic and exact_dtype is True. Parametrized tests are
    # expanded by instantiate_parametrized_tests at the bottom of the module.
    hw_classification = HardwareClassification.GENERIC

    class SchedulerTestNet(torch.nn.Module):
        # [CLASS DOC] HELPER: Tiny network with two 1x1 single channel convolutions; its two
        # layers supply the two optimizer param groups used by every test.
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Creates the convolution layers conv1 and conv2.
            super().__init__()
            self.conv1 = torch.nn.Conv2d(1, 1, 1)
            self.conv2 = torch.nn.Conv2d(1, 1, 1)

        def forward(self, x):
            # [FUNCTION DOC] HELPER: Applies conv1, a relu, then conv2 and returns the result.
            return self.conv2(F.relu(self.conv1(x)))

    class LambdaLRTestObject:
        # [CLASS DOC] HELPER: Callable object used as a learning rate lambda; keeps one
        # multiplier in its instance dict so tests can check that LambdaLR state dicts store
        # callable objects (unlike plain functions). Two instances are equal when their instance
        # dicts are equal.
        def __init__(self, value):
            # [FUNCTION DOC] HELPER: Stores the multiplier in the attribute value.
            self.value = value

        def __call__(self, epoch):
            # [FUNCTION DOC] HELPER: Returns the stored value multiplied by the epoch.
            return self.value * epoch

        def __eq__(self, other):
            # [FUNCTION DOC] HELPER: Equality: true only when other is an instance of the same
            # class and the instance attribute dicts match; any other type compares unequal.
            if isinstance(other, self.__class__):
                return self.__dict__ == other.__dict__
            else:
                return False

    exact_dtype = True

    def setUp(self):
        # [FUNCTION DOC] HELPER: Per-test fixture. Builds a SchedulerTestNet in self.net and an
        # SGD optimizer self.opt with two param groups: conv1 params using the default lr 0.05
        # and conv2 params using a tensor lr of 0.5. Note that other helpers call this method
        # again to reset self.opt.
        super().setUp()
        self.net = self.SchedulerTestNet()
        self.opt = SGD(
            [
                {"params": self.net.conv1.parameters()},
                {"params": self.net.conv2.parameters(), "lr": torch.tensor(0.5)},
            ],
            lr=0.05,
        )

    def _check_warning_is_epoch_deprecation_warning(self, w, *, num_warnings: int = 1):
        # [FUNCTION DOC] HELPER: Checks a list w of recorded warnings: its length must equal
        # num_warnings and each warning must carry exactly one argument equal to
        # EPOCH_DEPRECATION_WARNING. Fails the test otherwise. Used after stepping a scheduler
        # with an explicit epoch.
        """This function swallows the epoch deprecation warning which is produced when we
        call `scheduler.step(epoch)` with some not `None` value of `epoch`.
        this is deprecated, and this function will need to be removed/updated when
        the schedulers no longer accept the parameter at all.
        """
        self.assertEqual(len(w), num_warnings)
        for warning in w:
            self.assertEqual(len(warning.message.args), 1)
            self.assertEqual(warning.message.args[0], EPOCH_DEPRECATION_WARNING)

    def test_error_when_getlr_has_epoch(self):
        # [FUNCTION DOC] Defines a local LRScheduler subclass whose get_lr takes an extra step
        # parameter and asserts that constructing it over an SGD optimizer raises TypeError, i.e.
        # get_lr must not accept an epoch argument.
        class MultiStepLR(torch.optim.lr_scheduler.LRScheduler):
            # [CLASS DOC] HELPER: Local scheduler subclass that emulates multi step decay but
            # declares get_lr with an illegal extra parameter; only used to provoke the
            # TypeError.
            def __init__(self, optimizer, gamma, milestones, last_epoch=-1):
                # [FUNCTION DOC] HELPER: Records the initial lr of every param group, the gamma
                # and the milestones, then calls the base constructor.
                self.init_lr = [group["lr"] for group in optimizer.param_groups]
                self.gamma = gamma
                self.milestones = milestones
                super().__init__(optimizer, last_epoch)

            def get_lr(self, step):
                # [FUNCTION DOC] HELPER: Computes decayed lrs from last_epoch and the milestones.
                # Its extra step parameter is what the test expects to be rejected with TypeError
                # during construction.
                global_step = self.last_epoch
                gamma_power = (
                    [0]
                    + [i + 1 for i, m in enumerate(self.milestones) if global_step >= m]
                )[-1]
                return [init_lr * (self.gamma**gamma_power) for init_lr in self.init_lr]

        optimizer = SGD([torch.rand(1)], lr=1)

        with self.assertRaises(TypeError):
            scheduler = MultiStepLR(optimizer, gamma=1, milestones=[10, 20])

    @skipIfTorchDynamo(
        "Torchdynamo keeps references to optim in the guards and the stack of the graph break frames"
    )
    def test_no_cyclic_references(self):
        # [FUNCTION DOC] Skipped under TorchDynamo. Creates a LambdaLR over an SGD optimizer,
        # deletes the scheduler, then asserts nothing refers to the optimizer and that after
        # deleting it a garbage collection pass finds zero unreachable objects, i.e. the
        # scheduler creates no reference cycle.
        import gc

        param = Parameter(torch.empty(10))
        optim = SGD([param], lr=0.5)
        scheduler = LambdaLR(optim, lambda epoch: 1.0)
        del scheduler

        self.assertTrue(
            len(gc.get_referrers(optim)) == 0,
            "Optimizer should contain no cyclic references",
        )

        gc.collect()
        del optim
        self.assertEqual(
            gc.collect(), 0, msg="Optimizer should be garbage-collected on __del__"
        )

    @skipIfTorchDynamo(
        "Torchdynamo keeps references to optim in the guards and the stack of the graph break frames"
    )
    def test_no_cyclic_references_in_step(self):
        # [FUNCTION DOC] Skipped under TorchDynamo. With the garbage collector disabled, runs a
        # local function that does one backward, optimizer step and scheduler step and returns a
        # weak reference to the scheduler; raises AssertionError if the scheduler is still alive
        # afterwards, showing stepping creates no reference cycles. Re-enables gc at the end.
        import gc
        import weakref

        def run():
            # [FUNCTION DOC] HELPER: Builds a parameter, SGD and LambdaLR, performs backward,
            # optimizer step and scheduler step, and returns a weak reference to the scheduler.
            param = torch.empty(10, requires_grad=True)
            optim = SGD(params=[param], lr=0.5)
            scheduler = LambdaLR(optim, lambda epoch: 1.0)
            param.sum().backward()
            optim.step()
            scheduler.step()

            return weakref.ref(scheduler)

        # To ensure that there are no reference cycles in scheduler,
        # we need to turn off the garbage collector. Since gc will
        # automatically collect unreachable objects.
        gc.disable()
        ref = run()

        if ref() is not None:
            raise AssertionError("Expected scheduler to be garbage collected")
        gc.enable()  # restore

    def test_old_pattern_warning(self):
        # [FUNCTION DOC] Creating a StepLR must emit no warning; then calling scheduler step
        # before optimizer step for 35 epochs (the old pattern) must raise a UserWarning matching
        # the how-to-adjust-learning-rate message.
        epochs = 35
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            scheduler = StepLR(self.opt, gamma=0.1, step_size=3)
            self.assertTrue(len(ws) == 0, "No warning should be raised")

        def old_pattern():
            # [FUNCTION DOC] HELPER: Loop of 35 epochs: scheduler step followed by optimizer
            # step; closes over the scheduler and epoch count.
            for _ in range(epochs):
                scheduler.step()
                self.opt.step()

        self.assertWarnsRegex(UserWarning, r"how-to-adjust-learning-rate", old_pattern)

    def test_old_pattern_warning_with_arg(self):
        # [FUNCTION DOC] Meant to cover the old call order with an epoch argument, but behaves
        # exactly like the previous test: StepLR is created with no warning and the old order
        # must warn.
        # KNOWN ISSUE: despite the name no epoch argument is passed, so this duplicates the test
        # above.
        epochs = 35
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            scheduler = StepLR(self.opt, gamma=0.1, step_size=3)
            self.assertTrue(len(ws) == 0, "No warning should be raised")

        def old_pattern2():
            # [FUNCTION DOC] HELPER: Loop of 35 epochs: scheduler step then optimizer step;
            # closes over the scheduler.
            for _ in range(epochs):
                scheduler.step()
                self.opt.step()

        self.assertWarnsRegex(UserWarning, r"how-to-adjust-learning-rate", old_pattern2)

    def test_old_pattern_warning_resuming(self):
        # [FUNCTION DOC] Sets initial_lr to 0.01 on every group, resumes StepLR with last_epoch
        # 10 (no warning allowed at construction), and checks the old call order still emits the
        # how-to-adjust-learning-rate UserWarning.
        epochs = 35
        for group in self.opt.param_groups:
            group["initial_lr"] = 0.01

        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            scheduler = StepLR(self.opt, gamma=0.1, step_size=3, last_epoch=10)
            self.assertTrue(len(ws) == 0, "No warning should be raised")

        def old_pattern():
            # [FUNCTION DOC] HELPER: Loop of 35 epochs: scheduler step then optimizer step.
            for _ in range(epochs):
                scheduler.step()
                self.opt.step()

        self.assertWarnsRegex(UserWarning, r"how-to-adjust-learning-rate", old_pattern)

    def test_old_pattern_warning_resuming_with_arg(self):
        # [FUNCTION DOC] Same scenario as the previous resuming test: initial_lr set, StepLR
        # resumed at epoch 10, old order must warn.
        # KNOWN ISSUE: despite the name no epoch argument is passed, so this duplicates the test
        # above.
        epochs = 35
        for group in self.opt.param_groups:
            group["initial_lr"] = 0.01

        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            scheduler = StepLR(self.opt, gamma=0.1, step_size=3, last_epoch=10)
            self.assertTrue(len(ws) == 0, "No warning should be raised")

        def old_pattern2():
            # [FUNCTION DOC] HELPER: Loop of 35 epochs: scheduler step then optimizer step.
            for _ in range(epochs):
                scheduler.step()
                self.opt.step()

        self.assertWarnsRegex(UserWarning, r"how-to-adjust-learning-rate", old_pattern2)

    def test_old_pattern_warning_with_overridden_optim_step(self):
        # [FUNCTION DOC] Resumes StepLR at last_epoch 10 after setting initial_lr, then replaces
        # self.opt.step with a bound wrapper that delegates to the original step; the old call
        # order must still emit the how-to-adjust-learning-rate UserWarning. Mutates self.opt.
        epochs = 35
        for group in self.opt.param_groups:
            group["initial_lr"] = 0.01

        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            scheduler = StepLR(self.opt, gamma=0.1, step_size=3, last_epoch=10)
            self.assertTrue(len(ws) == 0, "No warning should be raised")

        # emulate use-case with optimizer.step overridden
        import types

        old_step = self.opt.step

        def new_step(o, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Wrapper installed as the optimizer step: calls the saved
            # original step with the given arguments and returns its result.
            retval = old_step(*args, **kwargs)
            return retval

        self.opt.step = types.MethodType(new_step, self.opt)

        def old_pattern2():
            # [FUNCTION DOC] HELPER: Loop of 35 epochs: scheduler step then optimizer step.
            for _ in range(epochs):
                scheduler.step()
                self.opt.step()

        self.assertWarnsRegex(UserWarning, r"how-to-adjust-learning-rate", old_pattern2)

    def test_new_pattern_no_warning(self):
        # [FUNCTION DOC] Correct call order (optimizer step then scheduler step) for 35 epochs
        # must emit no warnings, both when creating StepLR and during the loop.
        epochs = 35
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            scheduler = StepLR(self.opt, gamma=0.1, step_size=3)
            self.assertTrue(len(ws) == 0, "No warning should be raised")

        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            for _ in range(epochs):
                self.opt.step()
                scheduler.step()
            self.assertTrue(len(ws) == 0, "No warning should be raised")

    def test_new_pattern_no_warning_with_arg(self):
        # [FUNCTION DOC] Same as the previous test: the right call order emits no warnings.
        # KNOWN ISSUE: despite the name no epoch argument is passed, so this duplicates the test
        # above.
        epochs = 35
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            scheduler = StepLR(self.opt, gamma=0.1, step_size=3)
            self.assertTrue(len(ws) == 0, "No warning should be raised")

        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            for _ in range(epochs):
                self.opt.step()
                scheduler.step()
            self.assertTrue(len(ws) == 0, "No warning should be raised")

    def test_new_pattern_no_warning_with_overridden_optim_step(self):
        # [FUNCTION DOC] Right call order but with self.opt.step replaced by a delegating
        # wrapper; stepping the scheduler must warn (UserWarning) that the optimizer step has
        # been overridden. Mutates self.opt.
        epochs = 35
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            scheduler = StepLR(self.opt, gamma=0.1, step_size=3)
            self.assertTrue(len(ws) == 0, "No warning should be raised")

        # emulate use-case with optimizer.step overridden
        import types

        old_step = self.opt.step

        def new_step(o, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Wrapper installed as the optimizer step: delegates to the
            # saved original step and returns its result.
            retval = old_step(*args, **kwargs)
            return retval

        self.opt.step = types.MethodType(new_step, self.opt)

        def new_pattern():
            # [FUNCTION DOC] HELPER: Loop of 35 epochs: optimizer step then scheduler step.
            for _ in range(epochs):
                self.opt.step()
                scheduler.step()

        self.assertWarnsRegex(
            UserWarning, r"`optimizer.step\(\)` has been overridden", new_pattern
        )

    def _test_lr_is_constant_for_constant_epoch(self, scheduler):
        # [FUNCTION DOC] HELPER: For 10 iterations steps the scheduler's optimizer and then the
        # scheduler with the fixed epoch 2 under a warning recorder (checking exactly one epoch
        # deprecation warning), recording group 0 lr; asserts the smallest and largest recorded
        # lr are equal.
        l = []

        for _ in range(10):
            scheduler.optimizer.step()
            with warnings.catch_warnings(record=True) as w:
                scheduler.step(2)
                self._check_warning_is_epoch_deprecation_warning(w)

            l.append(self.opt.param_groups[0]["lr"])
        self.assertEqual(min(l), max(l))

    def test_step_lr_is_constant_for_constant_epoch(self):
        # [FUNCTION DOC] Builds StepLR with step size 2 and checks via the shared helper that
        # repeatedly stepping to the same epoch leaves the lr constant.
        scheduler = StepLR(self.opt, 2)
        self._test_lr_is_constant_for_constant_epoch(scheduler)

    def test_exponential_lr_is_constant_for_constant_epoch(self):
        # [FUNCTION DOC] Builds ExponentialLR with gamma 0.9 and checks via the shared helper
        # that repeatedly stepping to the same epoch leaves the lr constant.
        scheduler = ExponentialLR(self.opt, gamma=0.9)
        self._test_lr_is_constant_for_constant_epoch(scheduler)

    def test_constantlr_is_constant_for_constant_epoch(self):
        # [FUNCTION DOC] Builds ConstantLR with defaults and checks via the shared helper that
        # repeatedly stepping to the same epoch leaves the lr constant.
        scheduler = ConstantLR(self.opt)
        self._test_lr_is_constant_for_constant_epoch(scheduler)

    def test_linear_linearlr_is_constant_for_constant_epoch(self):
        # [FUNCTION DOC] Builds LinearLR with defaults and checks via the shared helper that
        # repeatedly stepping to the same epoch leaves the lr constant.
        scheduler = LinearLR(self.opt)
        self._test_lr_is_constant_for_constant_epoch(scheduler)

    def test_polynomial_lr_is_constant_for_constant_epoch(self):
        # [FUNCTION DOC] Builds PolynomialLR with power 0.9 and checks via the shared helper that
        # repeatedly stepping to the same epoch leaves the lr constant.
        scheduler = PolynomialLR(self.opt, power=0.9)
        self._test_lr_is_constant_for_constant_epoch(scheduler)

    def test_step_lr(self):
        # [FUNCTION DOC] StepLR with gamma 0.1 and step size 3: expects lr 0.05 dropping tenfold
        # every 3 epochs over 10 epochs; the second group (base 0.5) expects ten times those
        # values.
        # lr = 0.05     if epoch < 3
        # lr = 0.005    if 30 <= epoch < 6
        # lr = 0.0005   if epoch >= 9
        epochs = 10
        single_targets = [0.05] * 3 + [0.005] * 3 + [0.0005] * 3 + [0.00005] * 3
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = StepLR(self.opt, gamma=0.1, step_size=3)
        self._test(scheduler, targets, epochs)

    def test_get_last_lr_step_lr(self):
        # [FUNCTION DOC] StepLR with step size 3 and gamma 0.1 on a private SGD optimizer with lr
        # 0.1; get_last_lr must report 0.1, 0.01, 0.001, 0.0001 per three epoch block.
        from torch.nn import Parameter

        epochs = 10
        optimizer = SGD([Parameter(torch.randn(2, 2, requires_grad=True))], 0.1)
        targets = [[0.1] * 3 + [0.01] * 3 + [0.001] * 3 + [0.0001]]
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.1)
        self._test_get_last_lr(scheduler, targets, epochs)

    def test_get_last_lr_multi_step_lr(self):
        # [FUNCTION DOC] MultiStepLR with gamma 0.1 and milestones 2, 5, 9: get_last_lr must
        # follow the decayed targets for 10 epochs for both groups.
        # lr = 0.05     if epoch < 2
        # lr = 0.005    if 2 <= epoch < 5
        # lr = 0.0005   if 5 <= epoch < 9
        # lr = 0.00005   if 9 <= epoch
        epochs = 10
        single_targets = [0.05] * 2 + [0.005] * 3 + [0.0005] * 4 + [0.00005] * 1
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        self._test_get_last_lr(scheduler, targets, epochs)

    def test_raise_error_when_last_epoch_is_greater_than_0_and_initial_lr_is_not_specified(
        self,
    ):
        # [FUNCTION DOC] Resuming StepLR with last_epoch 1 on an optimizer lacking initial_lr
        # must raise KeyError whose message says initial_lr is not specified in the param group.
        optimizer = SGD([Parameter(torch.randn(2, 2, requires_grad=True))], 0.1)
        with self.assertRaisesRegex(
            KeyError,
            r"param \'initial_lr\' is not specified in param_groups\[0\] when resuming scheduler with last_epoch >= 0",
        ):
            StepLR(optimizer, step_size=3, gamma=0.1, last_epoch=1)

    def test_multi_step_lr(self):
        # [FUNCTION DOC] MultiStepLR with gamma 0.1 and milestones 2, 5, 9 over 10 epochs; lr is
        # checked before each scheduler step against hand computed targets.
        # lr = 0.05     if epoch < 2
        # lr = 0.005    if 2 <= epoch < 5
        # lr = 0.0005   if epoch < 9
        # lr = 0.00005   if epoch >= 9
        epochs = 10
        single_targets = [0.05] * 2 + [0.005] * 3 + [0.0005] * 4 + [0.00005] * 3
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        self._test(scheduler, targets, epochs)

    def test_multi_step_lr_with_epoch(self):
        # [FUNCTION DOC] Same targets as the plain multi step test, but each scheduler step is
        # given the epoch explicitly (deprecated usage), checking the deprecation warning too.
        # lr = 0.05     if epoch < 2
        # lr = 0.005    if 2 <= epoch < 5
        # lr = 0.0005   if epoch < 9
        # lr = 0.00005   if epoch >= 9
        epochs = 10
        single_targets = [0.05] * 2 + [0.005] * 3 + [0.0005] * 4 + [0.00005] * 3
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        self._test_with_epoch(scheduler, targets, epochs)

    def test_get_last_lr_constantlr(self):
        # [FUNCTION DOC] ConstantLR with factor one half for 5 iterations: get_last_lr must give
        # 0.025 for 5 epochs, then 0.05.
        # lr = 0.025     if epoch < 5
        # lr = 0.005    if 5 <= epoch
        epochs = 10
        single_targets = [0.025] * 5 + [0.05] * 5
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = ConstantLR(self.opt, factor=1.0 / 2, total_iters=5)
        self._test_get_last_lr(scheduler, targets, epochs)

    def test_get_last_lr_linearlr(self):
        # [FUNCTION DOC] LinearLR from factor 0.25 to 0.6 over 4 iterations: get_last_lr must
        # follow the linear interpolation and then stay at 0.6 of the base lr.
        # lr = 0.025     if epoch == 0
        # lr = 0.03125   if epoch == 1
        # lr = 0.0375    if epoch == 2
        # lr = 0.04375   if epoch == 3
        # lr = 0.005     if 4 <= epoch
        epochs = 10
        start_factor = 1.0 / 4
        end_factor = 3.0 / 5
        iters = 4
        interpolation = [
            start_factor + i * (end_factor - start_factor) / iters for i in range(iters)
        ]
        single_targets = [x * 0.05 for x in interpolation] + [0.05 * end_factor] * (
            epochs - iters
        )
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = LinearLR(
            self.opt,
            start_factor=start_factor,
            end_factor=end_factor,
            total_iters=iters,
        )
        self._test_get_last_lr(scheduler, targets, epochs)

    def test_constantlr(self):
        # [FUNCTION DOC] ConstantLR factor one half for 5 iterations: lr is 0.025 then 0.05.
        # lr = 0.025     if epoch < 5
        # lr = 0.005    if 5 <= epoch
        epochs = 10
        single_targets = [0.025] * 5 + [0.05] * 5
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = ConstantLR(self.opt, factor=1.0 / 2, total_iters=5)
        self._test(scheduler, targets, epochs)

    def test_linearlr(self):
        # [FUNCTION DOC] LinearLR from factor 0.5 over 4 iterations: lr rises linearly to the
        # base lr then stays there.
        # lr = 0.025     if epoch == 0
        # lr = 0.03125   if epoch == 1
        # lr = 0.0375    if epoch == 2
        # lr = 0.04375   if epoch == 3
        # lr = 0.005     if 4 <= epoch
        epochs = 10
        start_factor = 1.0 / 2
        iters = 4
        interpolation = [
            start_factor + i * (1 - start_factor) / iters for i in range(iters)
        ]
        single_targets = [x * 0.05 for x in interpolation] + [0.05] * (epochs - iters)
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = LinearLR(self.opt, start_factor=start_factor, total_iters=iters)
        self._test(scheduler, targets, epochs)

    def test_linearlr_start_factor_limits1(self):
        # [FUNCTION DOC] LinearLR with start_factor 0.0 must raise ValueError.
        start_factor = 0.0
        iters = 4
        with self.assertRaises(ValueError):
            LinearLR(self.opt, start_factor=start_factor, total_iters=iters)

    def test_linearlr_start_factor_limits2(self):
        # [FUNCTION DOC] LinearLR with start_factor 1.1 must raise ValueError.
        start_factor = 1.1
        iters = 4
        with self.assertRaises(ValueError):
            LinearLR(self.opt, start_factor=start_factor, total_iters=iters)

    def test_constantlr_with_epoch(self):
        # [FUNCTION DOC] Like the constant lr test but each scheduler step passes the epoch
        # explicitly, which also checks the epoch deprecation warning.
        # lr = 0.025     if epoch < 5
        # lr = 0.005    if 5 <= epoch
        epochs = 10
        single_targets = [0.025] * 5 + [0.05] * 5
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = ConstantLR(self.opt, factor=1.0 / 2, total_iters=5)
        self._test_with_epoch(scheduler, targets, epochs)

    def test_linearlr_with_epoch(self):
        # [FUNCTION DOC] Like the linear lr test but stepping with explicit epochs; start factor
        # one half, end factor one, 4 iterations.
        # lr = 0.025     if epoch == 0
        # lr = 0.03125   if epoch == 1
        # lr = 0.0375    if epoch == 2
        # lr = 0.04375   if epoch == 3
        # lr = 0.005     if 4 <= epoch
        epochs = 10
        start_factor = 1.0 / 2
        end_factor = 1.0
        iters = 4
        interpolation = [
            start_factor + i * (end_factor - start_factor) / iters for i in range(iters)
        ]
        single_targets = [x * 0.05 for x in interpolation] + [0.05] * (epochs - iters)
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = LinearLR(self.opt, start_factor=start_factor, total_iters=iters)
        self._test_with_epoch(scheduler, targets, epochs)

    def test_exp_lr(self):
        # [FUNCTION DOC] ExponentialLR with gamma 0.9: lr is 0.05 times 0.9 to the epoch for 10
        # epochs.
        epochs = 10
        single_targets = [0.05 * (0.9**x) for x in range(epochs)]
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = ExponentialLR(self.opt, gamma=0.9)
        self._test(scheduler, targets, epochs)

    def test_poly_lr(self):
        # [FUNCTION DOC] PolynomialLR with power 0.9 and 5 total iterations: lr follows the
        # polynomial decay to zero and stays at zero.
        epochs = 10
        power = 0.9
        total_iters = 5
        single_targets = [
            (1.0 - x / total_iters) ** power * 0.05 for x in range(total_iters)
        ] + [0.0] * (epochs - total_iters)
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = PolynomialLR(self.opt, power=power, total_iters=total_iters)
        self._test(scheduler, targets, epochs)

    def test_cos_anneal_lr(self):
        # [FUNCTION DOC] CosineAnnealingLR with T_max equal to the 10 epochs and a tiny eta_min:
        # lr follows the half cosine from 0.05 downward.
        epochs = 10
        eta_min = 1e-10
        single_targets = [
            eta_min + (0.05 - eta_min) * (1 + math.cos(math.pi * x / epochs)) / 2
            for x in range(epochs)
        ]
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = CosineAnnealingLR(self.opt, T_max=epochs, eta_min=eta_min)
        self._test(scheduler, targets, epochs)

    def test_closed_form_step_lr(self):
        # [FUNCTION DOC] Builds two identical StepLR objects and runs the closed form comparison
        # helper for 20 epochs.
        scheduler = StepLR(self.opt, gamma=0.1, step_size=3)
        closed_form_scheduler = StepLR(self.opt, gamma=0.1, step_size=3)
        self._test_against_closed_form(scheduler, closed_form_scheduler, 20)

    def test_closed_form_linearlr(self):
        # [FUNCTION DOC] Builds two identical LinearLR objects and runs the closed form
        # comparison helper for 20 epochs.
        scheduler = LinearLR(
            self.opt, start_factor=1.0 / 3, end_factor=0.7, total_iters=4
        )
        closed_form_scheduler = LinearLR(
            self.opt, start_factor=1.0 / 3, end_factor=0.7, total_iters=4
        )
        self._test_against_closed_form(scheduler, closed_form_scheduler, 20)

    def test_closed_form_constantlr(self):
        # [FUNCTION DOC] Builds two identical ConstantLR objects and runs the closed form
        # comparison helper for 20 epochs.
        scheduler = ConstantLR(self.opt, factor=1.0 / 3, total_iters=4)
        closed_form_scheduler = ConstantLR(self.opt, factor=1.0 / 3, total_iters=4)
        self._test_against_closed_form(scheduler, closed_form_scheduler, 20)

    def test_closed_form_multi_step_lr(self):
        # [FUNCTION DOC] Builds two identical MultiStepLR objects and runs the closed form
        # comparison helper for 20 epochs.
        scheduler = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        closed_form_scheduler = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        self._test_against_closed_form(scheduler, closed_form_scheduler, 20)

    def test_closed_form_exp_lr(self):
        # [FUNCTION DOC] Builds two identical ExponentialLR objects and runs the closed form
        # comparison helper for 20 epochs.
        scheduler = ExponentialLR(self.opt, gamma=0.9)
        closed_form_scheduler = ExponentialLR(self.opt, gamma=0.9)
        self._test_against_closed_form(scheduler, closed_form_scheduler, 20)

    def test_closed_form_poly_lr(self):
        # [FUNCTION DOC] Builds two identical PolynomialLR objects and runs the closed form
        # comparison helper for 20 epochs.
        scheduler = PolynomialLR(self.opt, power=0.9)
        closed_form_scheduler = PolynomialLR(self.opt, power=0.9)
        self._test_against_closed_form(scheduler, closed_form_scheduler, 20)

    def test_closed_form_cos_anneal_lr(self):
        # [FUNCTION DOC] Builds two identical CosineAnnealingLR objects and runs the closed form
        # comparison helper for 20 epochs.
        eta_min = 1e-10
        epochs = 20
        T_max = 5
        scheduler = CosineAnnealingLR(self.opt, T_max=T_max, eta_min=eta_min)
        closed_form_scheduler = CosineAnnealingLR(
            self.opt, T_max=T_max, eta_min=eta_min
        )
        self._test_against_closed_form(scheduler, closed_form_scheduler, epochs)

    def test_cos_anneal_lr_continue(self):
        # [FUNCTION DOC] Steps CosineAnnealingLR once, then builds a second scheduler on the same
        # optimizer with last_epoch 0; the second one's initial last lrs must be close (tight
        # tolerances) to the first one's after its step.
        eta_min = 0.1
        T_max = 5
        scheduler = CosineAnnealingLR(self.opt, T_max=T_max, eta_min=eta_min)
        self.opt.step()
        scheduler.step()
        original_lrs = scheduler._last_lr
        new_scheduler = CosineAnnealingLR(
            self.opt, T_max=T_max, eta_min=eta_min, last_epoch=0
        )
        new_lrs = new_scheduler._last_lr
        torch.testing.assert_close(original_lrs, new_lrs, rtol=1e-4, atol=1e-5)

    def test_reduce_lr_on_plateau1(self):
        # [FUNCTION DOC] ReduceLROnPlateau (min mode, absolute threshold 0.01, patience 5,
        # cooldown 5) with metrics improving by more than the threshold each epoch: lr stays 0.5
        # for the 10 epochs checked.
        epochs = 10
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        targets = [[0.5] * 20]
        metrics = [10 - i * 0.0167 for i in range(20)]
        scheduler = ReduceLROnPlateau(
            self.opt,
            threshold_mode="abs",
            mode="min",
            threshold=0.01,
            patience=5,
            cooldown=5,
        )
        self._test_reduce_lr_on_plateau(scheduler, targets, metrics, epochs)

    def test_reduce_lr_on_plateau2(self):
        # [FUNCTION DOC] ReduceLROnPlateau min mode, absolute threshold 0.1, patience 5, no
        # cooldown: slow metric improvement below the threshold counts as a plateau, so the lr is
        # cut tenfold periodically over 22 epochs.
        epochs = 22
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        targets = [[0.5] * 6 + [0.05] * 7 + [0.005] * 7 + [0.0005] * 2]
        metrics = [10 - i * 0.0165 for i in range(22)]
        scheduler = ReduceLROnPlateau(
            self.opt,
            patience=5,
            cooldown=0,
            threshold_mode="abs",
            mode="min",
            threshold=0.1,
        )
        self._test_reduce_lr_on_plateau(scheduler, targets, metrics, epochs)

    def test_reduce_lr_on_plateau3(self):
        # [FUNCTION DOC] ReduceLROnPlateau max mode, absolute threshold, patience 5, cooldown 5:
        # metric jumps once then stays flat, producing the staged lr cuts in the target list over
        # 22 epochs.
        epochs = 22
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        targets = [[0.5] * (2 + 6) + [0.05] * (5 + 6) + [0.005] * 4]
        metrics = [-0.8] * 2 + [-0.234] * 20
        scheduler = ReduceLROnPlateau(
            self.opt, mode="max", patience=5, cooldown=5, threshold_mode="abs"
        )
        self._test_reduce_lr_on_plateau(scheduler, targets, metrics, epochs)

    def test_reduce_lr_on_plateau4(self):
        # [FUNCTION DOC] ReduceLROnPlateau max mode, relative threshold 0.1, patience 3: metric
        # grows 2.5 percent per epoch, enough to count as improvement, so lr stays 0.5 for 20
        # epochs.
        epochs = 20
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        targets = [[0.5] * 20]
        metrics = [1.5 * (1.025**i) for i in range(20)]  # 1.025 > 1.1**0.25
        scheduler = ReduceLROnPlateau(
            self.opt, mode="max", patience=3, threshold_mode="rel", threshold=0.1
        )
        self._test_reduce_lr_on_plateau(scheduler, targets, metrics, epochs)

    def test_reduce_lr_on_plateau5(self):
        # [FUNCTION DOC] ReduceLROnPlateau max mode, relative threshold 0.1, patience 5, cooldown
        # 5: metric grows only 0.5 percent per epoch so lr is reduced tenfold in stages over 20
        # epochs.
        epochs = 20
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        targets = [[0.5] * 6 + [0.05] * (5 + 6) + [0.005] * 4]
        metrics = [1.5 * (1.005**i) for i in range(20)]
        scheduler = ReduceLROnPlateau(
            self.opt,
            mode="max",
            threshold_mode="rel",
            threshold=0.1,
            patience=5,
            cooldown=5,
        )
        self._test_reduce_lr_on_plateau(scheduler, targets, metrics, epochs)

    def test_reduce_lr_on_plateau6(self):
        # [FUNCTION DOC] ReduceLROnPlateau min mode, relative threshold 0.1: metric falls 15
        # percent per epoch, always improving, so lr stays 0.5 for 20 epochs.
        epochs = 20
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        targets = [[0.5] * 20]
        metrics = [1.5 * (0.85**i) for i in range(20)]
        scheduler = ReduceLROnPlateau(
            self.opt, mode="min", threshold_mode="rel", threshold=0.1
        )
        self._test_reduce_lr_on_plateau(scheduler, targets, metrics, epochs)

    def test_reduce_lr_on_plateau7(self):
        # [FUNCTION DOC] ReduceLROnPlateau min mode, relative threshold, patience 5, cooldown 5:
        # metric is flat at 1, drops once to 0.6, then 0.5; expected staged lr cuts.
        epochs = 20
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        targets = [[0.5] * 6 + [0.05] * (5 + 6) + [0.005] * 4]
        metrics = [1] * 7 + [0.6] + [0.5] * 12
        scheduler = ReduceLROnPlateau(
            self.opt,
            mode="min",
            threshold_mode="rel",
            threshold=0.1,
            patience=5,
            cooldown=5,
        )
        self._test_reduce_lr_on_plateau(scheduler, targets, metrics, epochs)

    def test_reduce_lr_on_plateau8(self):
        # [FUNCTION DOC] Like the max mode relative threshold test but with per group min_lr of
        # 0.4 and 0.3 so each group's lr is clamped at its own floor (two target sequences).
        epochs = 20
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        targets = [[0.5] * 6 + [0.4] * 14, [0.5] * 6 + [0.3] * 14]
        metrics = [1.5 * (1.005**i) for i in range(20)]
        scheduler = ReduceLROnPlateau(
            self.opt,
            mode="max",
            threshold_mode="rel",
            min_lr=[0.4, 0.3],
            threshold=0.1,
            patience=5,
            cooldown=5,
        )
        self._test_reduce_lr_on_plateau(scheduler, targets, metrics, epochs)

    def test_reduce_lr_on_plateau_get_last_lr_before_step(self):
        # [FUNCTION DOC] ReduceLROnPlateau constructed after setting all group lrs to 0.5:
        # get_last_lr before any step must return 0.5 for every group.
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        scheduler = ReduceLROnPlateau(
            self.opt,
        )
        self.assertEqual(
            scheduler.get_last_lr(), [0.5 for param_group in self.opt.param_groups]
        )

    def test_reduce_lr_on_plateau_preserves_lr_type(self):
        # [FUNCTION DOC] Records the type of each group lr (including the tensor lr), steps
        # ReduceLROnPlateau with patience 0 using worsening metrics so the lr reduction path
        # runs, then asserts every group lr kept its original type. The local variable types
        # shadows the imported types module.
        # Ensures that tensor lrs are preserved, preventing recompilations.
        types = [type(group["lr"]) for group in self.opt.param_groups]
        scheduler = ReduceLROnPlateau(self.opt, mode="min", patience=0)
        scheduler.step(1.0)
        scheduler.step(2.0)  # Triggers scheduler._reduce_lr
        for group, type_ in zip(self.opt.param_groups, types):
            self.assertEqual(type(group["lr"]), type_)

    def test_sequentiallr1(self):
        # [FUNCTION DOC] SequentialLR with ExponentialLR gamma 0.8 for 3 epochs then StepLR gamma
        # 0.1 step size 4; checks 19 epochs of lr values.
        epochs = 19
        schedulers = [None] * 2
        targets = [
            [0.05, 0.04, 0.032]
            + [0.05 for x in range(4)]
            + [0.05 * 0.1 for x in range(4)]
            + [0.05 * 0.01 for x in range(4)]
            + [0.05 * 0.001 for x in range(4)]
        ]
        milestones = [3]
        schedulers[0] = ExponentialLR(self.opt, gamma=0.8)
        schedulers[1] = StepLR(self.opt, gamma=0.1, step_size=4)
        scheduler = SequentialLR(self.opt, schedulers=schedulers, milestones=milestones)
        self._test(scheduler, targets, epochs)

    def test_sequentiallr2(self):
        # [FUNCTION DOC] SequentialLR with ConstantLR factor 0.1 for 3 epochs then ExponentialLR
        # gamma 0.9; checks 13 epochs of lr.
        epochs = 13
        schedulers = [None] * 2
        targets = [[0.005, 0.005, 0.005] + [0.05 * 0.9**x for x in range(10)]]
        milestones = [3]
        schedulers[0] = ConstantLR(self.opt, factor=0.1, total_iters=3)
        schedulers[1] = ExponentialLR(self.opt, gamma=0.9)
        scheduler = SequentialLR(self.opt, schedulers=schedulers, milestones=milestones)
        self._test(scheduler, targets, epochs)

    def test_sequentiallr3(self):
        # [FUNCTION DOC] SequentialLR of ConstantLR, ExponentialLR and StepLR with milestones 3
        # and 6; checks 12 epochs of lr.
        epochs = 12
        schedulers = [None] * 3
        targets = [
            [0.005, 0.005, 0.005]
            + [0.05, 0.04, 0.032]
            + [0.05, 0.05, 0.005, 0.005, 0.0005, 0.0005]
        ]
        milestones = [3, 6]
        schedulers[0] = ConstantLR(self.opt, factor=0.1, total_iters=3)
        schedulers[1] = ExponentialLR(self.opt, gamma=0.8)
        schedulers[2] = StepLR(self.opt, gamma=0.1, step_size=2)
        scheduler = SequentialLR(self.opt, schedulers=schedulers, milestones=milestones)
        self._test(scheduler, targets, epochs)

    def test_sequentiallr4(self):
        # [FUNCTION DOC] Builds SequentialLR from two ConstantLR schedulers (factors 1 and 0.1,
        # milestone 10) on an SGD optimizer with a tensor parameter and asserts the optimizer lr
        # after construction equals the lr before it.
        optimizer = SGD([torch.tensor(0.5)], lr=0.1)
        prev_lr = optimizer.param_groups[0]["lr"]

        schedulers = [
            torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1),
            torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.1),
        ]
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers, milestones=[10]
        )

        new_lr = optimizer.param_groups[0]["lr"]

        # Ensure that multiple schedulers does not affect the initial learning rate
        self.assertEqual(prev_lr, new_lr)

    def test_sequentiallr5(self):
        # [FUNCTION DOC] SequentialLR whose first scheduler is a ChainedScheduler of LinearLR and
        # ConstantLR (milestone 5) followed by a ConstantLR; checks 10 epochs of lr. The
        # docstring says it tests SequentialLR with a ChainedScheduler.
        """
        Test SequentialLR with a ChainedScheduler.
        """
        epochs = 10
        schedulers = []
        milestones = []

        targets = [
            [0.0005, 0.0014, 0.0023, 0.0032, 0.0041]
            + [0.025, 0.025, 0.025, 0.025, 0.025]
        ]

        const_sched = ConstantLR(optimizer=self.opt, factor=0.1, total_iters=5)
        lin_sched = LinearLR(optimizer=self.opt, start_factor=0.1, total_iters=5)
        milestones.append(5)

        chained = ChainedScheduler([lin_sched, const_sched])
        schedulers.append(chained)

        const_sched2 = ConstantLR(optimizer=self.opt, factor=0.5, total_iters=5)
        schedulers.append(const_sched2)

        scheduler = SequentialLR(self.opt, schedulers=schedulers, milestones=milestones)
        self._test(scheduler, targets, epochs)

    def test_sequentiallr_no_warnings(self):
        # [FUNCTION DOC] Builds SequentialLR from LinearLR and ExponentialLR with milestone 5,
        # then for 10 iterations steps the optimizer and the scheduler and asserts the step emits
        # no warning.
        scheduler1 = LinearLR(self.opt, start_factor=0.5, end_factor=0.1, total_iters=5)
        scheduler2 = ExponentialLR(self.opt, gamma=0.9)
        scheduler = SequentialLR(
            self.opt, schedulers=[scheduler1, scheduler2], milestones=[5]
        )

        for _ in range(10):
            self.opt.step()
            with warnings.catch_warnings(record=True) as ws:
                scheduler.step()
                self.assertTrue(len(ws) == 0, "No warning should be raised")

    def test_get_last_lr_sequentiallr(self):
        # [FUNCTION DOC] get_last_lr check for SequentialLR over ConstantLR, ExponentialLR and
        # StepLR with milestones 3 and 6 for 12 epochs; second group targets are ten times the
        # first.
        epochs = 12
        milestones = [3, 6]
        schedulers = [None] * 3
        schedulers[0] = ConstantLR(self.opt, factor=0.1, total_iters=3)
        schedulers[1] = ExponentialLR(self.opt, gamma=0.8)
        schedulers[2] = StepLR(self.opt, gamma=0.1, step_size=2)
        scheduler = SequentialLR(self.opt, schedulers=schedulers, milestones=milestones)
        constant_lr_target = [0.005] * 3
        exponential_lr_target = [0.05, 0.04, 0.032]
        step_lr_target = [0.05, 0.05, 0.005, 0.005, 0.0005, 0.0005]
        single_targets = constant_lr_target + exponential_lr_target + step_lr_target
        targets = [single_targets, [x * 10 for x in single_targets]]
        self._test_get_last_lr(scheduler, targets, epochs)

    def test_sequentiallr_does_not_alias_lr_and_initial_lr(self):
        # [FUNCTION DOC] Regression test for issue 162359. Temporarily replaces self.opt with SGD
        # using a tensor lr of 2.0, builds two LinearLR schedulers inside SequentialLR (milestone
        # 4) and checks 8 epochs of lrs to make sure lr and initial_lr are not aliased. Restores
        # self.opt at the end (skipped if an assertion fails earlier).
        # The TestLRScheduler object uses self.opt to avoid instantiating a new optimizer for each test.
        # self.opt has a float lr, and we need to use a Tensor lr to ensure that a former SequentialLR bug is fixed.
        # For more context, see https://github.com/pytorch/pytorch/issues/162359
        old_opt = self.opt
        lr = torch.tensor(2.0)
        self.opt = SGD(self.net.parameters(), lr=lr)
        milestone = 4
        epochs = 8
        start, end = 0.1, 0.8

        schedulers = [
            LinearLR(self.opt, start, end, total_iters=milestone),
            LinearLR(self.opt, end, start, total_iters=epochs - milestone),
        ]
        targets = [[0.2, 0.55, 0.9, 1.25, 1.6, 1.25, 0.9, 0.55]]

        scheduler = SequentialLR(self.opt, schedulers, milestones=[milestone])
        self._test(scheduler, targets, epochs)
        self.opt = old_opt

    def test_sequentiallr_skips_preceding_schedulers_with_zero_milestone(self):
        # [FUNCTION DOC] Replaces self.opt with SGD lr 1.0 and builds four constant-factor
        # LinearLR schedulers in SequentialLR with milestones 0, 0, 1; schedulers before a zero
        # milestone are skipped, so the lr sequence is 0.3 then 0.4. Restores self.opt
        # afterwards.
        old_opt = self.opt
        self.opt = SGD(self.net.parameters(), lr=1.0)
        schedulers = [
            LinearLR(self.opt, start_factor=0.1, end_factor=0.1, total_iters=10),
            LinearLR(self.opt, start_factor=0.2, end_factor=0.2, total_iters=10),
            LinearLR(self.opt, start_factor=0.3, end_factor=0.3, total_iters=10),
            LinearLR(self.opt, start_factor=0.4, end_factor=0.4, total_iters=10),
        ]
        scheduler = SequentialLR(self.opt, schedulers=schedulers, milestones=[0, 0, 1])
        targets = [[0.3, 0.4]]
        self._test(scheduler, targets, epochs=2)
        self.opt = old_opt

    def test_chained_lr2_get_last_lr_before_step(self):
        # [FUNCTION DOC] ChainedScheduler of LinearLR and MultiStepLR: before any step
        # get_last_lr must equal the last child scheduler's get_last_lr.
        schedulers = [
            LinearLR(self.opt, start_factor=0.4, total_iters=3),
            MultiStepLR(self.opt, milestones=[4, 8, 10], gamma=0.1),
        ]
        scheduler = ChainedScheduler(schedulers)
        self.assertEqual(scheduler.get_last_lr(), schedulers[-1].get_last_lr())

    def test_chained_lr1(self):
        # [FUNCTION DOC] ChainedScheduler with only a StepLR child; the lr follows StepLR and
        # get_last_lr equals the child's.
        epochs = 10
        schedulers = [None] * 1
        targets = [[0.05] * 3 + [0.005] * 3 + [0.0005] * 3 + [0.00005] * 3]
        schedulers[0] = StepLR(self.opt, gamma=0.1, step_size=3)
        scheduler = ChainedScheduler(schedulers)
        self._test([scheduler], targets, epochs)
        self.assertEqual(scheduler.get_last_lr(), schedulers[-1].get_last_lr())

    def test_chained_lr2(self):
        # [FUNCTION DOC] ChainedScheduler with only a LinearLR child (start factor 0.4, 3
        # iterations); lr ramps then stays 0.05, and get_last_lr equals the child's.
        epochs = 10
        schedulers = [None] * 1
        targets = [[0.02, 0.03, 0.04] + [0.05] * 9]
        schedulers[0] = LinearLR(self.opt, start_factor=0.4, total_iters=3)
        scheduler = ChainedScheduler(schedulers)
        self._test([scheduler], targets, epochs)
        self.assertEqual(scheduler.get_last_lr(), schedulers[-1].get_last_lr())

    def test_chained_lr3(self):
        # [FUNCTION DOC] ChainedScheduler combining LinearLR and MultiStepLR over 10 epochs;
        # checks lrs and that get_last_lr equals the last child's.
        epochs = 10
        schedulers = [None] * 2
        targets = [
            [0.02, 0.03, 0.04, 0.05] + [0.005] * 4 + [0.0005] * 3 + [0.00005] * 3
        ]
        schedulers[0] = LinearLR(self.opt, start_factor=0.4, total_iters=3)
        schedulers[1] = MultiStepLR(self.opt, milestones=[4, 8, 10], gamma=0.1)
        scheduler = ChainedScheduler(schedulers)
        self._test([scheduler], targets, epochs)
        self.assertEqual(scheduler.get_last_lr(), schedulers[-1].get_last_lr())

    def test_chained_lr4(self):
        # [FUNCTION DOC] ChainedScheduler of ExponentialLR, ConstantLR and StepLR over 9 epochs;
        # checks lrs and that get_last_lr equals the last child's.
        epochs = 9
        schedulers = [None] * 3
        targets = [
            [0.05 * 0.2 * 0.9**x for x in range(3)]
            + [0.05 * 0.2 * 0.9**3 * 0.1]
            + [0.05 * 0.9**x * 0.1 for x in range(4, 6)]
            + [0.05 * 0.9**x * 0.01 for x in range(6, 9)]
        ]
        schedulers[0] = ExponentialLR(self.opt, gamma=0.9)
        schedulers[1] = ConstantLR(self.opt, factor=0.2, total_iters=4)
        schedulers[2] = StepLR(self.opt, gamma=0.1, step_size=3)
        scheduler = ChainedScheduler(schedulers)
        self._test([scheduler], targets, epochs)
        self.assertEqual(scheduler.get_last_lr(), schedulers[-1].get_last_lr())

    def test_chained_lr5(self):
        # [FUNCTION DOC] ChainedScheduler of PolynomialLR and ConstantLR; expected lrs are the
        # polynomial schedule scaled by the constant factor 0.1 for both groups. Also checks
        # get_last_lr equals the last child's.
        def poly_lr(lr: float):
            # [FUNCTION DOC] HELPER: Returns the polynomial decay lrs for a base lr: for each of
            # total_iters steps the lr times one minus progress to the power, then zeros up to
            # the epoch count. Uses variables defined later in the enclosing test.
            return [
                (lr * ((1.0 - x / total_iters) ** power)) for x in range(total_iters)
            ] + [0.0] * (epochs - total_iters)

        schedulers = [None] * 2
        epochs = 10
        power = 0.9
        total_iters = 5
        const_factor = 0.1
        single_targets = [x * const_factor for x in poly_lr(lr=0.05)]
        targets = [single_targets, [x * const_factor for x in poly_lr(0.5)]]
        schedulers[0] = PolynomialLR(self.opt, power=power, total_iters=total_iters)
        schedulers[1] = ConstantLR(self.opt, factor=const_factor)
        scheduler = ChainedScheduler(schedulers)
        self._test(scheduler, targets, epochs)
        self.assertEqual(scheduler.get_last_lr(), schedulers[-1].get_last_lr())

    def test_compound_step_and_multistep_lr(self):
        # [FUNCTION DOC] Applies StepLR and MultiStepLR together to one optimizer for 10 epochs;
        # checks the compounded lr.
        epochs = 10
        schedulers = [None] * 2
        schedulers[0] = StepLR(self.opt, gamma=0.1, step_size=3)
        schedulers[1] = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        targets = [[0.05] * 2 + [0.005] * 1 + [5e-4] * 2 + [5e-5] + [5e-6] * 3 + [5e-8]]
        self._test(schedulers, targets, epochs)

    def test_compound_step_and_exp_lr(self):
        # [FUNCTION DOC] Applies StepLR and ExponentialLR together; checks the product schedule
        # over 10 epochs.
        epochs = 10
        schedulers = [None] * 2
        single_targets = [0.05 * (0.9**x) for x in range(3)]
        single_targets += [0.005 * (0.9**x) for x in range(3, 6)]
        single_targets += [0.0005 * (0.9**x) for x in range(6, 9)]
        single_targets += [0.00005 * (0.9**x) for x in range(9, 12)]
        targets = [single_targets, [x * epochs for x in single_targets]]
        schedulers[0] = StepLR(self.opt, gamma=0.1, step_size=3)
        schedulers[1] = ExponentialLR(self.opt, gamma=0.9)
        self._test(schedulers, targets, epochs)

    def test_compound_exp_and_multistep_lr(self):
        # [FUNCTION DOC] Applies MultiStepLR and ExponentialLR together; checks the product
        # schedule.
        epochs = 10
        schedulers = [None] * 2
        single_targets = [0.05 * (0.9**x) for x in range(2)]
        single_targets += [0.005 * (0.9**x) for x in range(2, 5)]
        single_targets += [0.0005 * (0.9**x) for x in range(5, 9)]
        single_targets += [0.00005 * (0.9**x) for x in range(9, 11)]
        targets = [single_targets, [x * epochs for x in single_targets]]
        schedulers[0] = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        schedulers[1] = ExponentialLR(self.opt, gamma=0.9)
        self._test(schedulers, targets, epochs)

    def test_compound_exp_and_linearlr(self):
        # [FUNCTION DOC] Applies LinearLR (start 0.4, end 0.9, 4 iterations) and ExponentialLR
        # together; checks the compounded schedule.
        epochs = 10
        iters = 4
        start_factor = 0.4
        end_factor = 0.9
        schedulers = [None] * 2
        single_targets = [0.05 * (0.9**x) for x in range(11)]
        for i in range(iters):
            single_targets[i] *= start_factor + i / iters * (end_factor - start_factor)
        for i in range(iters, 11):
            single_targets[i] *= end_factor
        targets = [single_targets, [x * epochs for x in single_targets]]
        schedulers[0] = LinearLR(
            self.opt,
            start_factor=start_factor,
            end_factor=end_factor,
            total_iters=iters,
        )
        schedulers[1] = ExponentialLR(self.opt, gamma=0.9)
        self._test(schedulers, targets, epochs)

    def test_compound_step_and_constantlr(self):
        # [FUNCTION DOC] Applies StepLR and ConstantLR (factor 0.4, 4 iterations) together;
        # checks the compounded schedule.
        epochs = 10
        iters = 4
        factor = 0.4
        schedulers = [None] * 2
        single_targets = (
            [0.05 * 0.4] * 3
            + [0.005 * 0.4]
            + [0.005] * 2
            + [0.0005] * 3
            + [0.00005] * 3
        )
        targets = [single_targets, [x * epochs for x in single_targets]]
        schedulers[0] = StepLR(self.opt, gamma=0.1, step_size=3)
        schedulers[1] = ConstantLR(self.opt, factor=0.4, total_iters=4)
        self._test(schedulers, targets, epochs)

    def test_compound_linearlr_and_multistep_lr(self):
        # [FUNCTION DOC] Applies MultiStepLR and LinearLR together; checks the compounded
        # schedule.
        epochs = 10
        iters = 4
        start_factor = 0.4
        schedulers = [None] * 2
        single_targets = [0.05] * 2 + [0.005] * 3 + [0.0005] * 4 + [0.00005] * 2
        for i in range(iters):
            single_targets[i] *= start_factor + i / iters * (1 - start_factor)
        targets = [single_targets, [x * epochs for x in single_targets]]
        schedulers[0] = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        schedulers[1] = LinearLR(self.opt, start_factor=start_factor, total_iters=iters)
        self._test(schedulers, targets, epochs)

    def test_compound_cosanneal_and_step_lr(self):
        # [FUNCTION DOC] Applies CosineAnnealingLR and StepLR together; checks the compounded
        # schedule.
        epochs = 10
        eta_min = 1e-10
        single_targets = [
            eta_min + (0.05 - eta_min) * (1 + math.cos(math.pi * x / epochs)) / 2
            for x in range(epochs)
        ]
        single_targets = [x * 0.1 ** (i // 3) for i, x in enumerate(single_targets)]
        targets = [single_targets, [x * epochs for x in single_targets]]
        schedulers = [None] * 2
        schedulers[0] = CosineAnnealingLR(self.opt, T_max=epochs, eta_min=eta_min)
        schedulers[1] = StepLR(self.opt, gamma=0.1, step_size=3)
        self._test(schedulers, targets, epochs)

    def test_compound_cosanneal_and_multistep_lr(self):
        # [FUNCTION DOC] Applies CosineAnnealingLR and MultiStepLR together; checks the
        # compounded schedule.
        epochs = 10
        eta_min = 1e-10
        single_targets = [
            eta_min + (0.05 - eta_min) * (1 + math.cos(math.pi * x / epochs)) / 2
            for x in range(epochs)
        ]
        multipliers = [1] * 2 + [0.1] * 3 + [0.01] * 4 + [0.001]
        single_targets = [x * y for x, y in zip(single_targets, multipliers)]
        targets = [single_targets, [x * epochs for x in single_targets]]
        schedulers = [None] * 2
        schedulers[0] = CosineAnnealingLR(self.opt, T_max=epochs, eta_min=eta_min)
        schedulers[1] = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        self._test(schedulers, targets, epochs)

    def test_compound_cosanneal_and_linearlr(self):
        # [FUNCTION DOC] Applies LinearLR and CosineAnnealingLR together; checks the compounded
        # schedule.
        epochs = 10
        iters = 4
        start_factor = 0.4
        eta_min = 1e-10
        schedulers = [None] * 2
        single_targets = [
            eta_min + (0.05 - eta_min) * (1 + math.cos(math.pi * x / epochs)) / 2
            for x in range(epochs)
        ]
        for i in range(iters):
            single_targets[i] *= start_factor + i / iters * (1 - start_factor)
        targets = [single_targets, [x * epochs for x in single_targets]]
        schedulers[0] = LinearLR(self.opt, start_factor=start_factor, total_iters=iters)
        schedulers[1] = CosineAnnealingLR(self.opt, T_max=epochs, eta_min=eta_min)
        self._test(schedulers, targets, epochs)

    def test_compound_cosanneal_and_exp_lr(self):
        # [FUNCTION DOC] Applies CosineAnnealingLR and ExponentialLR with gamma 0.1; checks the
        # schedule.
        epochs = 10
        eta_min = 1e-10
        single_targets = [
            eta_min + (0.05 - eta_min) * (1 + math.cos(math.pi * x / epochs)) / 2
            for x in range(epochs)
        ]
        multipliers = [0.1**i for i in range(epochs)]
        single_targets = [x * y for x, y in zip(single_targets, multipliers)]
        targets = [single_targets, [x * epochs for x in single_targets]]
        schedulers = [None] * 2
        schedulers[0] = CosineAnnealingLR(self.opt, T_max=epochs, eta_min=eta_min)
        schedulers[1] = ExponentialLR(self.opt, gamma=0.1)
        self._test(schedulers, targets, epochs)

    def test_compound_reduce_lr_on_plateau1(self):
        # [FUNCTION DOC] ReduceLROnPlateau (abs threshold, patience 5, cooldown 5) together with
        # StepLR, stepped by the plateau helper with a slowly improving metric.
        # KNOWN ISSUE: the target list is sliced to drop its first element, leaving an empty
        # list, so the helper's comparison loop never runs and no learning rate is actually
        # verified.
        epochs = 10
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        single_targets = [0.5] * 20
        multipliers = [0.1 ** (i // 3) for i in range(20)]
        single_targets = [x * y for x, y in zip(multipliers, single_targets)]
        targets = [single_targets]
        targets = targets[1:]  # test runs step before checking lr
        metrics = [10 - i * 0.0167 for i in range(20)]
        schedulers = [None, None]
        schedulers[0] = ReduceLROnPlateau(
            self.opt,
            threshold_mode="abs",
            mode="min",
            threshold=0.01,
            patience=5,
            cooldown=5,
        )
        schedulers[1] = StepLR(self.opt, gamma=0.1, step_size=3)
        self._test_reduce_lr_on_plateau(schedulers, targets, metrics, epochs)

    def test_compound_reduce_lr_on_plateau2(self):
        # [FUNCTION DOC] ReduceLROnPlateau (patience 5, no cooldown) together with MultiStepLR
        # milestones 3, 8, 12 on 22 epochs of slowly improving metrics.
        # KNOWN ISSUE: the target list is sliced to drop its first element, leaving an empty
        # list, so the helper's comparison loop never runs and no learning rate is actually
        # verified.
        epochs = 22
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        single_targets = [0.5] * 6 + [0.05] * 7 + [0.005] * 7 + [0.0005] * 2
        multipliers = [1] * 3 + [0.1] * 5 + [0.01] * 4 + [0.001] * 10
        single_targets = [x * y for x, y in zip(single_targets, multipliers)]
        targets = [single_targets]
        targets = targets[1:]  # test runs step before checking lr
        metrics = [10 - i * 0.0165 for i in range(22)]
        schedulers = [None] * 2
        schedulers[0] = ReduceLROnPlateau(
            self.opt,
            patience=5,
            cooldown=0,
            threshold_mode="abs",
            mode="min",
            threshold=0.1,
        )
        schedulers[1] = MultiStepLR(self.opt, gamma=0.1, milestones=[3, 8, 12])
        self._test_reduce_lr_on_plateau(schedulers, targets, metrics, epochs)

    def test_compound_reduce_lr_on_plateau3(self):
        # [FUNCTION DOC] ReduceLROnPlateau in max mode together with ExponentialLR gamma 0.1 on
        # 22 epochs.
        # KNOWN ISSUE: the target list is sliced to drop its first element, leaving an empty
        # list, so the helper's comparison loop never runs and no learning rate is actually
        # verified.
        epochs = 22
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        single_targets = [0.5] * (2 + 6) + [0.05] * (5 + 6) + [0.005] * 4
        multipliers = [0.1**i for i in range(epochs)]
        single_targets = [x * y for x, y in zip(multipliers, single_targets)]
        targets = [single_targets]
        targets = targets[1:]  # test runs step before checking lr
        metrics = [-0.8] * 2 + [-0.234] * 20
        schedulers = [None, None]
        schedulers[0] = ReduceLROnPlateau(
            self.opt, mode="max", patience=5, cooldown=5, threshold_mode="abs"
        )
        schedulers[1] = ExponentialLR(self.opt, gamma=0.1)
        self._test_reduce_lr_on_plateau(schedulers, targets, metrics, epochs)

    def test_compound_reduce_lr_on_plateau4(self):
        # [FUNCTION DOC] ReduceLROnPlateau in max mode (relative threshold) together with
        # CosineAnnealingLR on a cosine target list.
        # KNOWN ISSUE: the target list is sliced to drop its first element, leaving an empty
        # list, so the helper's comparison loop never runs and no learning rate is actually
        # verified.
        epochs = 20
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.05
        epochs = 10
        eta_min = 1e-10
        single_targets = [
            eta_min + (0.05 - eta_min) * (1 + math.cos(math.pi * x / epochs)) / 2
            for x in range(epochs)
        ]
        targets = [single_targets]
        targets = targets[1:]  # test runs step before checking lr
        metrics = [1.5 * (1.025**i) for i in range(20)]  # 1.025 > 1.1**0.25
        schedulers = [None, None]
        schedulers[0] = ReduceLROnPlateau(
            self.opt, mode="max", patience=3, threshold_mode="rel", threshold=0.1
        )
        schedulers[1] = CosineAnnealingLR(self.opt, epochs, eta_min)
        self._test_reduce_lr_on_plateau(schedulers, targets, metrics, epochs)

    def test_compound_reduce_lr_on_plateau5(self):
        # [FUNCTION DOC] ReduceLROnPlateau together with LinearLR (start 0.4, 4 iterations) on 22
        # epochs.
        # KNOWN ISSUE: the target list is sliced to drop its first element, leaving an empty
        # list, so the helper's comparison loop never runs and no learning rate is actually
        # verified.
        iters = 4
        start_factor = 0.4
        epochs = 22
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        single_targets = [0.5] * 6 + [0.05] * 7 + [0.005] * 7 + [0.0005] * 2
        multipliers = [1] * 22
        for i in range(iters):
            multipliers[i] *= start_factor + i / iters * (1 - start_factor)
        single_targets = [x * y for x, y in zip(single_targets, multipliers)]
        targets = [single_targets]
        targets = targets[1:]  # test runs step before checking lr
        metrics = [10 - i * 0.0165 for i in range(22)]
        schedulers = [None] * 2
        schedulers[0] = ReduceLROnPlateau(
            self.opt,
            patience=5,
            cooldown=0,
            threshold_mode="abs",
            mode="min",
            threshold=0.1,
        )
        schedulers[1] = LinearLR(self.opt, start_factor=start_factor, total_iters=iters)
        self._test_reduce_lr_on_plateau(schedulers, targets, metrics, epochs)

    def test_cycle_lr_invalid_mode(self):
        # [FUNCTION DOC] CyclicLR with an unknown mode name must raise ValueError.
        with self.assertRaises(ValueError):
            scheduler = CyclicLR(self.opt, base_lr=0, max_lr=0, mode="CATS")

    def test_cycle_lr_triangular_mode_one_lr(self):
        # [FUNCTION DOC] CyclicLR triangular mode with one base and max lr applied to both
        # groups, step size up 4 and momentum cycling between 5 and 1; checks lr and momentum
        # over 11 batches.
        lr_target = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3]
        momentum_target = [5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3]
        lr_targets = [lr_target, lr_target]
        momentum_targets = [momentum_target, momentum_target]
        scheduler = CyclicLR(
            self.opt,
            base_lr=1,
            max_lr=5,
            step_size_up=4,
            cycle_momentum=True,
            base_momentum=1,
            max_momentum=5,
            mode="triangular",
        )
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, len(lr_target))

    def test_cycle_lr_triangular_mode_one_lr_no_momentum(self):
        # [FUNCTION DOC] CyclicLR triangular mode with cycle_momentum off: lr follows the
        # triangle while momentum stays at the optimizer default.
        lr_target = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3]
        lr_targets = [lr_target, lr_target]
        momentum_target = [self.opt.defaults["momentum"]] * len(lr_target)
        momentum_targets = [momentum_target, momentum_target]
        scheduler = CyclicLR(
            self.opt,
            base_lr=1,
            max_lr=5,
            step_size_up=4,
            cycle_momentum=False,
            mode="triangular",
        )
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, len(lr_target))

    def test_cycle_lr_triangular2_mode_one_lr(self):
        # [FUNCTION DOC] CyclicLR triangular2 mode (peak amplitude halves each cycle) with lr
        # from 1 to 5 and momentum from 5 to 1; checks 22 batches.
        lr_target = [
            1,
            2,
            3,
            4,
            5,
            4,
            3,
            2,
            1,
            1.5,
            2.0,
            2.5,
            3.0,
            2.5,
            2.0,
            1.5,
            1,
            1.25,
            1.50,
            1.75,
            2.00,
            1.75,
        ]
        momentum_target = [
            5.0,
            4.0,
            3.0,
            2.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            4.5,
            4.0,
            3.5,
            3.0,
            3.5,
            4.0,
            4.5,
            5.0,
            4.75,
            4.5,
            4.25,
            4.0,
            4.25,
        ]
        lr_targets = [lr_target, lr_target]
        momentum_targets = [momentum_target, momentum_target]
        scheduler = CyclicLR(
            self.opt,
            base_lr=1,
            max_lr=5,
            step_size_up=4,
            cycle_momentum=True,
            base_momentum=1,
            max_momentum=5,
            mode="triangular2",
        )
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, len(lr_target))

    def test_cycle_lr_exp_range_mode_one_lr(self):
        # [FUNCTION DOC] CyclicLR exp_range mode with gamma 0.9: the triangle amplitude decays by
        # gamma per iteration; checks lr and inverse momentum targets.
        base_lr, max_lr = 1, 5
        diff_lr = max_lr - base_lr
        gamma = 0.9
        xs = [0, 0.25, 0.5, 0.75, 1, 0.75, 0.50, 0.25, 0, 0.25, 0.5, 0.75, 1]
        lr_target = [base_lr + x * diff_lr * gamma**i for i, x in enumerate(xs)]
        momentum_target = [max_lr - x * diff_lr * gamma**i for i, x in enumerate(xs)]
        lr_targets = [lr_target, lr_target]
        momentum_targets = [momentum_target, momentum_target]
        scheduler = CyclicLR(
            self.opt,
            base_lr=base_lr,
            max_lr=max_lr,
            step_size_up=4,
            cycle_momentum=True,
            base_momentum=base_lr,
            max_momentum=max_lr,
            mode="exp_range",
            gamma=gamma,
        )
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, len(lr_target))

    def test_cycle_lr_triangular_mode(self):
        # [FUNCTION DOC] CyclicLR triangular mode with separate base and max lr and momentum
        # lists for each group.
        lr_target_1 = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3]
        lr_target_2 = [x + 1 for x in lr_target_1]
        lr_targets = [lr_target_1, lr_target_2]
        momentum_target_1 = [5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3]
        momentum_target_2 = [x + 1 for x in momentum_target_1]
        momentum_targets = [momentum_target_1, momentum_target_2]
        scheduler = CyclicLR(
            self.opt,
            base_lr=[1, 2],
            max_lr=[5, 6],
            step_size_up=4,
            cycle_momentum=True,
            base_momentum=[1, 2],
            max_momentum=[5, 6],
            mode="triangular",
        )
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, len(lr_target_1))

    def test_cycle_lr_triangular2_mode(self):
        # [FUNCTION DOC] CyclicLR triangular2 mode with separate per group lr and momentum
        # ranges; 22 batches.
        lr_target_1 = [
            1,
            2,
            3,
            4,
            5,
            4,
            3,
            2,
            1,
            1.5,
            2.0,
            2.5,
            3.0,
            2.5,
            2.0,
            1.5,
            1,
            1.25,
            1.50,
            1.75,
            2.00,
            1.75,
        ]
        lr_target_2 = [x + 2 for x in lr_target_1]
        lr_targets = [lr_target_1, lr_target_2]
        momentum_target_1 = [
            5.0,
            4.0,
            3.0,
            2.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            4.5,
            4.0,
            3.5,
            3.0,
            3.5,
            4.0,
            4.5,
            5.0,
            4.75,
            4.5,
            4.25,
            4.0,
            4.25,
        ]
        momentum_target_2 = [x + 2 for x in momentum_target_1]
        momentum_targets = [momentum_target_1, momentum_target_2]
        scheduler = CyclicLR(
            self.opt,
            base_lr=[1, 3],
            max_lr=[5, 7],
            step_size_up=4,
            cycle_momentum=True,
            base_momentum=[1, 3],
            max_momentum=[5, 7],
            mode="triangular2",
        )
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, len(lr_target_1))

    def test_cycle_lr_exp_range_mode(self):
        # [FUNCTION DOC] CyclicLR exp_range mode with per group lr and momentum ranges and gamma
        # 0.9.
        base_lr_1, max_lr_1 = 1, 5
        base_lr_2, max_lr_2 = 5, 12

        diff_lr_1 = max_lr_1 - base_lr_1
        diff_lr_2 = max_lr_2 - base_lr_2

        gamma = 0.9
        xs = [0, 0.25, 0.5, 0.75, 1, 0.75, 0.50, 0.25, 0, 0.25, 0.5, 0.75, 1]
        lr_target_1 = [base_lr_1 + x * diff_lr_1 * gamma**i for i, x in enumerate(xs)]
        lr_target_2 = [base_lr_2 + x * diff_lr_2 * gamma**i for i, x in enumerate(xs)]
        lr_targets = [lr_target_1, lr_target_2]
        momentum_target_1 = [
            max_lr_1 - x * diff_lr_1 * gamma**i for i, x in enumerate(xs)
        ]
        momentum_target_2 = [
            max_lr_2 - x * diff_lr_2 * gamma**i for i, x in enumerate(xs)
        ]
        momentum_targets = [momentum_target_1, momentum_target_2]
        scheduler = CyclicLR(
            self.opt,
            base_lr=[base_lr_1, base_lr_2],
            max_lr=[max_lr_1, max_lr_2],
            step_size_up=4,
            cycle_momentum=True,
            base_momentum=[base_lr_1, base_lr_2],
            max_momentum=[max_lr_1, max_lr_2],
            mode="exp_range",
            gamma=gamma,
        )
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, len(lr_target_1))

    def test_cycle_lr_triangular_mode_step_size_up_down(self):
        # [FUNCTION DOC] CyclicLR triangular mode with asymmetric step_size_up 4 and
        # step_size_down 6.
        lr_target = [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            13.0 / 3,
            11.0 / 3,
            9.0 / 3,
            7.0 / 3,
            5.0 / 3,
            1.0,
        ]
        lr_targets = [lr_target, lr_target]
        momentum_target = [
            5.0,
            4.0,
            3.0,
            2.0,
            1.0,
            5.0 / 3,
            7.0 / 3,
            3.0,
            11.0 / 3,
            13.0 / 3,
            5.0,
        ]
        momentum_targets = [momentum_target, momentum_target]

        scheduler = CyclicLR(
            self.opt,
            base_lr=1,
            max_lr=5,
            step_size_up=4,
            step_size_down=6,
            cycle_momentum=True,
            base_momentum=1,
            max_momentum=5,
            mode="triangular",
        )
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, len(lr_target))

    def test_cycle_lr_triangular2_mode_step_size_up_down(self):
        # [FUNCTION DOC] CyclicLR triangular2 mode with step_size_up 2 and step_size_down 6 and
        # per group ranges offset by 2; checks 24 batches.
        lr_base_target = [
            1.0,
            3.0,
            5.0,
            13.0 / 3,
            11.0 / 3,
            9.0 / 3,
            7.0 / 3,
            5.0 / 3,
            1.0,
            2.0,
            3.0,
            8.0 / 3,
            7.0 / 3,
            6.0 / 3,
            5.0 / 3,
            4.0 / 3,
            1.0,
            3.0 / 2,
            2.0,
            11.0 / 6,
            10.0 / 6,
            9.0 / 6,
            8.0 / 6,
            7.0 / 6,
        ]
        momentum_base_target = [
            5.0,
            3.0,
            1.0,
            5.0 / 3,
            7.0 / 3,
            3.0,
            11.0 / 3,
            13.0 / 3,
            5.0,
            4.0,
            3.0,
            10.0 / 3,
            11.0 / 3,
            4.0,
            13.0 / 3,
            14.0 / 3,
            5.0,
            4.5,
            4.0,
            25.0 / 6,
            13.0 / 3,
            4.5,
            14.0 / 3,
            29.0 / 6,
        ]
        deltas = [2 * i for i in range(2)]
        base_lrs = [1 + delta for delta in deltas]
        max_lrs = [5 + delta for delta in deltas]
        lr_targets = [[x + delta for x in lr_base_target] for delta in deltas]
        momentum_targets = [
            [x + delta for x in momentum_base_target] for delta in deltas
        ]
        scheduler = CyclicLR(
            self.opt,
            base_lr=base_lrs,
            max_lr=max_lrs,
            step_size_up=2,
            step_size_down=6,
            cycle_momentum=True,
            base_momentum=base_lrs,
            max_momentum=max_lrs,
            mode="triangular2",
        )
        self._test_cycle_lr(
            scheduler, lr_targets, momentum_targets, len(lr_base_target)
        )

    def test_cycle_lr_exp_range_mode_step_size_up_down(self):
        # [FUNCTION DOC] CyclicLR exp_range mode with step_size_up 2, step_size_down 6 and gamma
        # 0.9.
        base_lr, max_lr = 1, 5
        diff_lr = max_lr - base_lr
        gamma = 0.9
        xs = [
            0.0,
            0.5,
            1.0,
            5.0 / 6,
            4.0 / 6,
            3.0 / 6,
            2.0 / 6,
            1.0 / 6,
            0.0,
            0.5,
            1.0,
            5.0 / 6,
            4.0 / 6,
        ]
        lr_target = [base_lr + x * diff_lr * gamma**i for i, x in enumerate(xs)]
        lr_targets = [lr_target, lr_target]
        momentum_target = [max_lr - x * diff_lr * gamma**i for i, x in enumerate(xs)]
        momentum_targets = [momentum_target, momentum_target]
        scheduler = CyclicLR(
            self.opt,
            base_lr=base_lr,
            max_lr=max_lr,
            step_size_up=2,
            step_size_down=6,
            cycle_momentum=True,
            base_momentum=base_lr,
            max_momentum=max_lr,
            mode="exp_range",
            gamma=gamma,
        )
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, len(lr_target))

    def test_cycle_lr_with_momentumless_optimizer(self):
        # [FUNCTION DOC] Regression for issue 19003: temporarily swaps self.opt for Adam (no
        # momentum entry), builds CyclicLR with cycle_momentum off and checks the lr triangle
        # while momentum targets are None. Restores self.opt at the end.
        # Note [Temporarily set optimizer to Adam]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # The TestLRScheduler object carries around an SGD optimizer to avoid having to
        # instantiate one for every test. This gets in the way for our very specific case
        # in which we need to use Adam (or really any optimizer that doesn't use momentum)
        # in order to test that the momentum bug in CyclicLR is fixed (the bug is described
        # in more detail in https://github.com/pytorch/pytorch/issues/19003 ).
        old_opt = self.opt
        self.opt = Adam(
            [
                {"params": self.net.conv1.parameters()},
                {"params": self.net.conv2.parameters(), "lr": 0.5},
            ],
            lr=0.05,
        )

        lr_target = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3]
        lr_targets = [lr_target, lr_target]
        momentum_target = [None] * len(lr_target)
        momentum_targets = [momentum_target, momentum_target]
        scheduler = CyclicLR(
            self.opt,
            base_lr=1,
            max_lr=5,
            step_size_up=4,
            cycle_momentum=False,
            mode="triangular",
        )
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, len(lr_target))

        self.opt = old_opt  # set optimizer back to SGD

    def test_cycle_lr_cycle_momentum_fail_with_momentumless_optimizer(self):
        # [FUNCTION DOC] CyclicLR with cycle_momentum true on Rprop (no momentum or betas) must
        # raise ValueError.
        with self.assertRaises(ValueError):
            rprop_opt = Rprop(self.net.parameters())
            scheduler = CyclicLR(rprop_opt, base_lr=1, max_lr=5, cycle_momentum=True)

    def test_cycle_lr_cycle_momentum_with_beta1_optimizer(self):
        # [FUNCTION DOC] Builds CyclicLR with cycle_momentum true on Adam (which has beta1) and
        # expects no error; nothing is asserted beyond construction succeeding.
        adam_opt = Adam(self.net.parameters())
        scheduler = CyclicLR(adam_opt, base_lr=1, max_lr=5, cycle_momentum=True)

    def test_cycle_lr_removed_after_out_of_scope(self):
        # [FUNCTION DOC] With gc disabled, a local function builds an Adam and a CyclicLR (no
        # momentum cycling) and returns a weak reference; asserts the scheduler was already freed
        # (AssertionError otherwise), then re-enables gc.
        import gc
        import weakref

        gc.disable()

        def test():
            # [FUNCTION DOC] HELPER: Creates Adam and CyclicLR, returns a weak reference to the
            # scheduler.
            adam_opt = Adam(self.net.parameters())
            scheduler = CyclicLR(adam_opt, base_lr=1, max_lr=5, cycle_momentum=False)
            return weakref.ref(scheduler)

        ref = test()
        if ref() is not None:
            raise AssertionError("Expected scheduler to be garbage collected")
        gc.enable()

    def test_cycle_lr_state_dict_picklable(self):
        # [FUNCTION DOC] Checks CyclicLR state dict handling of the scale function for pickling,
        # in three cases: built-in mode (no private scale function entry kept, custom entry
        # None), a plain function scale_fn (custom entry None) and a callable object (custom
        # entry equals the object's instance dict). Each state dict must be picklable.
        adam_opt = Adam(self.net.parameters())

        # Case 1: Built-in mode
        scheduler = CyclicLR(adam_opt, base_lr=1, max_lr=5, cycle_momentum=False)
        self.assertIsInstance(scheduler._scale_fn_ref, types.FunctionType)
        state = scheduler.state_dict()
        self.assertNotIn("_scale_fn_ref", state)
        self.assertIs(state["_scale_fn_custom"], None)
        pickle.dumps(state)

        # Case 2: Custom `scale_fn`, a function object
        def scale_fn(_):
            # [FUNCTION DOC] HELPER: Scale function returning a constant 0.5 regardless of its
            # argument.
            return 0.5

        scheduler = CyclicLR(
            adam_opt, base_lr=1, max_lr=5, cycle_momentum=False, scale_fn=scale_fn
        )
        state = scheduler.state_dict()
        self.assertNotIn("_scale_fn_ref", state)
        self.assertIs(state["_scale_fn_custom"], None)
        pickle.dumps(state)

        # Case 3: Custom `scale_fn`, a callable class
        class ScaleFn:
            # [CLASS DOC] HELPER: Callable class used as a CyclicLR scale function; stores x
            # equal to 0.5.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets attribute x to 0.5.
                self.x = 0.5

            def __call__(self, _):
                # [FUNCTION DOC] HELPER: Returns the stored attribute x, ignoring its argument.
                return self.x

        scale_fn = ScaleFn()

        scheduler = CyclicLR(
            adam_opt, base_lr=1, max_lr=5, cycle_momentum=False, scale_fn=scale_fn
        )
        state = scheduler.state_dict()
        self.assertNotIn("_scale_fn_ref", state)
        self.assertEqual(state["_scale_fn_custom"], scale_fn.__dict__)
        pickle.dumps(state)

    def test_cycle_lr_scale_fn_restored_from_state_dict(self):
        # [FUNCTION DOC] Checks CyclicLR load_state_dict: with a built-in mode (triangular2) the
        # restored scheduler gets the same mode and a scale function while custom is None; with a
        # custom scale_fn both schedulers keep the identical function object in the custom
        # attribute.
        adam_opt = Adam(self.net.parameters())

        # Case 1: Built-in mode
        scheduler = CyclicLR(
            adam_opt, base_lr=1, max_lr=5, cycle_momentum=False, mode="triangular2"
        )
        restored_scheduler = CyclicLR(
            adam_opt, base_lr=1, max_lr=5, cycle_momentum=False
        )
        restored_scheduler.load_state_dict(scheduler.state_dict())
        self.assertTrue(restored_scheduler.mode == scheduler.mode == "triangular2")
        self.assertIsNotNone(restored_scheduler._scale_fn_ref) and self.assertIsNotNone(
            scheduler._scale_fn_ref
        )
        self.assertIs(restored_scheduler._scale_fn_custom, None)
        self.assertIs(scheduler._scale_fn_custom, None)

        # Case 2: Custom `scale_fn`
        def scale_fn(_):
            # [FUNCTION DOC] HELPER: Scale function returning a constant 0.5.
            return 0.5

        scheduler = CyclicLR(
            adam_opt, base_lr=1, max_lr=5, cycle_momentum=False, scale_fn=scale_fn
        )
        restored_scheduler = CyclicLR(
            adam_opt, base_lr=1, max_lr=5, cycle_momentum=False, scale_fn=scale_fn
        )
        restored_scheduler.load_state_dict(scheduler.state_dict())
        self.assertIs(scheduler._scale_fn_custom, scale_fn)
        self.assertIs(restored_scheduler._scale_fn_custom, scale_fn)

    def test_onecycle_lr_invalid_anneal_strategy(self):
        # [FUNCTION DOC] OneCycleLR with an unknown anneal_strategy must raise ValueError.
        with self.assertRaises(ValueError):
            scheduler = OneCycleLR(
                self.opt, max_lr=1e-3, total_steps=10, anneal_strategy="CATS"
            )

    def test_onecycle_lr_invalid_pct_start(self):
        # [FUNCTION DOC] OneCycleLR with pct_start above 1 must raise ValueError.
        with self.assertRaises(ValueError):
            scheduler = OneCycleLR(self.opt, max_lr=1e-3, total_steps=10, pct_start=1.1)

    def test_onecycle_lr_cannot_calculate_total_steps(self):
        # [FUNCTION DOC] OneCycleLR with neither total_steps nor epochs and steps_per_epoch must
        # raise ValueError.
        with self.assertRaises(ValueError):
            scheduler = OneCycleLR(self.opt, max_lr=1e-3)

    def test_onecycle_lr_linear_annealing(self):
        # [FUNCTION DOC] OneCycleLR with linear annealing over 10 steps (max_lr 25,
        # final_div_factor 2, momentum between 1 and 22): checks lr and momentum for both groups.
        lr_target = [1, 13, 25, 21.5, 18, 14.5, 11, 7.5, 4, 0.5]
        momentum_target = [22, 11.5, 1, 4, 7, 10, 13, 16, 19, 22]
        lr_targets = [lr_target, lr_target]
        momentum_targets = [momentum_target, momentum_target]
        scheduler = OneCycleLR(
            self.opt,
            max_lr=25,
            final_div_factor=2,
            base_momentum=1,
            max_momentum=22,
            total_steps=10,
            anneal_strategy="linear",
        )
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, 10)

    def test_onecycle_lr_linear_annealing_three_phases(self):
        # [FUNCTION DOC] OneCycleLR three phase mode with linear annealing and pct_start 0.4:
        # checks lr and momentum over 10 steps.
        lr_target = [1, 9, 17, 25, 17, 9, 1, 0.75, 0.5, 0.25]
        momentum_target = [22, 15, 8, 1, 8, 15, 22, 22, 22, 22]
        lr_targets = [lr_target, lr_target]
        momentum_targets = [momentum_target, momentum_target]
        scheduler = OneCycleLR(
            self.opt,
            max_lr=25,
            div_factor=25,
            base_momentum=1,
            max_momentum=22,
            total_steps=10,
            anneal_strategy="linear",
            pct_start=0.4,
            final_div_factor=4,
            three_phase=True,
        )
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, 10)

    def test_onecycle_lr_cosine_annealing(self):
        # [FUNCTION DOC] OneCycleLR with default cosine annealing over 10 steps: lr and momentum
        # follow the cosine formula computed by a local helper.
        def annealing_cos(start, end, pct):
            # [FUNCTION DOC] HELPER: Cosine interpolation from start to end at fraction pct.
            cos_out = math.cos(math.pi * pct) + 1
            return end + (start - end) / 2.0 * cos_out

        lr_target = [
            1,
            13,
            25,
            annealing_cos(25, 0.5, 1 / 7.0),
            annealing_cos(25, 0.5, 2 / 7.0),
            annealing_cos(25, 0.5, 3 / 7.0),
            annealing_cos(25, 0.5, 4 / 7.0),
            annealing_cos(25, 0.5, 5 / 7.0),
            annealing_cos(25, 0.5, 6 / 7.0),
            0.5,
        ]
        momentum_target = [
            22,
            11.5,
            1,
            annealing_cos(1, 22, 1 / 7.0),
            annealing_cos(1, 22, 2 / 7.0),
            annealing_cos(1, 22, 3 / 7.0),
            annealing_cos(1, 22, 4 / 7.0),
            annealing_cos(1, 22, 5 / 7.0),
            annealing_cos(1, 22, 6 / 7.0),
            22,
        ]
        lr_targets = [lr_target, lr_target]
        momentum_targets = [momentum_target, momentum_target]
        scheduler = OneCycleLR(
            self.opt,
            max_lr=25,
            final_div_factor=2,
            base_momentum=1,
            max_momentum=22,
            total_steps=10,
        )
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, 10)

    def test_onecycle_lr_legacy_state_dict(self):
        # [FUNCTION DOC] Builds cosine OneCycleLR, deletes its private anneal func type
        # attribute, takes the state dict (must not contain that key), injects the legacy
        # anneal_func entry referencing the class cosine function, reloads it, and verifies the
        # cosine lr and momentum sequence still holds for 10 steps.
        scheduler = OneCycleLR(
            self.opt,
            max_lr=25,
            final_div_factor=2,
            base_momentum=1,
            max_momentum=22,
            total_steps=10,
            anneal_strategy="cos",
        )
        delattr(scheduler, "_anneal_func_type")
        state_dict = scheduler.state_dict()
        self.assertNotIn("anneal_func_type", state_dict)
        state_dict["anneal_func"] = OneCycleLR._annealing_cos
        scheduler.load_state_dict(state_dict)

        def annealing_cos(start, end, pct):
            # [FUNCTION DOC] HELPER: Cosine interpolation from start to end at fraction pct.
            cos_out = math.cos(math.pi * pct) + 1
            return end + (start - end) / 2.0 * cos_out

        lr_target = [
            1,
            13,
            25,
            annealing_cos(25, 0.5, 1 / 7.0),
            annealing_cos(25, 0.5, 2 / 7.0),
            annealing_cos(25, 0.5, 3 / 7.0),
            annealing_cos(25, 0.5, 4 / 7.0),
            annealing_cos(25, 0.5, 5 / 7.0),
            annealing_cos(25, 0.5, 6 / 7.0),
            0.5,
        ]
        momentum_target = [
            22,
            11.5,
            1,
            annealing_cos(1, 22, 1 / 7.0),
            annealing_cos(1, 22, 2 / 7.0),
            annealing_cos(1, 22, 3 / 7.0),
            annealing_cos(1, 22, 4 / 7.0),
            annealing_cos(1, 22, 5 / 7.0),
            annealing_cos(1, 22, 6 / 7.0),
            22,
        ]
        lr_targets = [lr_target, lr_target]
        momentum_targets = [momentum_target, momentum_target]
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, 10)

    def test_cycle_lr_with_adam(self):
        # [FUNCTION DOC] Temporarily swaps self.opt for Adam and checks linear OneCycleLR where
        # momentum is taken from the first beta (use_beta1 on) over 10 steps. Restores self.opt
        # at the end.
        old_opt = self.opt
        self.opt = Adam(
            [
                {"params": self.net.conv1.parameters()},
                {"params": self.net.conv2.parameters(), "lr": 0.5},
            ],
            lr=0.05,
        )

        lr_target = [1, 13, 25, 21.5, 18, 14.5, 11, 7.5, 4, 0.5]
        momentum_target = [22, 11.5, 1, 4, 7, 10, 13, 16, 19, 22]
        lr_targets = [lr_target, lr_target]
        momentum_targets = [momentum_target, momentum_target]
        scheduler = OneCycleLR(
            self.opt,
            max_lr=25,
            final_div_factor=2,
            base_momentum=1,
            max_momentum=22,
            total_steps=10,
            anneal_strategy="linear",
        )
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, 10, use_beta1=True)
        self.opt = old_opt  # set optimizer back to SGD

    def test_lambda_lr(self):
        # [FUNCTION DOC] LambdaLR with one lambda per group (0.9 to the epoch and 0.8 to the
        # epoch) on base lrs 0.05 and 0.4; checks 10 epochs.
        epochs = 10
        self.opt.param_groups[0]["lr"] = 0.05
        self.opt.param_groups[1]["lr"] = 0.4
        targets = [
            [0.05 * (0.9**x) for x in range(epochs)],
            [0.4 * (0.8**x) for x in range(epochs)],
        ]
        scheduler = LambdaLR(
            self.opt, lr_lambda=[lambda x1: 0.9**x1, lambda x2: 0.8**x2]
        )
        self._test(scheduler, targets, epochs)

    def test_multiplicative_lr(self):
        # [FUNCTION DOC] MultiplicativeLR with constant factors 0.9 and 0.8 per group; checks the
        # cumulative geometric lr for 10 epochs.
        epochs = 10
        self.opt.param_groups[0]["lr"] = 0.05
        self.opt.param_groups[1]["lr"] = 0.4
        targets = [
            [0.05 * (0.9**x) for x in range(epochs)],
            [0.4 * (0.8**x) for x in range(epochs)],
        ]
        scheduler = MultiplicativeLR(
            self.opt, lr_lambda=[lambda x1: 0.9, lambda x2: 0.8]
        )
        self._test(scheduler, targets, epochs)

    def test_multiplicative_lr_with_lr_lambda(self):
        # [FUNCTION DOC] MultiplicativeLR given a plain number instead of a callable (alone or in
        # a list) must raise TypeError saying lr_lambda should be a function.
        lr_lambda = 0.95
        with self.assertRaisesRegex(TypeError, "lr_lambda should be a function"):
            MultiplicativeLR(self.opt, lr_lambda)

        lr_lambda2 = 0.95
        with self.assertRaisesRegex(TypeError, "lr_lambda should be a function"):
            MultiplicativeLR(self.opt, [lr_lambda, lr_lambda2])

    @parametrize("T_mult", [1, 2, 4])
    def test_CosineAnnealingWarmRestarts_lr1(self, T_mult):
        # [FUNCTION DOC] Parametrized over T_mult 1, 2, 4. CosineAnnealingWarmRestarts with T_0
        # 10: builds the expected lr by simulating the restart counter and period growth over 100
        # integer steps and compares with the scheduler.
        iters = 100
        eta_min = 1e-10
        T_i = 10
        T_cur = 0
        targets = [[0.05], [0.5]]
        scheduler = CosineAnnealingWarmRestarts(
            self.opt, T_0=T_i, T_mult=T_mult, eta_min=eta_min
        )
        for _ in range(1, iters, 1):
            T_cur += 1
            if T_cur >= T_i:
                T_cur = T_cur - T_i
                T_i = int(T_mult) * T_i
            targets[0] += [
                eta_min + (0.05 - eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2
            ]
            targets[1] += [
                eta_min + (0.5 - eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2
            ]
        self._test(scheduler, targets, iters)

    def test_CosineAnnealingWarmRestarts_lr2(self):
        # [FUNCTION DOC] For T_mult 1, 2, 4 steps CosineAnnealingWarmRestarts with fractional
        # epochs in 0.1 increments over 30 epochs and compares with the simulated cosine
        # restarts.
        iters = 30
        eta_min = 1e-10
        T_mults = [1, 2, 4]
        for T_mult in T_mults:
            T_i = 10
            T_cur = 0
            targets = [[0.05], [0.5]]
            scheduler = CosineAnnealingWarmRestarts(
                self.opt, T_0=T_i, T_mult=T_mult, eta_min=eta_min
            )
            for _ in torch.arange(0.1, iters, 0.1):
                T_cur = round(T_cur + 0.1, 1)
                if T_cur >= T_i:
                    T_cur = T_cur - T_i
                    T_i = int(T_mult) * T_i
                targets[0] += [
                    eta_min
                    + (0.05 - eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2
                ]
                targets[1] += [
                    eta_min
                    + (0.5 - eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2
                ]
            self._test_CosineAnnealingWarmRestarts(scheduler, targets, iters)

    def test_CosineAnnealingWarmRestarts_lr3(self):
        # [FUNCTION DOC] CosineAnnealingWarmRestarts with T_mult 1, 2, 3 stepped at arbitrary,
        # non monotonic epoch values; expected lrs come from listed T_cur and T_i values.
        epochs_for_T_mults = [
            [0, 1, 2, 3, 4, 5, 12, 27, 3, 4, 5, 6, 13],
            [0, 1, 2, 3, 4, 5, 25, 32, 33, 34, 80, 81, 3],
            [0, 0.1, 0.2, 0.3, 1.3, 2.3, 17.5, 18.5, 19.5, 29.5, 30.5, 31.5, 50],
        ]
        T_curs_for_T_mults = [
            [1, 2, 3, 4, 5, 2, 7, 3, 4, 5, 6, 3],
            [1, 2, 3, 4, 5, 15, 2, 3, 4, 10, 11, 3],
            [0.1, 0.2, 0.3, 1.3, 2.3, 7.5, 8.5, 9.5, 19.5, 20.5, 21.5, 10],
        ]
        T_is_for_T_mults = [
            [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10, 20, 40, 40, 40, 80, 80, 10],
            [10, 10, 10, 10, 10, 30, 30, 30, 30, 30, 30, 90],
        ]
        eta_min = 1e-10
        T_mults = [1, 2, 3]
        for epochs, T_mult, T_curs, T_is in zip(
            epochs_for_T_mults, T_mults, T_curs_for_T_mults, T_is_for_T_mults
        ):
            targets = [[0.05], [0.5]]
            scheduler = CosineAnnealingWarmRestarts(
                self.opt, T_0=10, T_mult=T_mult, eta_min=eta_min
            )
            for T_cur, T_i in zip(T_curs, T_is):
                targets[0] += [
                    eta_min
                    + (0.05 - eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2
                ]
                targets[1] += [
                    eta_min
                    + (0.5 - eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2
                ]
            self._test_interleaved_CosineAnnealingWarmRestarts(
                scheduler, targets, epochs
            )

    def test_CosineAnnealingWarmRestarts_T_cur_reset(self):
        # [FUNCTION DOC] Sets T_cur manually to 7, 8, 9 on CosineAnnealingWarmRestarts with T_0
        # 4, steps, and asserts T_cur becomes the next epoch modulo T_0.
        sch = CosineAnnealingWarmRestarts(self.opt, T_0=4)
        for epoch in [7, 8, 9]:
            sch.T_cur = epoch
            sch.step()
            expect_T_cur = (epoch + 1) % sch.T_0
            self.assertEqual(sch.T_cur, expect_T_cur)

    def test_swalr_no_anneal(self):
        # [FUNCTION DOC] SWALR with anneal_epochs 1: lr stays at the initial value until epoch 5
        # then jumps to swa_lr 0.01; runs 10 epochs via _test_swalr without a base scheduler.
        epochs, swa_start, swa_lr = 10, 5, 0.01
        initial_lrs = [group["lr"] for group in self.opt.param_groups]
        targets = [
            [lr] * (swa_start + 1) + [swa_lr] * (epochs - swa_start - 1)
            for lr in initial_lrs
        ]
        swa_scheduler = SWALR(self.opt, anneal_epochs=1, swa_lr=swa_lr)
        self._test_swalr(swa_scheduler, None, targets, swa_start, epochs)

    def test_swalr_cosine_anneal_after_multiplicative(self):
        # [FUNCTION DOC] MultiplicativeLR (factor 0.9) until epoch 5, then SWALR with cosine
        # annealing over 5 epochs toward a shared swa_lr; expected lr blends the last lr and
        # swa_lr by the cosine coefficient.
        # same swa_lr for different param_groups
        epochs, swa_start, swa_lr, anneal_epochs = 15, 5, 0.01, 5
        mult_factor = 0.9
        scheduler = MultiplicativeLR(self.opt, lr_lambda=lambda epoch: mult_factor)
        swa_scheduler = SWALR(self.opt, anneal_epochs=anneal_epochs, swa_lr=swa_lr)

        def anneal_coef(t):
            # [FUNCTION DOC] HELPER: Cosine annealing weight: zero once the annealing is
            # complete, otherwise half of one plus cosine of the progress.
            if t + 1 >= anneal_epochs:
                return 0.0
            return (1 + math.cos(math.pi * (t + 1) / anneal_epochs)) / 2

        initial_lrs = [group["lr"] for group in self.opt.param_groups]
        targets_before_swa = [
            [lr * mult_factor**i for i in range(swa_start + 1)] for lr in initial_lrs
        ]
        swa_epochs = epochs - swa_start - 1
        targets = [
            lrs
            + [
                lrs[-1] * anneal_coef(t) + swa_lr * (1 - anneal_coef(t))
                for t in range(swa_epochs)
            ]
            for lrs in targets_before_swa
        ]

        self._test_swalr(swa_scheduler, scheduler, targets, swa_start, epochs)

    def test_swalr_linear_anneal_after_multiplicative(self):
        # [FUNCTION DOC] Same as above but with linear annealing over 4 epochs and a separate
        # swa_lr for each param group.
        # separate swa_lr for different param_groups
        epochs, swa_start, swa_lrs, anneal_epochs = 15, 5, [0.01, 0.02], 4
        mult_factor = 0.9
        scheduler = MultiplicativeLR(self.opt, lr_lambda=lambda epoch: mult_factor)
        swa_scheduler = SWALR(
            self.opt,
            anneal_epochs=anneal_epochs,
            anneal_strategy="linear",
            swa_lr=swa_lrs,
        )

        def anneal_coef(t):
            # [FUNCTION DOC] HELPER: Linear annealing weight: zero once annealing is complete,
            # otherwise one minus the progress.
            if t + 1 >= anneal_epochs:
                return 0.0
            return 1 - (t + 1) / anneal_epochs

        initial_lrs = [group["lr"] for group in self.opt.param_groups]
        targets_before_swa = [
            [lr * mult_factor**i for i in range(swa_start + 1)] for lr in initial_lrs
        ]
        swa_epochs = epochs - swa_start - 1
        targets = [
            lrs
            + [
                lrs[-1] * anneal_coef(t) + swa_lr * (1 - anneal_coef(t))
                for t in range(swa_epochs)
            ]
            for lrs, swa_lr in zip(targets_before_swa, swa_lrs)
        ]

        self._test_swalr(swa_scheduler, scheduler, targets, swa_start, epochs)

    def _test_swalr(self, swa_scheduler, scheduler, targets, swa_start, epochs):
        # [FUNCTION DOC] HELPER: For each epoch asserts group lrs equal targets, then steps
        # optimizer and swa_scheduler once the epoch reaches swa_start; before that, steps
        # optimizer and scheduler if scheduler is not None.
        for epoch in range(epochs):
            for param_group, target in zip(self.opt.param_groups, targets):
                self.assertEqual(
                    target[epoch],
                    param_group["lr"],
                    msg=lambda msg: f"{msg}\n"
                    + (
                        "LR is wrong in epoch {}: expected {}, got {}".format(
                            epoch, target[epoch], param_group["lr"]
                        )
                    ),
                    atol=1e-5,
                    rtol=0,
                )
            if epoch >= swa_start:
                self.opt.step()
                swa_scheduler.step()
            elif scheduler is not None:
                self.opt.step()
                scheduler.step()

    def test_swalr_hypers(self):
        # [FUNCTION DOC] SWALR must raise ValueError for an unknown anneal strategy, for negative
        # or non integer anneal_epochs, and for a swa_lr list whose length mismatches the groups.
        # Test that SWALR raises errors for incorrect hyper-parameters
        with self.assertRaisesRegex(ValueError, "anneal_strategy must"):
            swa_scheduler = SWALR(self.opt, anneal_strategy="exponential", swa_lr=1.0)

        with self.assertRaisesRegex(ValueError, "anneal_epochs must"):
            swa_scheduler = SWALR(self.opt, anneal_epochs=-1, swa_lr=1.0)
        with self.assertRaisesRegex(ValueError, "anneal_epochs must"):
            swa_scheduler = SWALR(self.opt, anneal_epochs=1.7, swa_lr=1.0)
        with self.assertRaisesRegex(ValueError, "swa_lr must"):
            swa_scheduler = SWALR(self.opt, swa_lr=[1.0, 0.1, 0.01])

    def test_step_lr_state_dict(self):
        # [FUNCTION DOC] State dict round trip for StepLR using the shared check helper with
        # different constructor arguments for the copy.
        self._check_scheduler_state_dict(
            lambda: StepLR(self.opt, gamma=0.1, step_size=3),
            lambda: StepLR(self.opt, gamma=0.01 / 2, step_size=1),
        )

    def test_multi_step_lr_state_dict(self):
        # [FUNCTION DOC] State dict round trip for MultiStepLR via the shared check helper.
        self._check_scheduler_state_dict(
            lambda: MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9]),
            lambda: MultiStepLR(self.opt, gamma=0.01, milestones=[1, 4, 6]),
        )

    def test_exp_step_lr_state_dict(self):
        # [FUNCTION DOC] State dict round trip for ExponentialLR via the shared check helper.
        self._check_scheduler_state_dict(
            lambda: ExponentialLR(self.opt, gamma=0.1),
            lambda: ExponentialLR(self.opt, gamma=0.01),
        )

    def test_cosine_lr_state_dict(self):
        # [FUNCTION DOC] State dict round trip for CosineAnnealingLR via the shared check helper.
        epochs = 10
        eta_min = 1e-10
        self._check_scheduler_state_dict(
            lambda: CosineAnnealingLR(self.opt, T_max=epochs, eta_min=eta_min),
            lambda: CosineAnnealingLR(self.opt, T_max=epochs // 2, eta_min=eta_min / 2),
            epochs=epochs,
        )

    def test_reduce_lr_on_plateau_state_dict(self):
        # [FUNCTION DOC] Steps ReduceLROnPlateau through ten metrics, loads its state dict into a
        # scheduler with different settings, then asserts all attributes except optimizer and
        # is_better match.
        scheduler = ReduceLROnPlateau(self.opt, mode="min", factor=0.1, patience=2)
        for score in [1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 3.0, 2.0, 1.0]:
            scheduler.step(score)
        scheduler_copy = ReduceLROnPlateau(
            self.opt, mode="max", factor=0.5, patience=10
        )
        scheduler_copy.load_state_dict(scheduler.state_dict())
        for key in scheduler.__dict__:
            if key not in {"optimizer", "is_better"}:
                self.assertEqual(scheduler.__dict__[key], scheduler_copy.__dict__[key])

    def test_lambda_lr_state_dict_fn(self):
        # [FUNCTION DOC] LambdaLR with a plain function: the saved state's lambda entry is None;
        # after loading into a copy, all attributes except optimizer and lr_lambdas are equal.
        scheduler = LambdaLR(self.opt, lr_lambda=lambda x: x)
        state = scheduler.state_dict()
        self.assertIsNone(state["lr_lambdas"][0])

        scheduler_copy = LambdaLR(self.opt, lr_lambda=lambda x: x)
        scheduler_copy.load_state_dict(state)
        for key in scheduler.__dict__:
            if key not in {"optimizer", "lr_lambdas"}:
                self.assertEqual(scheduler.__dict__[key], scheduler_copy.__dict__[key])

    def test_lambda_lr_state_dict_obj(self):
        # [FUNCTION DOC] LambdaLR with a callable object: the saved lambda entry is not None;
        # loading into a copy built with a different object makes all attributes except optimizer
        # equal.
        scheduler = LambdaLR(self.opt, lr_lambda=self.LambdaLRTestObject(10))
        state = scheduler.state_dict()
        self.assertIsNotNone(state["lr_lambdas"][0])

        scheduler_copy = LambdaLR(self.opt, lr_lambda=self.LambdaLRTestObject(-1))
        scheduler_copy.load_state_dict(state)
        for key in scheduler.__dict__:
            if key != "optimizer":
                self.assertEqual(scheduler.__dict__[key], scheduler_copy.__dict__[key])

    def test_CosineAnnealingWarmRestarts_lr_state_dict(self):
        # [FUNCTION DOC] State dict round trip for CosineAnnealingWarmRestarts via the shared
        # check helper.
        self._check_scheduler_state_dict(
            lambda: CosineAnnealingWarmRestarts(self.opt, T_0=10, T_mult=2),
            lambda: CosineAnnealingWarmRestarts(self.opt, T_0=100),
        )

    def test_swa_lr_state_dict(self):
        # [FUNCTION DOC] State dict round trip for SWALR via the shared check helper.
        self._check_scheduler_state_dict(
            lambda: SWALR(self.opt, anneal_epochs=3, swa_lr=0.5),
            lambda: SWALR(
                self.opt, anneal_epochs=10, anneal_strategy="linear", swa_lr=5.0
            ),
        )

    def _check_scheduler_state_dict(self, constr, constr2, epochs=10):
        # [FUNCTION DOC] HELPER: Builds a scheduler with constr, runs epochs optimizer and
        # scheduler steps, loads its state dict into a scheduler from constr2 and asserts all
        # attributes except optimizer and get_last_lr match.
        scheduler = constr()
        for _ in range(epochs):
            scheduler.optimizer.step()
            scheduler.step()
        scheduler_copy = constr2()
        scheduler_copy.load_state_dict(scheduler.state_dict())
        for key in scheduler.__dict__:
            if key != "optimizer":
                self.assertEqual(scheduler.__dict__[key], scheduler_copy.__dict__[key])
        self.assertEqual(scheduler.get_last_lr(), scheduler_copy.get_last_lr())

    def _test_get_last_lr(self, schedulers, targets, epochs=10):
        # [FUNCTION DOC] HELPER: Accepts one scheduler or a list. For each epoch reads each
        # scheduler's get_last_lr, steps the optimizers and schedulers, and asserts the lr read
        # before stepping equals the per group targets for that epoch (atol 1e-5).
        if isinstance(schedulers, LRScheduler):
            schedulers = [schedulers]
        optimizers = {scheduler.optimizer for scheduler in schedulers}
        for epoch in range(epochs):
            result = [scheduler.get_last_lr() for scheduler in schedulers]
            [optimizer.step() for optimizer in optimizers]
            [scheduler.step() for scheduler in schedulers]
            target = [[t[epoch] for t in targets]] * len(schedulers)
            for t, r in zip(target, result):
                self.assertEqual(
                    t,
                    r,
                    msg=lambda msg: f"{msg}\nLR is wrong in epoch {epoch}: expected {t}, got {r}",
                    atol=1e-5,
                    rtol=0,
                )

    def _test_with_epoch(self, schedulers, targets, epochs=10):
        # [FUNCTION DOC] HELPER: For each epoch steps the optimizers, then steps each scheduler
        # with the explicit epoch while checking one epoch deprecation warning per scheduler,
        # then asserts self.opt group lrs equal the targets.
        if isinstance(schedulers, LRScheduler):
            schedulers = [schedulers]
        optimizers = {scheduler.optimizer for scheduler in schedulers}
        for epoch in range(epochs):
            [optimizer.step() for optimizer in optimizers]
            with warnings.catch_warnings(record=True) as w:
                [
                    scheduler.step(epoch) for scheduler in schedulers
                ]  # step before assert: skip initial lr
                self._check_warning_is_epoch_deprecation_warning(
                    w, num_warnings=len(schedulers)
                )
            for param_group, target in zip(self.opt.param_groups, targets):
                self.assertEqual(
                    target[epoch],
                    param_group["lr"],
                    msg=lambda msg: f"{msg}\n"
                    + (
                        "LR is wrong in epoch {}: expected {}, got {}".format(
                            epoch, target[epoch], param_group["lr"]
                        )
                    ),
                    atol=1e-5,
                    rtol=0,
                )

    def _test(self, schedulers, targets, epochs=10):
        # [FUNCTION DOC] HELPER: Core check: for each epoch asserts every self.opt group lr
        # equals its target (atol 1e-5, rtol 0), then steps each scheduler. The optimizer itself
        # is not stepped here.
        if isinstance(schedulers, LRScheduler):
            schedulers = [schedulers]
        for epoch in range(epochs):
            for param_group, target in zip(self.opt.param_groups, targets):
                self.assertEqual(
                    target[epoch],
                    param_group["lr"],
                    msg=lambda msg: f"{msg}\n"
                    + (
                        "LR is wrong in epoch {}: expected {}, got {}".format(
                            epoch, target[epoch], param_group["lr"]
                        )
                    ),
                    atol=1e-5,
                    rtol=0,
                )
            [scheduler.step() for scheduler in schedulers]

    def _test_CosineAnnealingWarmRestarts(self, scheduler, targets, epochs=10):
        # [FUNCTION DOC] HELPER: Steps the scheduler with fractional epochs in 0.1 increments and
        # asserts group lrs equal the target at each index.
        for index, epoch in enumerate(torch.arange(0, epochs, 0.1)):
            epoch = round(epoch.item(), 1)
            scheduler.step(epoch)
            for param_group, target in zip(self.opt.param_groups, targets):
                self.assertEqual(
                    target[index],
                    param_group["lr"],
                    msg=lambda msg: f"{msg}\n"
                    + (
                        "LR is wrong in epoch {}: expected {}, got {}".format(
                            epoch, target[index], param_group["lr"]
                        )
                    ),
                    atol=1e-5,
                    rtol=0,
                )

    def _test_interleaved_CosineAnnealingWarmRestarts(self, scheduler, targets, epochs):
        # [FUNCTION DOC] HELPER: Steps the scheduler with each epoch in the given list in order
        # and asserts group lrs equal the target at that position.
        for index, epoch in enumerate(epochs):
            scheduler.step(epoch)
            for param_group, target in zip(self.opt.param_groups, targets):
                self.assertEqual(
                    target[index],
                    param_group["lr"],
                    msg=lambda msg: f"{msg}\n"
                    + (
                        "LR is wrong in epoch {}: expected {}, got {}".format(
                            epoch, target[index], param_group["lr"]
                        )
                    ),
                    atol=1e-5,
                    rtol=0,
                )

    def _test_against_closed_form(self, scheduler, closed_form_scheduler, epochs=10):
        # [FUNCTION DOC] HELPER: Resets self.setUp, steps closed_form_scheduler with explicit
        # epochs recording the group lrs, resets again, then steps scheduler normally and
        # compares lrs.
        # KNOWN ISSUE: setUp rebinds self.opt to a new optimizer, but both schedulers were built
        # over the old one, so the lrs read from self.opt never change and both loops compare
        # unchanged initial values; the closed form comparison is effectively vacuous.
        self.setUp()
        targets = []
        for epoch in range(epochs):
            closed_form_scheduler.optimizer.step()
            with warnings.catch_warnings(record=True) as w:
                closed_form_scheduler.step(epoch)
                self._check_warning_is_epoch_deprecation_warning(w)
            targets.append([group["lr"] for group in self.opt.param_groups])
        self.setUp()
        for epoch in range(epochs):
            self.opt.step()
            scheduler.step()
            for i, param_group in enumerate(self.opt.param_groups):
                self.assertEqual(
                    targets[epoch][i],
                    param_group["lr"],
                    msg=lambda msg: f"{msg}\n"
                    + (
                        "LR is wrong in epoch {}: expected {}, got {}".format(
                            epoch, targets[epoch][i], param_group["lr"]
                        )
                    ),
                    atol=1e-5,
                    rtol=0,
                )

    def _test_reduce_lr_on_plateau(
        self, schedulers, targets, metrics, epochs=10, verbose=False
    ):
        # [FUNCTION DOC] HELPER: Steps the optimizer and then each scheduler (the plateau
        # scheduler gets the metric for that epoch) per epoch, optionally prints the lr, and
        # asserts group lrs equal targets. A single scheduler is accepted too.
        if isinstance(schedulers, (LRScheduler, ReduceLROnPlateau)):
            schedulers = [schedulers]
        for epoch in range(epochs):
            self.opt.step()
            for scheduler in schedulers:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(metrics[epoch])
                else:
                    scheduler.step()
            if verbose:
                print("epoch{}:\tlr={}".format(epoch, self.opt.param_groups[0]["lr"]))
            for param_group, target in zip(self.opt.param_groups, targets):
                self.assertEqual(
                    target[epoch],
                    param_group["lr"],
                    msg=lambda msg: f"{msg}\n"
                    + (
                        "LR is wrong in epoch {}: expected {}, got {}".format(
                            epoch, target[epoch], param_group["lr"]
                        )
                    ),
                    atol=1e-5,
                    rtol=0,
                )

    def _test_cycle_lr(
        self,
        scheduler,
        lr_targets,
        momentum_targets,
        batch_iterations,
        verbose=False,
        use_beta1=False,
    ):
        # [FUNCTION DOC] HELPER: For each batch optionally prints lr and momentum, asserts each
        # group's lr and its momentum (or first beta when use_beta1 is set and betas exist;
        # momentum is only checked when present), then steps the optimizer and scheduler.
        for batch_num in range(batch_iterations):
            if verbose:
                if "momentum" in self.opt.param_groups[0]:
                    print(
                        "batch{}:\tlr={},momentum={}".format(
                            batch_num,
                            self.opt.param_groups[0]["lr"],
                            self.opt.param_groups[0]["momentum"],
                        )
                    )
                elif use_beta1 and "betas" in self.opt.param_groups[0]:
                    print(
                        "batch{}:\tlr={},beta1={}".format(
                            batch_num,
                            self.opt.param_groups[0]["lr"],
                            self.opt.param_groups[0]["betas"][0],
                        )
                    )
                else:
                    print(
                        "batch{}:\tlr={}".format(
                            batch_num, self.opt.param_groups[0]["lr"]
                        )
                    )

            for param_group, lr_target, momentum_target in zip(
                self.opt.param_groups, lr_targets, momentum_targets
            ):
                self.assertEqual(
                    lr_target[batch_num],
                    param_group["lr"],
                    msg=lambda msg: f"{msg}\n"
                    + (
                        "LR is wrong in batch_num {}: expected {}, got {}".format(
                            batch_num, lr_target[batch_num], param_group["lr"]
                        )
                    ),
                    atol=1e-5,
                    rtol=0,
                )

                if use_beta1 and "betas" in param_group:
                    self.assertEqual(
                        momentum_target[batch_num],
                        param_group["betas"][0],
                        msg=lambda msg: f"{msg}\n"
                        + (
                            "Beta1 is wrong in batch_num {}: expected {}, got {}".format(
                                batch_num,
                                momentum_target[batch_num],
                                param_group["betas"][0],
                            )
                        ),
                        atol=1e-5,
                        rtol=0,
                    )
                elif "momentum" in param_group:
                    self.assertEqual(
                        momentum_target[batch_num],
                        param_group["momentum"],
                        msg=lambda msg: f"{msg}\n"
                        + (
                            "Momentum is wrong in batch_num {}: expected {}, got {}".format(
                                batch_num,
                                momentum_target[batch_num],
                                param_group["momentum"],
                            )
                        ),
                        atol=1e-5,
                        rtol=0,
                    )
            self.opt.step()
            scheduler.step()

    def test_cosine_then_cyclic(self):
        # [FUNCTION DOC] Regression for issue 21965: cosine annealing runs while the loop index
        # is at most T_max, then a CyclicLR on the same optimizer takes over for 40 iterations;
        # the final lr must not exceed the CyclicLR max_lr.
        # https://github.com/pytorch/pytorch/issues/21965

        max_lr = 0.3
        base_lr = 0.1
        optim_lr = 0.5

        model = torch.nn.Linear(2, 1)
        optimizer = SGD(model.parameters(), lr=optim_lr)
        lr_scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=20, eta_min=0.1
        )
        lr_scheduler_2 = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=1, step_size_down=3
        )

        for i in range(40):
            optimizer.step()
            if i <= lr_scheduler_1.T_max:
                lr_scheduler_1.step()
            else:
                lr_scheduler_2.step()
            last_lr = optimizer.param_groups[0]["lr"]

        self.assertLessEqual(last_lr, max_lr)

    @parametrize(
        "LRClass",
        [
            partial(LambdaLR, lr_lambda=lambda e: e // 10),
            partial(MultiplicativeLR, lr_lambda=lambda: 0.95),
            partial(StepLR, step_size=30),
            partial(MultiStepLR, milestones=[30, 80]),
            ConstantLR,
            LinearLR,
            partial(ExponentialLR, gamma=0.9),
            PolynomialLR,
            partial(CosineAnnealingLR, T_max=10),
            lambda opt, **kwargs: ChainedScheduler(
                schedulers=[ConstantLR(opt), ConstantLR(opt)], **kwargs
            ),
            lambda opt, **kwargs: SequentialLR(
                opt,
                schedulers=[ConstantLR(opt), ConstantLR(opt)],
                milestones=[2],
                **kwargs,
            ),
            ReduceLROnPlateau,
            partial(CyclicLR, base_lr=0.01, max_lr=0.1),
            partial(OneCycleLR, max_lr=0.01, total_steps=10, anneal_strategy="linear"),
            partial(CosineAnnealingWarmRestarts, T_0=20),
            partial(SWALR, swa_lr=0.01),
        ],
    )
    @parametrize("weights_only", [True, False])
    def test_lr_scheduler_state_dict_load(self, LRClass, weights_only):
        # [FUNCTION DOC] Parametrized over many scheduler classes and weights_only true or false.
        # Saves a scheduler state dict to a temporary file, loads it back, asserts equality,
        # loads it into a second scheduler and checks its state dict equals the original.
        scheduler = LRClass(self.opt)
        state_dict = scheduler.state_dict()

        with tempfile.TemporaryFile() as f:
            torch.save(state_dict, f)
            f.seek(0)
            state_dict_loaded = torch.load(f, weights_only=weights_only)
            self.assertEqual(state_dict, state_dict_loaded)
            # Make sure state_dict can be loaded
            scheduler2 = LRClass(self.opt)
            scheduler2.load_state_dict(state_dict_loaded)
            self.assertEqual(scheduler2.state_dict(), state_dict)

    @parametrize("min_lr", ["scalar", "list"])
    def test_add_param_group_does_not_break_reduce_lr_on_plateau(self, min_lr):
        # [FUNCTION DOC] Parametrized with min_lr as scalar or list. Regression for issue 104361:
        # adds a third param group at epoch 8 (and a matching min_lr for the list case) and
        # checks the first group's lr sequence of ReduceLROnPlateau is unaffected.
        epochs = 20
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        targets = [[0.5] * 6 + [0.05] * (5 + 6) + [0.005] * 4]
        metrics = [1] * 7 + [0.6] + [0.5] * 12
        scheduler = ReduceLROnPlateau(
            self.opt,
            mode="min",
            threshold_mode="rel",
            threshold=0.1,
            patience=5,
            cooldown=5,
            min_lr=0 if min_lr == "scalar" else [1e-5, 1e-4],
        )
        for epoch in range(epochs):
            # Point is to test the use case in #104361
            if epoch == 8:
                param = torch.nn.Parameter(torch.rand(2, 3))
                self.opt.add_param_group({"params": [param], "lr": 0.05})
                if min_lr == "list":
                    scheduler.min_lrs.append(1e-6)
            self.opt.step()
            scheduler.step(metrics[epoch])
            for param_group, target in zip(self.opt.param_groups, targets):
                self.assertEqual(
                    target[epoch],
                    param_group["lr"],
                    msg=lambda msg: f"{msg}\n"
                    + (
                        "LR is wrong in epoch {}: expected {}, got {}".format(
                            epoch, target[epoch], param_group["lr"]
                        )
                    ),
                    atol=1e-5,
                    rtol=0,
                )

    def test_add_param_group_errors_reduce_lr_on_plateau(self):
        # [FUNCTION DOC] Adding a param group after building ReduceLROnPlateau with a two entry
        # min_lr list must raise RuntimeError about the number of param groups when the scheduler
        # later tries to reduce the lr.
        scheduler = ReduceLROnPlateau(
            self.opt,
            mode="min",
            threshold_mode="rel",
            threshold=1e-5,
            patience=0,
            cooldown=0,
            min_lr=[1e-5, 1e-4],
        )
        param = torch.nn.Parameter(torch.rand(2, 3))
        self.opt.add_param_group({"params": [param], "lr": 0.05})
        self.opt.step()
        scheduler.step(1)
        with self.assertRaisesRegex(RuntimeError, "The number of param groups in the"):
            self.opt.step()
            scheduler.step(1.3)

    @parametrize(
        "LRClass",
        [
            partial(LambdaLR, lr_lambda=lambda e: e // 10),
            partial(MultiplicativeLR, lr_lambda=lambda e: 0.95),
            partial(StepLR, step_size=30),
            partial(MultiStepLR, milestones=[30, 80]),
            ConstantLR,
            LinearLR,
            partial(ExponentialLR, gamma=0.9),
            PolynomialLR,
            partial(CosineAnnealingLR, T_max=10),
            partial(CosineAnnealingWarmRestarts, T_0=20),
        ],
    )
    def test_constant_initial_lr(self, LRClass):
        # [FUNCTION DOC] Parametrized over several scheduler classes. With a tensor lr 0.1, steps
        # with epochs while multiplying the lr in place; initial_lr in the groups must stay
        # constant, base_lrs must remain 0.1 and must not be the same object as initial_lr.
        # Test that the initial learning rate is constant and that it does not alias base_lrs
        lr = torch.as_tensor(0.1)
        opt = SGD([torch.nn.Parameter(torch.randn(1))], lr=lr)
        sch = LRClass(opt)

        ori_param_groups = copy.deepcopy(opt.param_groups)

        for i in range(2):
            opt.step()
            sch.step(i)
            lr.multiply_(0.1)
            for group, ori_group in zip(opt.param_groups, ori_param_groups):
                self.assertEqual(group["initial_lr"], ori_group["initial_lr"])
                self.assertEqual(sch.base_lrs, [0.1])
                self.assertIsNot(sch.base_lrs[0], group["initial_lr"])

    def test_constant_initial_params_cyclelr(self):
        # [FUNCTION DOC] CyclicLR built from tensor lrs and momenta that are halved in place each
        # iteration; group initial_lr and momentum bounds, base_lrs and max_lrs must stay at the
        # original values.
        # Test that the initial learning rate is constant
        lr = torch.as_tensor(0.1)
        max_lr = torch.as_tensor(0.2)
        base_momentum = torch.as_tensor(0.8)
        max_momentum = torch.as_tensor(0.9)
        opt = SGD([torch.nn.Parameter(torch.randn(1))], lr=lr)
        sch = CyclicLR(
            opt,
            base_lr=lr,
            max_lr=max_lr,
            base_momentum=base_momentum,
            max_momentum=max_momentum,
        )
        ori_param_groups = copy.deepcopy(opt.param_groups)

        for i in range(2):
            lr.multiply_(0.5)
            max_lr.multiply_(0.5)
            base_momentum.multiply_(0.5)
            max_momentum.multiply_(0.5)
            opt.step()
            sch.step(i)
            for group, ori_group in zip(opt.param_groups, ori_param_groups):
                self.assertEqual(group["initial_lr"], ori_group["initial_lr"])
                self.assertEqual(group["max_momentum"], ori_group["max_momentum"])
                self.assertEqual(group["base_momentum"], ori_group["base_momentum"])
                self.assertEqual(sch.base_lrs, [0.1])
                self.assertEqual(sch.max_lrs, [0.2])
                self.assertEqual(group["max_momentum"], 0.9)
                self.assertEqual(group["base_momentum"], 0.8)

    def test_constant_initial_params_onecyclelr(self):
        # [FUNCTION DOC] OneCycleLR built from tensor lr and momenta halved in place each
        # iteration; group initial_lr, max_lr, min_lr and momentum bounds must stay at their
        # original values.
        # Test that the initial learning rate is constant
        lr = torch.as_tensor(0.1)
        base_momentum = torch.as_tensor(0.85)
        max_momentum = torch.as_tensor(0.95)
        opt = SGD([torch.nn.Parameter(torch.randn(1))], lr=lr)
        sch = OneCycleLR(
            opt,
            max_lr=lr,
            total_steps=10,
            base_momentum=base_momentum,
            max_momentum=max_momentum,
        )
        ori_param_groups = copy.deepcopy(opt.param_groups)

        for i in range(2):
            lr.multiply_(0.5)
            base_momentum.multiply_(0.5)
            max_momentum.multiply_(0.5)
            opt.step()
            sch.step(i)

            for group, ori_group in zip(opt.param_groups, ori_param_groups):
                self.assertEqual(group["initial_lr"], ori_group["initial_lr"])
                self.assertEqual(group["max_lr"], ori_group["max_lr"])
                self.assertEqual(group["min_lr"], ori_group["min_lr"])
                self.assertEqual(group["max_momentum"], ori_group["max_momentum"])
                self.assertEqual(group["base_momentum"], ori_group["base_momentum"])
                self.assertEqual(group["max_momentum"], 0.95)
                self.assertEqual(group["base_momentum"], 0.85)

    def test_constant_initial_params_swalr(self):
        # [FUNCTION DOC] SWALR built from tensor lr and swa_lr halved in place; initial_lr,
        # swa_lr (0.05) and base_lrs (0.1) must stay constant.
        # Test that the initial learning rate is constant
        lr = torch.as_tensor(0.1)
        swa_lr = torch.as_tensor(0.05)
        opt = SGD([torch.nn.Parameter(torch.randn(1))], lr=lr)
        sch = SWALR(opt, swa_lr=swa_lr)
        ori_param_groups = copy.deepcopy(opt.param_groups)

        for _ in range(2):
            lr.multiply_(0.5)
            swa_lr.multiply_(0.5)
            opt.step()
            sch.step()
            for group, ori_group in zip(opt.param_groups, ori_param_groups):
                self.assertEqual(group["initial_lr"], ori_group["initial_lr"])
                self.assertEqual(group["swa_lr"], ori_group["swa_lr"])
                self.assertEqual(group["swa_lr"], 0.05)
                self.assertEqual(sch.base_lrs, [0.1])

    @parametrize(
        "LRClass",
        [
            partial(ExponentialLR, gamma=0.999),
            partial(LambdaLR, lr_lambda=lambda epoch: epoch // 30),
            partial(MultiplicativeLR, lr_lambda=lambda epoch: 0.95),
            partial(StepLR, step_size=30),
            partial(MultiStepLR, milestones=[30, 80]),
            ConstantLR,
            LinearLR,
            PolynomialLR,
            partial(CosineAnnealingLR, T_max=10),
            partial(CosineAnnealingWarmRestarts, T_0=20),
            partial(CyclicLR, base_lr=0.01, max_lr=0.1),
            partial(OneCycleLR, max_lr=0.01, total_steps=10),
            partial(SWALR, swa_lr=0.01),
        ],
    )
    def test_lr_scheduler_checkpoint(self, LRClass):
        # [FUNCTION DOC] Parametrized over many scheduler classes. Steps a scheduler on AdamW,
        # copies the optimizer state to a fresh AdamW, builds a new scheduler with last_epoch 0
        # and asserts its lr equals the original optimizer lr.
        # KNOWN ISSUE: the hasattr check is done on the test case, which never has a closed form
        # method, so the closed form branch is dead and the last lr branch is always used.
        model = torch.nn.Linear(3, 3)
        optim = torch.optim.AdamW(model.parameters())
        sch = LRClass(optim)
        optim.step()
        sch.step()
        optim2 = torch.optim.AdamW(model.parameters())
        optim2.load_state_dict(optim.state_dict())
        sch2 = LRClass(optim2, last_epoch=0)
        self.assertEqual(
            sch2._get_closed_form_lr()[0]
            if hasattr(self, "_get_closed_form_lr")
            else sch2.get_last_lr()[0],
            optim.param_groups[0]["lr"],
        )

    def test_lr_scheduler_checkpoint_on_plateau(self):
        # [FUNCTION DOC] Same checkpoint idea for ReduceLROnPlateau: steps with a metric, copies
        # optimizer state, builds a new plateau scheduler and compares its lr with the original
        # lr.
        # KNOWN ISSUE: the closed form hasattr check runs on the test case, so only the last lr
        # branch is ever used.
        model = torch.nn.Linear(3, 3)
        optim = torch.optim.AdamW(model.parameters())
        sch = ReduceLROnPlateau(optim, mode="min")
        optim.step()
        sch.step(1)
        optim2 = torch.optim.AdamW(model.parameters())
        optim2.load_state_dict(optim.state_dict())
        sch2 = ReduceLROnPlateau(optim2, mode="min")
        self.assertEqual(
            sch2._get_closed_form_lr()[0]
            if hasattr(self, "_get_closed_form_lr")
            else sch2.get_last_lr()[0],
            optim.param_groups[0]["lr"],
        )

    # NOTE: Expected-failure tests for known scheduler issues.
    #
    # These tests should be converted to normal regression tests once
    # the underlying behavior is fixed.

    @expectedFailure
    def test_sequentiallr_resume_reproducibility(self):
        # [FUNCTION DOC] Expected failure (see the comment above). Saves optimizer and
        # SequentialLR state mid run and restores into fresh objects; the resumed lr sequence
        # should match the uninterrupted one but does not because loading does not resync the
        # optimizer lr.
        # KNOWN ISSUE: marked as an expected failure for the SequentialLR restore bug explained
        # in the comment inside the body.
        # Note: Saving and restoring both the optimizer and SequentialLR
        # state mid training should reproduce the same LR sequence as a
        # continuous uninterrupted run.
        #
        # This currently fails around scheduler transition boundaries after
        # restoring from checkpoint state.
        #
        # The problematic state transition comes from SequentialLR's restore
        # model:
        # 1. SequentialLR.__init__() sets last_epoch, resets each param group
        #    lr back to initial_lr, calls recursive_undo(), and then replays
        #    _schedulers[0]._initial_step().
        # 2. At this resume point, that constructor path leaves the optimizer
        #    lr at 0.05 even though the saved scheduler state corresponds to
        #    an effective lr of 0.08.
        # 3. SequentialLR.load_state_dict() restores scheduler fields like
        #    last_epoch and _last_lr, but it does not re-synchronize the
        #    optimizer param-group lr with that loaded scheduler state.
        # 4. LinearLR.get_lr() is recursive: it multiplies the current
        #    optimizer lr, so the next resumed step advances
        #    0.05 -> 0.05625 instead of the uninterrupted run's 0.08 -> 0.09.

        base_lr = 0.1
        milestone = 5
        total_steps = 8
        resume_step = 3

        def make_scheduler(optim):
            # [FUNCTION DOC] HELPER: Builds a SequentialLR with LinearLR (start 0.5) then
            # ExponentialLR gamma 0.5 and the given milestone on the supplied optimizer.
            return SequentialLR(
                optim,
                [
                    LinearLR(optim, start_factor=0.5, total_iters=milestone),
                    ExponentialLR(optim, gamma=0.5),
                ],
                milestones=[milestone],
            )

        # run the full schedule and record the LR.
        model = torch.nn.Linear(1, 1)
        optim = torch.optim.SGD(model.parameters(), lr=base_lr)
        sched = make_scheduler(optim)

        reference_lrs = []
        for _ in range(total_steps):
            optim.step()
            sched.step()
            reference_lrs.append(sched.get_last_lr()[0])

        # run a fresh optimizer/scheduler pair up to an intermediate step
        model2 = torch.nn.Linear(1, 1)
        optim2 = torch.optim.SGD(model2.parameters(), lr=base_lr)
        sched2 = make_scheduler(optim2)

        for _ in range(resume_step):
            optim2.step()
            sched2.step()

        # save state to simulate checkpointing.
        optim_state = optim2.state_dict()
        sched_state = sched2.state_dict()

        # restore into a new optimizer/scheduler pair
        model3 = torch.nn.Linear(1, 1)
        optim3 = torch.optim.SGD(model3.parameters(), lr=base_lr)
        optim3.load_state_dict(optim_state)

        sched3 = make_scheduler(optim3)
        sched3.load_state_dict(sched_state)

        loaded_optimizer_lr = optim3.param_groups[0]["lr"]
        loaded_scheduler_lr = sched3.get_last_lr()[0]

        resumed_lrs = []
        for _ in range(resume_step, total_steps):
            optim3.step()
            sched3.step()
            resumed_lrs.append(sched3.get_last_lr()[0])

        self.assertEqual(loaded_optimizer_lr, loaded_scheduler_lr)
        self.assertEqual(resumed_lrs[0], reference_lrs[resume_step])
        self.assertEqual(resumed_lrs, reference_lrs[resume_step:])


instantiate_parametrized_tests(TestLRScheduler)


if __name__ == "__main__":
    print("These tests should be run through test/test_optim.py instead")
