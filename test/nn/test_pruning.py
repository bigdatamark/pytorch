# Owner(s): ["module: nn"]
# [MODULE DOC] Tests for the pruning utilities in the nn utils prune module: amount validation,
# mask computation for random, L1, Ln, custom and global pruning, the pruning container,
# reparametrization bookkeeping (original parameter, mask buffer, removal), rollback on failure,
# pickling and state dict serialization, and RNN weight handling. Registers the parametrized test
# class at import time and runs the tests when executed as a script.
import pickle
import unittest
import unittest.mock as mock

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    run_tests,
    TemporaryFileName,
    TEST_NUMPY,
)


class TestPruningNN(NNTestCase):
    # [CLASS DOC] Test case for pruning behavior; declares a generic hardware classification and
    # enables the CUDA memory leak check and non-default CUDA stream mode. Each test builds its
    # own small modules, so there is no shared state.
    hw_classification = HardwareClassification.GENERIC
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    # torch/nn/utils/prune.py
    @unittest.skipIf(not TEST_NUMPY, "numpy not found")
    def test_validate_pruning_amount_init(self):
        # [FUNCTION DOC] Checks the init-time amount validation helper: a non-number raises
        # TypeError, a fraction outside zero to one or a negative integer raises ValueError,
        # and assorted valid fractions and non-negative integers raise nothing. Skipped
        # without numpy.
        r"""Test the first util function that validates the pruning
        amount requested by the user the moment the pruning method
        is initialized. This test checks that the expected errors are
        raised whenever the amount is invalid.
        The original function runs basic type checking + value range checks.
        It doesn't check the validity of the pruning amount with
        respect to the size of the tensor to prune. That's left to
        `_validate_pruning_amount`, tested below.
        """
        # neither float not int should raise TypeError
        with self.assertRaises(TypeError):
            prune._validate_pruning_amount_init(amount="I'm a string")

        # float not in [0, 1] should raise ValueError
        with self.assertRaises(ValueError):
            prune._validate_pruning_amount_init(amount=1.1)
        with self.assertRaises(ValueError):
            prune._validate_pruning_amount_init(amount=20.0)

        # negative int should raise ValueError
        with self.assertRaises(ValueError):
            prune._validate_pruning_amount_init(amount=-10)

        # all these should pass without errors because they're valid amounts
        prune._validate_pruning_amount_init(amount=0.34)
        prune._validate_pruning_amount_init(amount=1500)
        prune._validate_pruning_amount_init(amount=0)
        prune._validate_pruning_amount_init(amount=0.0)
        prune._validate_pruning_amount_init(amount=1)
        prune._validate_pruning_amount_init(amount=1.0)
        self.assertTrue(True)

    @unittest.skipIf(not TEST_NUMPY, "numpy not found")
    def test_validate_pruning_amount(self):
        # [FUNCTION DOC] Checks the size-aware amount validation helper: an integer amount larger
        # than the tensor size raises ValueError, while fractional amounts and amounts that fit
        # pass. Skipped without numpy.
        r"""Tests the second util function that validates the pruning
        amount requested by the user, this time with respect to the size
        of the tensor to prune. The rationale is that if the pruning amount,
        converted to absolute value of units to prune, is larger than
        the number of units in the tensor, then we expect the util function
        to raise a value error.
        """
        # if amount is int and amount > tensor_size, raise ValueError
        with self.assertRaises(ValueError):
            prune._validate_pruning_amount(amount=20, tensor_size=19)

        # amount is a float so this should not raise an error
        prune._validate_pruning_amount(amount=0.3, tensor_size=0)

        # this is okay
        prune._validate_pruning_amount(amount=19, tensor_size=20)
        prune._validate_pruning_amount(amount=0, tensor_size=0)
        prune._validate_pruning_amount(amount=1, tensor_size=1)
        self.assertTrue(True)

    @unittest.skipIf(not TEST_NUMPY, "numpy not found")
    def test_compute_nparams_to_prune(self):
        # [FUNCTION DOC] Checks conversion of an amount into a number of units to prune: integers
        # are taken as counts, a float is a fraction of the tensor size (rounded, for example
        # forty percent of seventeen gives seven), zero gives zero. Skipped without numpy.
        r"""Test that requested pruning `amount` gets translated into the
        correct absolute number of units to prune.
        """
        self.assertEqual(prune._compute_nparams_toprune(amount=0, tensor_size=15), 0)
        self.assertEqual(prune._compute_nparams_toprune(amount=10, tensor_size=15), 10)
        # if 1 is int, means 1 unit
        self.assertEqual(prune._compute_nparams_toprune(amount=1, tensor_size=15), 1)
        # if 1. is float, means 100% of units
        self.assertEqual(prune._compute_nparams_toprune(amount=1.0, tensor_size=15), 15)
        self.assertEqual(prune._compute_nparams_toprune(amount=0.4, tensor_size=17), 7)

    def test_random_pruning_sizes(self):
        # [FUNCTION DOC] For linear and 3D convolution modules and for weight and bias, applies
        # random unstructured pruning and checks the mask, original copy and pruned tensor all
        # have the size of the original tensor.
        r"""Test that the new parameters and buffers created by the pruning
        method have the same size as the input tensor to prune. These, in
        fact, correspond to the pruned version of the tensor itself, its
        mask, and its original copy, so the size must match.
        """
        # fixturize test
        # TODO: add other modules
        modules = [nn.Linear(5, 7), nn.Conv3d(2, 2, 2)]
        names = ["weight", "bias"]

        for m in modules:
            for name in names:
                with self.subTest(m=m, name=name):
                    original_tensor = getattr(m, name)

                    prune.random_unstructured(m, name=name, amount=0.1)
                    # mask has the same size as tensor being pruned
                    self.assertEqual(
                        original_tensor.size(), getattr(m, name + "_mask").size()
                    )
                    # 'orig' tensor has the same size as the original tensor
                    self.assertEqual(
                        original_tensor.size(), getattr(m, name + "_orig").size()
                    )
                    # new tensor has the same size as the original tensor
                    self.assertEqual(original_tensor.size(), getattr(m, name).size())

    def test_random_pruning_orig(self):
        # [FUNCTION DOC] After random unstructured pruning of weight and bias of a linear
        # and a 3D convolution module, the stored original-parameter copy equals the
        # pre-pruning tensor.
        r"""Test that original tensor is correctly stored in 'orig'
        after pruning is applied. Important to make sure we don't
        lose info about the original unpruned parameter.
        """
        # fixturize test
        # TODO: add other modules
        modules = [nn.Linear(5, 7), nn.Conv3d(2, 2, 2)]
        names = ["weight", "bias"]

        for m in modules:
            for name in names:
                with self.subTest(m=m, name=name):
                    # tensor prior to pruning
                    original_tensor = getattr(m, name)
                    prune.random_unstructured(m, name=name, amount=0.1)
                    self.assertEqual(original_tensor, getattr(m, name + "_orig"))

    def test_random_pruning_new_weight(self):
        # [FUNCTION DOC] After random pruning the module attribute must equal the stored original
        # parameter multiplied by the mask cast to the tensor dtype.
        r"""Test that module.name now contains a pruned version of
        the original tensor obtained from multiplying it by the mask.
        """
        # fixturize test
        # TODO: add other modules
        modules = [nn.Linear(5, 7), nn.Conv3d(2, 2, 2)]
        names = ["weight", "bias"]

        for m in modules:
            for name in names:
                with self.subTest(m=m, name=name):
                    # tensor prior to pruning
                    original_tensor = getattr(m, name)
                    prune.random_unstructured(m, name=name, amount=0.1)
                    # weight = weight_orig * weight_mask
                    self.assertEqual(
                        getattr(m, name),
                        getattr(m, name + "_orig")
                        * getattr(m, name + "_mask").to(dtype=original_tensor.dtype),
                    )

    def test_identity_pruning(self):
        # [FUNCTION DOC] Checks that the identity pruning method (all-ones mask) leaves forward
        # output, weight and bias gradients unchanged compared with the unpruned module, and that
        # two forward calls in a row agree. Gradients land on the original weight parameter.
        r"""Test that a mask of 1s does not change forward or backward."""
        input_ = torch.ones(1, 5)
        m = nn.Linear(5, 2)
        y_prepruning = m(input_)  # output prior to pruning

        # compute grad pre-pruning and check it's equal to all ones
        y_prepruning.sum().backward()
        old_grad_weight = m.weight.grad.clone()  # don't grab pointer!
        self.assertEqual(old_grad_weight, torch.ones_like(m.weight))
        old_grad_bias = m.bias.grad.clone()
        self.assertEqual(old_grad_bias, torch.ones_like(m.bias))

        # remove grads
        m.zero_grad()

        # force the mask to be made of all 1s
        prune.identity(m, name="weight")

        # with mask of 1s, output should be identical to no mask
        y_postpruning = m(input_)
        self.assertEqual(y_prepruning, y_postpruning)

        # with mask of 1s, grad should be identical to no mask
        y_postpruning.sum().backward()
        self.assertEqual(old_grad_weight, m.weight_orig.grad)
        self.assertEqual(old_grad_bias, m.bias.grad)

        # calling forward twice in a row shouldn't change output
        y1 = m(input_)
        y2 = m(input_)
        self.assertEqual(y1, y2)

    def test_random_pruning_0perc(self):
        # [FUNCTION DOC] Same invariants as the identity test but gets the all-ones mask by
        # patching the random pruning method's mask computation to return ones, so the requested
        # amount is ignored; checks outputs, gradients and repeated forward consistency.
        # KNOWN ISSUE: the docstring is copied from the identity pruning test and does not
        # mention the mock.
        r"""Test that a mask of 1s does not change forward or backward."""
        input_ = torch.ones(1, 5)
        m = nn.Linear(5, 2)
        y_prepruning = m(input_)  # output prior to pruning

        # compute grad pre-pruning and check it's equal to all ones
        y_prepruning.sum().backward()
        old_grad_weight = m.weight.grad.clone()  # don't grab pointer!
        self.assertEqual(old_grad_weight, torch.ones_like(m.weight))
        old_grad_bias = m.bias.grad.clone()
        self.assertEqual(old_grad_bias, torch.ones_like(m.bias))

        # remove grads
        m.zero_grad()

        # force the mask to be made of all 1s
        with mock.patch(
            "torch.nn.utils.prune.RandomUnstructured.compute_mask"
        ) as compute_mask:
            compute_mask.return_value = torch.ones_like(m.weight)
            prune.random_unstructured(
                m, name="weight", amount=0.9
            )  # amount won't count

        # with mask of 1s, output should be identical to no mask
        y_postpruning = m(input_)
        self.assertEqual(y_prepruning, y_postpruning)

        # with mask of 1s, grad should be identical to no mask
        y_postpruning.sum().backward()
        self.assertEqual(old_grad_weight, m.weight_orig.grad)
        self.assertEqual(old_grad_bias, m.bias.grad)

        # calling forward twice in a row shouldn't change output
        y1 = m(input_)
        y2 = m(input_)
        self.assertEqual(y1, y2)

    def test_random_pruning(self):
        # [FUNCTION DOC] Patches the random pruning mask computation to a hand-made mask with
        # two zeros; checks the original weight gradient equals that mask, the bias gradient is
        # ones, and after a manual gradient step the masked entries of the original weight do
        # not change.
        input_ = torch.ones(1, 5)
        m = nn.Linear(5, 2)

        # define custom mask to assign with mock
        mask = torch.ones_like(m.weight)
        mask[1, 0] = 0
        mask[0, 3] = 0

        # check grad is zero for masked weights
        with mock.patch(
            "torch.nn.utils.prune.RandomUnstructured.compute_mask"
        ) as compute_mask:
            compute_mask.return_value = mask
            prune.random_unstructured(m, name="weight", amount=0.9)

        y_postpruning = m(input_)
        y_postpruning.sum().backward()
        # weight_orig is the parameter, so it's the tensor that will accumulate the grad
        self.assertEqual(m.weight_orig.grad, mask)  # all 1s, except for masked units
        self.assertEqual(m.bias.grad, torch.ones_like(m.bias))

        # make sure that weight_orig update doesn't modify [1, 0] and [0, 3]
        old_weight_orig = m.weight_orig.clone()
        # update weights
        learning_rate = 1.0
        for p in m.parameters():
            p.data.sub_(p.grad.data * learning_rate)
        # since these are pruned, they should not be updated
        self.assertEqual(old_weight_orig[1, 0], m.weight_orig[1, 0])
        self.assertEqual(old_weight_orig[0, 3], m.weight_orig[0, 3])

    def test_random_pruning_forward(self):
        # [FUNCTION DOC] Patches the mask to keep only two weight entries and checks each output
        # of a linear layer on a ones input equals the single kept original weight plus the bias.
        r"""check forward with mask (by hand)."""
        input_ = torch.ones(1, 5)
        m = nn.Linear(5, 2)

        # define custom mask to assign with mock
        mask = torch.zeros_like(m.weight)
        mask[1, 0] = 1
        mask[0, 3] = 1

        with mock.patch(
            "torch.nn.utils.prune.RandomUnstructured.compute_mask"
        ) as compute_mask:
            compute_mask.return_value = mask
            prune.random_unstructured(m, name="weight", amount=0.9)

        yhat = m(input_)
        self.assertEqual(yhat[0, 0], m.weight_orig[0, 3] + m.bias[0])
        self.assertEqual(yhat[0, 1], m.weight_orig[1, 0] + m.bias[1])

    def test_remove_pruning_forward(self):
        # [FUNCTION DOC] Prunes with a patched hand-made mask, then removes the pruning and
        # checks the forward output stays the same as in the pruned state.
        r"""Remove pruning and check forward is unchanged from previous
        pruned state.
        """
        input_ = torch.ones(1, 5)
        m = nn.Linear(5, 2)

        # define custom mask to assign with mock
        mask = torch.ones_like(m.weight)
        mask[1, 0] = 0
        mask[0, 3] = 0

        # check grad is zero for masked weights
        with mock.patch(
            "torch.nn.utils.prune.RandomUnstructured.compute_mask"
        ) as compute_mask:
            compute_mask.return_value = mask
            prune.random_unstructured(m, name="weight", amount=0.9)

        y_postpruning = m(input_)

        prune.remove(m, "weight")

        y_postremoval = m(input_)
        self.assertEqual(y_postpruning, y_postremoval)

    def test_pruning_id_consistency(self):
        # [FUNCTION DOC] Checks that pruning and then removing pruning keeps the identity of
        # the first parameter object unchanged, so existing optimizers still reference the
        # right parameter.
        r"""Test that pruning doesn't change the id of the parameters, which
        would otherwise introduce issues with pre-existing optimizers that
        point to old parameters.
        """
        m = nn.Linear(5, 2, bias=False)

        tensor_id = id(next(iter(m.parameters())))

        prune.random_unstructured(m, name="weight", amount=0.9)
        self.assertEqual(tensor_id, id(next(iter(m.parameters()))))

        prune.remove(m, "weight")
        self.assertEqual(tensor_id, id(next(iter(m.parameters()))))

    def test_random_pruning_pickle(self):
        # [FUNCTION DOC] Prunes weight and bias of a linear and a 3D convolution module, then
        # pickles and unpickles the module and checks the copy has the same type.
        modules = [nn.Linear(5, 7), nn.Conv3d(2, 2, 2)]
        names = ["weight", "bias"]

        for m in modules:
            for name in names:
                with self.subTest(m=m, name=name):
                    prune.random_unstructured(m, name=name, amount=0.1)
                    m_new = pickle.loads(pickle.dumps(m))
                    self.assertIsInstance(m_new, type(m))

    def test_multiple_pruning_calls(self):
        # [FUNCTION DOC] Applies L1 unstructured then Ln structured pruning on the same weight
        # and checks the forward hook becomes a pruning container of length two with the right
        # tensor name, entries in order, and a mask that keeps earlier zeros; prunes a third time
        # with an infinity norm and checks the container tensor name again.
        # if you call pruning twice, the hook becomes a PruningContainer
        m = nn.Conv3d(2, 2, 2)
        prune.l1_unstructured(m, name="weight", amount=0.1)
        weight_mask0 = m.weight_mask  # save it for later sanity check

        # prune again
        prune.ln_structured(m, name="weight", amount=0.3, n=2, dim=0)
        hook = next(iter(m._forward_pre_hooks.values()))
        self.assertIsInstance(hook, torch.nn.utils.prune.PruningContainer)
        # check that container._tensor_name is correctly set no matter how
        # many pruning methods are in the container
        self.assertEqual(hook._tensor_name, "weight")

        # check that the pruning container has the right length
        # equal to the number of pruning iters
        self.assertEqual(len(hook), 2)  # m.weight has been pruned twice

        # check that the entries of the pruning container are of the expected
        # type and in the expected order
        self.assertIsInstance(hook[0], torch.nn.utils.prune.L1Unstructured)
        self.assertIsInstance(hook[1], torch.nn.utils.prune.LnStructured)

        # check that all entries that are 0 in the 1st mask are 0 in the
        # 2nd mask too
        self.assertTrue(torch.all(m.weight_mask[weight_mask0 == 0] == 0))

        # prune again
        prune.ln_structured(m, name="weight", amount=0.1, n=float("inf"), dim=1)
        # check that container._tensor_name is correctly set no matter how
        # many pruning methods are in the container
        hook = next(iter(m._forward_pre_hooks.values()))
        self.assertEqual(hook._tensor_name, "weight")

    def test_pruning_container(self):
        # [FUNCTION DOC] Checks pruning container method addition: an added method with the same
        # tensor name works; one with a different tensor name raises ValueError; adding a non
        # pruning method object (a number or a string) raises TypeError.
        # create an empty container
        container = prune.PruningContainer()
        container._tensor_name = "test"
        self.assertEqual(len(container), 0)

        p = prune.L1Unstructured(amount=2)
        p._tensor_name = "test"

        # test adding a pruning method to a container
        container.add_pruning_method(p)

        # test error raised if tensor name is different
        q = prune.L1Unstructured(amount=2)
        q._tensor_name = "another_test"
        with self.assertRaises(ValueError):
            container.add_pruning_method(q)

        # test that adding a non-pruning method object to a pruning container
        # raises a TypeError
        with self.assertRaises(TypeError):
            container.add_pruning_method(10)
        with self.assertRaises(TypeError):
            container.add_pruning_method("ugh")

    def test_pruning_container_compute_mask(self):
        # [FUNCTION DOC] Checks the container mask computation with a known tensor and prior
        # mask: after an unstructured L1 step, then adding a structured pruning of one row, then
        # of one column, the computed mask equals the hand-derived expected masks each time.
        r"""Test `compute_mask` of pruning container with a known `t` and
        `default_mask`. Indirectly checks that Ln structured pruning is
        acting on the right axis.
        """
        # create an empty container
        container = prune.PruningContainer()
        container._tensor_name = "test"

        # 1) test unstructured pruning
        # create a new pruning method
        p = prune.L1Unstructured(amount=2)
        p._tensor_name = "test"
        # add the pruning method to the container
        container.add_pruning_method(p)

        # create tensor to be pruned
        t = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).to(dtype=torch.float32)
        # create prior mask by hand
        default_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 1]])
        # since we are pruning the two lowest magnitude units, the outcome of
        # the calculation should be this:
        expected_mask = torch.tensor([[0, 0, 1, 0], [1, 1, 0, 1]], dtype=torch.float32)
        computed_mask = container.compute_mask(t, default_mask)
        self.assertEqual(expected_mask, computed_mask)

        # 2) test structured pruning
        q = prune.LnStructured(amount=1, n=2, dim=0)
        q._tensor_name = "test"
        container.add_pruning_method(q)
        # since we are pruning the lowest magnitude one of the two rows, the
        # outcome of the calculation should be this:
        expected_mask = torch.tensor([[0, 0, 0, 0], [1, 1, 0, 1]], dtype=torch.float32)
        computed_mask = container.compute_mask(t, default_mask)
        self.assertEqual(expected_mask, computed_mask)

        # 2) test structured pruning, along another axis
        r = prune.LnStructured(amount=1, n=2, dim=1)
        r._tensor_name = "test"
        container.add_pruning_method(r)
        # since we are pruning the lowest magnitude of the four columns, the
        # outcome of the calculation should be this:
        expected_mask = torch.tensor([[0, 1, 1, 0], [0, 1, 0, 1]], dtype=torch.float32)
        computed_mask = container.compute_mask(t, default_mask)
        self.assertEqual(expected_mask, computed_mask)

    def test_l1_unstructured_pruning(self):
        # [FUNCTION DOC] Checks L1 unstructured pruning removes the two smallest magnitude
        # weights of a hand-set weight matrix, and a second call prunes the next two while
        # keeping the earlier mask.
        r"""Test that l1 unstructured pruning actually removes the lowest
        entries by l1 norm (by hand). It also checks that applying l1
        unstructured pruning more than once respects the previous mask.
        """
        m = nn.Linear(4, 2)
        # modify its weight matrix by hand
        m.weight = torch.nn.Parameter(
            torch.tensor([[1, 2, 3, 4], [-4, -3, -2, -1]], dtype=torch.float32)
        )

        prune.l1_unstructured(m, "weight", amount=2)
        expected_weight = torch.tensor(
            [[0, 2, 3, 4], [-4, -3, -2, 0]], dtype=m.weight.dtype
        )
        self.assertEqual(expected_weight, m.weight)

        # check that pruning again removes the next two smallest entries
        prune.l1_unstructured(m, "weight", amount=2)
        expected_weight = torch.tensor(
            [[0, 0, 3, 4], [-4, -3, 0, 0]], dtype=m.weight.dtype
        )
        self.assertEqual(expected_weight, m.weight)

    def test_l1_unstructured_pruning_with_importance_scores(self):
        # [FUNCTION DOC] Same as the L1 test but selection uses a supplied importance
        # scores tensor instead of the weight magnitudes; compares weights after two
        # successive prunings.
        r"""Test that l1 unstructured pruning actually removes the lowest
        entries of importance scores and not the parameter by l1 norm (by hand).
        It also checks that applying l1 unstructured pruning more than once
        respects the previous mask.
        """
        m = nn.Linear(4, 2)
        # modify its weight matrix by hand
        m.weight = torch.nn.Parameter(
            torch.tensor([[1, 2, 3, 4], [-4, -3, -2, -1]], dtype=torch.float32)
        )
        importance_scores = torch.tensor(
            [[4, 2, 1, 3], [-3, -1, -2, -4]], dtype=torch.float32
        )

        prune.l1_unstructured(
            m, "weight", amount=2, importance_scores=importance_scores
        )
        expected_weight = torch.tensor(
            [[1, 2, 0, 4], [-4, 0, -2, -1]], dtype=m.weight.dtype
        )
        self.assertEqual(expected_weight, m.weight)

        # check that pruning again removes two entries of m.weight that are colocated with
        # the next two smallest absolute values of importance scores.
        prune.l1_unstructured(
            m, "weight", amount=2, importance_scores=importance_scores
        )
        expected_weight = torch.tensor(
            [[1, 0, 0, 4], [-4, 0, 0, -1]], dtype=m.weight.dtype
        )
        self.assertEqual(expected_weight, m.weight)

    def test_unstructured_pruning_same_magnitude(self):
        # [FUNCTION DOC] With a tensor containing many equal magnitudes, checks L1 unstructured
        # pruning removes exactly the requested count of entries (bottom fraction) rather than
        # everything at the threshold value.
        r"""Since it may happen that the tensor to prune has entries with the
        same exact magnitude, it is important to check that pruning happens
        consistently based on the bottom % of weights, and not by threshold,
        which would instead kill off *all* units with magnitude = threshold.
        """
        AMOUNT = 0.2
        p = prune.L1Unstructured(amount=AMOUNT)
        # create a random tensors with entries in {-2, 0, 2}
        t = 2 * torch.randint(low=-1, high=2, size=(10, 7))
        nparams_toprune = prune._compute_nparams_toprune(AMOUNT, t.nelement())

        computed_mask = p.compute_mask(t, default_mask=torch.ones_like(t))
        nparams_pruned = torch.sum(computed_mask == 0)
        self.assertEqual(nparams_toprune, nparams_pruned)

    def test_random_structured_pruning_amount(self):
        # [FUNCTION DOC] Checks random structured pruning of a given amount along the last axis
        # of a three-dimensional tensor zeroes exactly one whole slice of twenty entries and
        # leaves the rest intact; raises AssertionError otherwise.
        AMOUNT = 0.6
        AXIS = 2
        p = prune.RandomStructured(amount=AMOUNT, dim=AXIS)
        t = 2 * torch.randint(low=-1, high=2, size=(5, 4, 2)).to(dtype=torch.float32)

        computed_mask = p.compute_mask(t, default_mask=torch.ones_like(t))
        # check that 1 column is fully prune, the others are left untouched
        remaining_axes = [_ for _ in range(len(t.shape)) if _ != AXIS]
        per_column_sums = sorted(torch.sum(computed_mask == 0, axis=remaining_axes))
        if per_column_sums != [0, 20]:
            raise AssertionError(
                f"Expected per_column_sums == [0, 20], got {per_column_sums}"
            )

    def test_ln_structured_pruning(self):
        # [FUNCTION DOC] Checks Ln structured pruning by hand on a convolution weight: first
        # removes one of three channels by L2 norm, then one of two columns along the last axis
        # by L1 norm, comparing against expected masks.
        r"""Check Ln structured pruning by hand."""
        m = nn.Conv2d(3, 1, 2)
        m.weight.data = torch.tensor(
            [
                [
                    [[1.0, 2.0], [1.0, 2.5]],
                    [[0.5, 1.0], [0.1, 0.1]],
                    [[-3.0, -5.0], [0.1, -1.0]],
                ]
            ]
        )
        # expected effect of pruning 1 of the 3 channels by L2-norm
        expected_mask_axis1 = torch.ones_like(m.weight)
        expected_mask_axis1[:, 1] = 0.0

        prune.ln_structured(m, "weight", amount=1, n=2, dim=1)
        self.assertEqual(expected_mask_axis1, m.weight_mask)

        # expected effect of pruning 1 of the 2 columns along axis -1 by L1-norm
        expected_mask_axis3 = expected_mask_axis1
        expected_mask_axis3[:, :, :, 0] = 0.0

        prune.ln_structured(m, "weight", amount=1, n=1, dim=-1)
        self.assertEqual(expected_mask_axis3, m.weight_mask)

    def test_ln_structured_pruning_importance_scores(self):
        # [FUNCTION DOC] Same as the Ln structured test but the channel and column to remove are
        # chosen by a supplied importance scores tensor instead of the weight values.
        r"""Check Ln structured pruning by hand."""
        m = nn.Conv2d(3, 1, 2)
        m.weight.data = torch.tensor(
            [
                [
                    [[1.0, 2.0], [1.0, 2.5]],
                    [[0.5, 1.0], [0.1, 0.1]],
                    [[-3.0, -5.0], [0.1, -1.0]],
                ]
            ]
        )
        importance_scores = torch.tensor(
            [
                [
                    [[10.0, 1.0], [10.0, 1.0]],
                    [[30.0, 3.0], [30.0, 3.0]],
                    [[-20.0, -2.0], [-20.0, -2.0]],
                ]
            ]
        )
        # expected effect of pruning 1 of the 3 channels by L2-norm
        expected_mask_axis1 = torch.ones_like(m.weight)
        expected_mask_axis1[:, 0] = 0.0

        prune.ln_structured(
            m, "weight", amount=1, n=2, dim=1, importance_scores=importance_scores
        )
        self.assertEqual(expected_mask_axis1, m.weight_mask)

        # expected effect of pruning 1 of the 2 columns along axis -1 by L1-norm
        expected_mask_axis3 = expected_mask_axis1
        expected_mask_axis3[:, :, :, 1] = 0.0

        prune.ln_structured(
            m, "weight", amount=1, n=1, dim=-1, importance_scores=importance_scores
        )
        self.assertEqual(expected_mask_axis3, m.weight_mask)

    def test_remove_pruning(self):
        # [FUNCTION DOC] For linear and 3D convolution modules, after pruning the module has the
        # original parameter and mask buffer and not a plain parameter; after removing pruning
        # the plain parameter returns, the extra ones are gone, and the final tensor equals the
        # pruned one.
        r"""`prune.remove` removes the hook and the reparametrization
        and makes the pruning final in the original parameter.
        """
        modules = [nn.Linear(5, 7), nn.Conv3d(2, 2, 2)]
        names = ["weight", "bias"]

        for m in modules:
            for name in names:
                with self.subTest(m=m, name=name):
                    # first prune
                    prune.random_unstructured(m, name, amount=0.5)
                    self.assertIn(name + "_orig", dict(m.named_parameters()))
                    self.assertIn(name + "_mask", dict(m.named_buffers()))
                    self.assertNotIn(name, dict(m.named_parameters()))
                    self.assertTrue(hasattr(m, name))
                    pruned_t = getattr(m, name)

                    # then remove pruning
                    prune.remove(m, name)
                    self.assertIn(name, dict(m.named_parameters()))
                    self.assertNotIn(name + "_orig", dict(m.named_parameters()))
                    self.assertNotIn(name + "_mask", dict(m.named_buffers()))
                    final_t = getattr(m, name)

                    self.assertEqual(pruned_t, final_t)

    def test_remove_pruning_exception(self):
        # [FUNCTION DOC] Checks that removing pruning from modules never pruned raises
        # ValueError, and that the pruned-state query reports false first.
        # KNOWN ISSUE: the docstring says an assertion error is thrown but the test
        # expects ValueError.
        r"""Removing from an unpruned tensor throws an assertion error"""
        modules = [nn.Linear(5, 7), nn.Conv3d(2, 2, 2)]
        names = ["weight", "bias"]

        for m in modules:
            for name in names:
                with self.subTest(m=m, name=name):
                    # check that the module isn't pruned
                    self.assertFalse(prune.is_pruned(m))
                    # since it isn't pruned, pruning can't be removed from it
                    with self.assertRaises(ValueError):
                        prune.remove(m, name)

    def test_global_pruning(self):
        # [FUNCTION DOC] Checks global L1 unstructured pruning across two linear weights removes
        # the four smallest magnitude weights globally, comparing both resulting weights with
        # expected values.
        r"""Test that global l1 unstructured pruning over 2 parameters removes
        the `amount=4` smallest global weights across the 2 parameters.
        """
        m = nn.Linear(4, 2)
        n = nn.Linear(3, 1)
        # modify the weight matrices by hand
        m.weight = torch.nn.Parameter(
            torch.tensor([[1, 2, 3, 4], [-4, -3, -2, -1]]).to(dtype=torch.float32)
        )
        n.weight = torch.nn.Parameter(
            torch.tensor([[0, 0.1, -2]]).to(dtype=torch.float32)
        )

        params_to_prune = (
            (m, "weight"),
            (n, "weight"),
        )

        # prune the 4 smallest weights globally by L1 magnitude
        prune.global_unstructured(
            params_to_prune, pruning_method=prune.L1Unstructured, amount=4
        )

        expected_mweight = torch.tensor(
            [[0, 2, 3, 4], [-4, -3, -2, 0]], dtype=m.weight.dtype
        )
        self.assertEqual(expected_mweight, m.weight)

        expected_nweight = torch.tensor([[0, 0, -2]]).to(dtype=n.weight.dtype)
        self.assertEqual(expected_nweight, n.weight)

    def test_global_pruning_importance_scores(self):
        # [FUNCTION DOC] Same as the global pruning test but selection uses a mapping from each
        # parameter pair to a supplied importance scores tensor; compares both resulting weights.
        r"""Test that global l1 unstructured pruning over 2 parameters removes
        the `amount=4` smallest global weights across the 2 parameters.
        """
        m = nn.Linear(4, 2)
        n = nn.Linear(3, 1)
        # modify the weight matrices by hand
        m.weight = torch.nn.Parameter(
            torch.tensor([[1, 2, 3, 4], [-4, -3, -2, -1]]).to(dtype=torch.float32)
        )
        m_importance_scores = torch.tensor(
            [[4, 2, 1, 3], [-3, -1, -2, -4]], dtype=torch.float32
        )
        n.weight = torch.nn.Parameter(
            torch.tensor([[0, 0.1, -2]]).to(dtype=torch.float32)
        )
        n_importance_scores = torch.tensor([[0, 10.0, -0.2]]).to(dtype=torch.float32)

        params_to_prune = (
            (m, "weight"),
            (n, "weight"),
        )
        importance_scores = {
            (m, "weight"): m_importance_scores,
            (n, "weight"): n_importance_scores,
        }

        # prune the 4 smallest weights globally by L1 magnitude
        prune.global_unstructured(
            params_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=4,
            importance_scores=importance_scores,
        )

        expected_m_weight = torch.tensor(
            [[1, 2, 0, 4], [-4, 0, -2, -1]], dtype=m.weight.dtype
        )
        self.assertEqual(expected_m_weight, m.weight)

        expected_n_weight = torch.tensor([[0, 0.1, 0]]).to(dtype=n.weight.dtype)
        self.assertEqual(expected_n_weight, n.weight)

    def test_custom_from_mask_pruning(self):
        # [FUNCTION DOC] Checks the custom-from-mask method combines a given mask with the
        # previous default mask (an elementwise product) when computing the new mask.
        r"""Test that the CustomFromMask is capable of receiving
        as input at instantiation time a custom mask, and combining it with
        the previous default mask to generate the correct final mask.
        """
        # new mask
        mask = torch.tensor([[0, 1, 1, 0], [0, 0, 1, 1]])
        # old mask
        default_mask = torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1]])

        # some tensor (not actually used)
        t = torch.rand_like(mask.to(dtype=torch.float32))

        p = prune.CustomFromMask(mask=mask)

        computed_mask = p.compute_mask(t, default_mask)
        expected_mask = torch.tensor(
            [[0, 0, 0, 0], [0, 0, 1, 1]], dtype=computed_mask.dtype
        )

        self.assertEqual(computed_mask, expected_mask)

    def test_pruning_rollback(self):
        # [FUNCTION DOC] With the L1 mask computation patched to raise an exception, pruning must
        # raise, and the module must be left unchanged: parameter still present, no mask buffer
        # and no original-parameter entry.
        r"""Test that if something fails when the we try to compute the mask,
        then the model isn't left in some intermediate half-pruned state.
        The try/except statement in `apply` should handle rolling back
        to the previous state before pruning began.
        """
        modules = [nn.Linear(5, 7), nn.Conv3d(2, 2, 2)]
        names = ["weight", "bias"]

        for m in modules:
            for name in names:
                with self.subTest(m=m, name=name):
                    with mock.patch(
                        "torch.nn.utils.prune.L1Unstructured.compute_mask"
                    ) as compute_mask:
                        compute_mask.side_effect = Exception("HA!")
                        with self.assertRaises(Exception):
                            prune.l1_unstructured(m, name=name, amount=0.9)

                        self.assertTrue(name in dict(m.named_parameters()))
                        self.assertFalse(name + "_mask" in dict(m.named_buffers()))
                        self.assertFalse(name + "_orig" in dict(m.named_parameters()))

    def test_pruning_serialization_model(self):
        # [FUNCTION DOC] Prunes one layer of a sequential model, checks state dict keys change to
        # the original weight and mask, saves the whole model to a temporary file and loads it
        # back (full unpickling), then checks keys and pruned weight values survive.
        # create a model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1),
        )
        # check that everything looks normal before pruning
        self.assertNotIn("0.weight_orig", model.state_dict())
        self.assertNotIn("0.weight_mask", model.state_dict())
        self.assertIn("0.weight", model.state_dict())

        # prune one of its parameters
        prune.l1_unstructured(module=model[0], name="weight", amount=0.9)

        # check that the original weight and the new mask are present
        self.assertIn("0.weight_orig", model.state_dict())
        self.assertIn("0.weight_mask", model.state_dict())
        self.assertNotIn("0.weight", model.state_dict())
        self.assertTrue(hasattr(model[0], "weight"))

        pruned_weight = model[0].weight

        with TemporaryFileName() as fname:
            torch.save(model, fname)
            # weights_only=False as this is legacy code that saves the model
            new_model = torch.load(fname, weights_only=False)

        # check that the original weight and the new mask are present
        self.assertIn("0.weight_orig", new_model.state_dict())
        self.assertIn("0.weight_mask", new_model.state_dict())
        self.assertNotIn("0.weight", new_model.state_dict())
        self.assertTrue(hasattr(new_model[0], "weight"))

        self.assertEqual(pruned_weight, new_model[0].weight)

    def test_pruning_serialization_state_dict(self):
        # [FUNCTION DOC] Prunes then removes pruning on one layer, saves only the state dict to a
        # temporary file, loads it into a fresh model, and checks no original/mask keys exist and
        # the weight equals the pruned weight.
        # create a model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1),
        )
        # check that everything looks normal before pruning
        self.assertNotIn("0.weight_orig", model.state_dict())
        self.assertNotIn("0.weight_mask", model.state_dict())
        self.assertIn("0.weight", model.state_dict())

        # prune one of its parameters
        prune.l1_unstructured(module=model[0], name="weight", amount=0.9)

        # check that the original weight and the new mask are present
        self.assertIn("0.weight_orig", model.state_dict())
        self.assertIn("0.weight_mask", model.state_dict())
        self.assertNotIn("0.weight", model.state_dict())
        self.assertTrue(hasattr(model[0], "weight"))

        pruned_weight = model[0].weight

        # make pruning permanent and restore parameter names as in base
        # architecture
        prune.remove(module=model[0], name="weight")

        # check that the original weight and the new mask are no longer present
        self.assertNotIn("0.weight_orig", model.state_dict())
        self.assertNotIn("0.weight_mask", model.state_dict())
        self.assertIn("0.weight", model.state_dict())

        # save the state dict of model and reload it into new_model
        new_model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1),
        )
        with TemporaryFileName() as fname:
            torch.save(model.state_dict(), fname)
            new_model.load_state_dict(torch.load(fname))

        # check that the original weight and the new mask are not present in
        # new_model either.
        self.assertNotIn("0.weight_orig", new_model.state_dict())
        self.assertNotIn("0.weight_mask", new_model.state_dict())
        self.assertIn("0.weight", new_model.state_dict())

        self.assertEqual(pruned_weight, new_model[0].weight)

    def test_prune(self):
        # [FUNCTION DOC] Checks the prune method of an L1 method with a prior mask returns the
        # tensor multiplied by the expected mask (two lowest magnitude units zeroed among
        # unmasked ones).
        # create a new pruning method
        p = prune.L1Unstructured(amount=2)
        # create tensor to be pruned
        t = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).to(dtype=torch.float32)
        # create prior mask by hand
        default_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 1]])
        # since we are pruning the two lowest magnitude units, the outcome of
        # the calculation should be this:
        expected_mask = torch.tensor([[0, 0, 1, 0], [1, 1, 0, 1]])
        pruned_tensor = p.prune(t, default_mask)
        self.assertEqual(t * expected_mask, pruned_tensor)

    def test_prune_importance_scores(self):
        # [FUNCTION DOC] Same as the prune method test but unit selection follows a supplied
        # importance scores tensor rather than the tensor values.
        # create a new pruning method
        p = prune.L1Unstructured(amount=2)
        # create tensor to be pruned
        t = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).to(dtype=torch.float32)
        importance_scores = torch.tensor([[1, 2, 3, 4], [1.5, 1.6, 1.7, 1.8]]).to(
            dtype=torch.float32
        )
        # create prior mask by hand
        default_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 1]])
        # since we are pruning the two lowest magnitude units, the outcome of
        # the calculation should be this:
        expected_mask = torch.tensor([[0, 1, 1, 0], [0, 1, 0, 1]])
        pruned_tensor = p.prune(t, default_mask, importance_scores=importance_scores)
        self.assertEqual(t * expected_mask, pruned_tensor)

    def test_prune_importance_scores_mimic_default(self):
        # [FUNCTION DOC] Checks that passing the tensor itself as importance scores gives the
        # same pruned tensor as omitting scores, and that it matches the expected masked result.
        # create a new pruning method
        p = prune.L1Unstructured(amount=2)
        # create tensor to be pruned
        t = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).to(dtype=torch.float32)
        # create prior mask by hand
        default_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 1]])
        # since we are pruning the two lowest magnitude units, the outcome of
        # the calculation should be this:
        expected_mask = torch.tensor([[0, 0, 1, 0], [1, 1, 0, 1]])
        pruned_tensor_without_importance_scores = p.prune(t, default_mask)
        pruned_tensor_with_importance_scores = p.prune(
            t, default_mask, importance_scores=t
        )
        self.assertEqual(
            pruned_tensor_without_importance_scores,
            pruned_tensor_with_importance_scores,
        )
        self.assertEqual(t * expected_mask, pruned_tensor_without_importance_scores)

    def test_rnn_pruning(self):
        # [FUNCTION DOC] Prunes one weight of an LSTM and checks only three of the four flat
        # weights are parameters, removes the pruning and checks four parameters again with the
        # original name restored in the parameter dictionary and the reparametrization part gone;
        # raises AssertionError on any mismatch.
        l = torch.nn.LSTM(32, 32)
        # This Module has 4 parameters called:
        # 'weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0'

        # Pruning one of them causes one of the weights to become a tensor
        prune.l1_unstructured(l, "weight_ih_l0", 0.5)
        param_count = sum(isinstance(p, torch.nn.Parameter) for p in l._flat_weights)
        if param_count != 3:
            raise AssertionError(
                f"Expected 3 Parameters in _flat_weights after pruning, got {param_count}"
            )

        # Removing the pruning reparameterization restores the Parameter
        prune.remove(l, "weight_ih_l0")
        param_count = sum(isinstance(p, torch.nn.Parameter) for p in l._flat_weights)
        if param_count != 4:
            raise AssertionError(
                f"Expected 4 Parameters in _flat_weights after removal, got {param_count}"
            )

        # Make sure that, upon removal of the reparameterization, the
        # `._parameters` and `.named_parameters` contain the right params.
        # Specifically, the original weight ('weight_ih_l0') should be placed
        # back in the parameters, while the reparameterization component
        # ('weight_ih_l0_orig') should be removed.
        if "weight_ih_l0" not in l._parameters:
            raise AssertionError("'weight_ih_l0' should be in l._parameters")
        if l._parameters["weight_ih_l0"] is None:
            raise AssertionError("l._parameters['weight_ih_l0'] should not be None")
        if "weight_ih_l0_orig" in l._parameters:
            raise AssertionError("'weight_ih_l0_orig' should not be in l._parameters")
        if "weight_ih_l0" not in dict(l.named_parameters()):
            raise AssertionError("'weight_ih_l0' should be in l.named_parameters()")
        if dict(l.named_parameters())["weight_ih_l0"] is None:
            raise AssertionError(
                "l.named_parameters()['weight_ih_l0'] should not be None"
            )
        if "weight_ih_l0_orig" in dict(l.named_parameters()):
            raise AssertionError(
                "'weight_ih_l0_orig' should not be in l.named_parameters()"
            )


instantiate_parametrized_tests(TestPruningNN)

if __name__ == "__main__":
    run_tests()
