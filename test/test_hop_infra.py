# Owner(s): ["module: higher order operators"]
# [MODULE DOC] Infrastructure checks for higher order operators (HOPs). Verifies every registered
# HOP has an OpInfo in hop_db (or is allowlisted), hop_db entries carry no decorators or skips,
# HOPs are exported from the higher order ops package, and HOPs deepcopy to themselves. Also covers
# the triton kernel wrapper HOPs under a fake tensor mode. Import-time side effect: the module
# walks and imports every submodule of the higher order ops package so that all HOPs are
# registered.
import copy
import importlib
import pkgutil

import torch
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase
from torch.testing._internal.hop_db import (
    FIXME_hop_that_doesnt_have_opinfo_test_allowlist,
    hop_db,
)


def do_imports():
    # [FUNCTION DOC] HELPER: Walks the submodules of the higher order ops package (using its path
    # and the dotted package prefix) and imports each by name, so every HOP gets registered in the
    # global registry. Returns nothing; called once at import time of this test module.
    for mod in pkgutil.walk_packages(
        torch._higher_order_ops.__path__, "torch._higher_order_ops."
    ):
        modname = mod.name
        importlib.import_module(modname)


do_imports()


@skipIfTorchDynamo("not applicable")
class TestHOPInfra(TestCase):
    # [CLASS DOC] Test case (skipped under TorchDynamo) holding the HOP registry and export checks.
    # Stateless; tests read the global HOP registry and hop_db.
    def test_all_hops_have_opinfo(self):
        # [FUNCTION DOC] Collects the names of all registered HOPs and fails if any is neither the
        # name of a hop_db OpInfo nor in the allowlist of known missing entries. The failure
        # message lists the missing names and points to the hop_db file.
        """All HOPs should have an OpInfo in torch/testing/_internal/hop_db.py"""
        from torch._ops import _higher_order_ops

        hops_that_have_op_info = {k.name for k in hop_db}
        all_hops = _higher_order_ops.keys()

        missing_ops = set()

        for op in all_hops:
            if (
                op not in hops_that_have_op_info
                and op not in FIXME_hop_that_doesnt_have_opinfo_test_allowlist
            ):
                missing_ops.add(op)

        self.assertTrue(
            len(missing_ops) == 0,
            lambda msg: f"{msg}\nMissing hop_db OpInfo entries for {missing_ops}, please add them to torch/testing/_internal/hop_db.py",
        )

    def test_hop_db_has_no_decorators_or_skips(self):
        # [FUNCTION DOC] Builds the list of full names of hop_db OpInfos that have a non-empty
        # decorators or skips collection and asserts it is empty, so skips live in the tests that
        # need them.
        """hop_db entries should not decide how tests are skipped or decorated."""
        offending_op_infos = [
            op.full_name for op in hop_db if op.decorators or op.skips
        ]

        self.assertEqual(
            offending_op_infos,
            [],
            msg=(
                "Move hop_db decorators/skips to the tests that need them "
                "instead of storing them on the OpInfo."
            ),
        )

    def test_all_hops_are_imported(self):
        # [FUNCTION DOC] Compares the registered HOP names against the package export list, after
        # removing a hard-coded allowlist of HOPs that are known not to be exported. Asserts that
        # the remaining set is empty, so new HOPs must be exported from the package.
        """All HOPs should be listed in torch._higher_order_ops.__all__

        Some constraints (see test_testing.py::TestImports)
        - Sympy must be lazily imported
        - Dynamo must be lazily imported
        """
        imported_hops = torch._higher_order_ops.__all__
        registered_hops = torch._ops._higher_order_ops.keys()

        # Please don't add anything here.
        # We want to ensure that all HOPs are imported at "import torch" time.
        # It is bad if someone tries to access torch.ops.higher_order.cond
        # and it doesn't exist (this may happen if your HOP isn't imported at
        # "import torch" time).
        FIXME_ALLOWLIST = {
            "autograd_function_apply",
            "run_with_rng_state",
            "run_dtensor_rng_op",
            "graphsafe_run_with_rng_state",
            "map_impl",
            "_export_tracepoint",
            "run_and_save_rng_state",
            "map",
            "custom_function_call",
            "trace_wrapped",
            "triton_kernel_wrapper_functional",
            "triton_kernel_wrapper_mutation",
            "wrap",  # Really weird failure -- importing this causes Dynamo to choke on checkpoint
            "control_deps",  # Inductor-internal ordering HOP, not a public torch.ops.higher_order API
        }
        not_imported_hops = registered_hops - imported_hops
        not_imported_hops = not_imported_hops - FIXME_ALLOWLIST
        self.assertEqual(
            not_imported_hops,
            set(),
            msg="All HOPs must be listed under torch/_higher_order_ops/__init__.py's __all__.",
        )

    def test_imports_from_all_work(self):
        # [FUNCTION DOC] Iterates over each name in the higher order ops package export list and
        # fetches the attribute from the package, failing with an error if any exported name cannot
        # be resolved.
        """All APIs listed in torch._higher_order_ops.__all__ must be importable"""
        stuff = torch._higher_order_ops.__all__
        for attr in stuff:
            getattr(torch._higher_order_ops, attr)

    def test_hop_deepcopy_preserves_identity(self):
        # [FUNCTION DOC] For every registered HOP, deep-copies it and asserts the copy is the very
        # same object (identity check and equal object ids), verifying HOPs behave as singletons.
        """HigherOrderOperators should be singletons - deepcopy should return the same object"""
        from torch._ops import _higher_order_ops

        for name, hop in _higher_order_ops.items():
            copied_hop = copy.deepcopy(hop)
            self.assertIs(
                hop,
                copied_hop,
                lambda msg: f"{msg}\ndeepcopy of HigherOrderOperator '{name}' should return the same object (singleton pattern)",
            )
            self.assertEqual(id(hop), id(copied_hop))

    def test_triton_kernel_wrapper_functional_fake_tensor_nested_kwargs(self):
        # [FUNCTION DOC] Creates a fake tensor under a FakeTensorMode with a strided layout, then
        # calls the functional triton kernel wrapper with that fake tensor inside the kwargs and
        # one tensor to clone. Asserts the result mapping has only that key, holds a fake tensor
        # from the same mode, and keeps the input size and stride.
        from torch._higher_order_ops.triton_kernel_wrap import (
            triton_kernel_wrapper_functional,
        )
        from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

        mode = FakeTensorMode()
        with mode:
            fake_output = torch.empty_strided((2, 3), (3, 1))

        self.assertIsInstance(fake_output, FakeTensor)

        out = triton_kernel_wrapper_functional(
            kernel_idx=0,
            constant_args_idx=0,
            grid=[(1,)],
            tma_descriptor_metadata={},
            kwargs={"out_ptr": fake_output},
            tensors_to_clone=["out_ptr"],
        )

        self.assertEqual(set(out.keys()), {"out_ptr"})
        self.assertIsInstance(out["out_ptr"], FakeTensor)
        self.assertIs(out["out_ptr"].fake_mode, mode)
        self.assertEqual(out["out_ptr"].size(), fake_output.size())
        self.assertEqual(out["out_ptr"].stride(), fake_output.stride())

    def test_triton_kernel_wrapper_mutation_fake_tensor_nested_kwargs(self):
        # [FUNCTION DOC] Creates a fake tensor under a FakeTensorMode, calls the mutation triton
        # kernel wrapper with it in the kwargs, and asserts the call returns None (no real kernel
        # is launched on fake inputs).
        from torch._higher_order_ops.triton_kernel_wrap import (
            triton_kernel_wrapper_mutation,
        )
        from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

        mode = FakeTensorMode()
        with mode:
            fake_output = torch.empty_strided((2, 3), (3, 1))

        self.assertIsInstance(fake_output, FakeTensor)

        out = triton_kernel_wrapper_mutation(
            kernel_idx=0,
            constant_args_idx=0,
            grid=[(1,)],
            tma_descriptor_metadata={},
            kwargs={"out_ptr": fake_output},
        )

        self.assertIsNone(out)


if __name__ == "__main__":
    run_tests()
