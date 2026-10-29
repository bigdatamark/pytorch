# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Unit tests for torch.distributed.tensor OpSchema: equality/hash semantics
# with static args and kwargs, and absence of a DTensorSpec reference leak.
# No import-time side effects; runs via run_tests when executed as a script.

import random

from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import OpSchema, RuntimeSchemaInfo
from torch.testing._internal.common_utils import run_tests, TestCase


class TestOpSchema(TestCase):
    # [CLASS DOC] Stateless unit tests for OpSchema equality, hashing and reference
    # handling (no process group or devices needed).
    def test_equality_checks_lists_of_dtensor_spec(self):
        # [FUNCTION DOC] Regression test: two OpSchemas whose second arg is a list of one vs
        # two DTensorSpecs must compare unequal and have different hashes.
        """If x == y, then we must have h(x) == h(y)."""
        dts = DTensorSpec(mesh=None, placements=tuple(), tensor_meta=None)
        schema1 = OpSchema(op=None, args_schema=(dts, [dts]), kwargs_schema={})
        schema2 = OpSchema(op=None, args_schema=(dts, [dts, dts]), kwargs_schema={})
        # This is a regression test; these schemas used to compare equal.
        self.assertNotEqual(schema1, schema2)
        self.assertNotEqual(hash(schema1), hash(schema2))

    def test_equality_respects_static_attributes(self):
        # [FUNCTION DOC] Checks OpSchema equality over all pairs of the four static-info
        # variants, with matching values and with mismatching static arg and/or kwarg values.
        # Schemas are equal only if their static annotations agree and, for annotated
        # positions, the values match; unannotated args/kwargs (random values) never matter.
        def _get_sample_op_schemas(static_arg_val, static_kwarg_val):
            # [FUNCTION DOC] HELPER: builds four OpSchemas (no schema_info, static arg only,
            # static kwarg only, both) sharing one DTensorSpec, a random ignored positional
            # arg, the given value at argument index 2 and the given value for keyword
            # "statickwarg" (plus a random ignored kwarg). Returns a list of tuples of
            # (has_static_arg, has_static_kwarg, OpSchema).
            dts = DTensorSpec(mesh=None, placements=tuple(), tensor_meta=None)
            static_argnum = 2
            static_kwargkey = ["statickwarg"]
            annotated_schemas = [
                (False, False, None),
                (True, False, RuntimeSchemaInfo(static_argnum=static_argnum)),
                (False, True, RuntimeSchemaInfo(static_kwargkey=static_kwargkey)),
                (
                    True,
                    True,
                    RuntimeSchemaInfo(
                        static_argnum=static_argnum, static_kwargkey=static_kwargkey
                    ),
                ),
            ]

            # non-tensor args show up in hash iff the argnum is static/
            # kwargs show up in hash iff their name is in static_kwargkey.
            # random elements are random because they are not supposed to matter for
            # equality at all.
            args_schema = (dts, random.randint(1, 1000000), static_arg_val)
            kwargs_schema = {
                "ignoredkwarg": random.randint(1, 1000000),
                "statickwarg": static_kwarg_val,
            }
            return [
                (
                    has_static_arg,
                    has_static_kwarg,
                    OpSchema(
                        op=None,
                        args_schema=args_schema,
                        kwargs_schema=kwargs_schema,
                        schema_info=si,
                    ),
                )
                for (has_static_arg, has_static_kwarg, si) in annotated_schemas
            ]

        for lhs_has_static_arg, lhs_has_static_kwarg, lhs in _get_sample_op_schemas(
            1, 2
        ):
            # Static arg/kwarg both match
            for rhs_has_static_arg, rhs_has_static_kwarg, rhs in _get_sample_op_schemas(
                1, 2
            ):
                if (
                    lhs_has_static_arg == rhs_has_static_arg
                    and lhs_has_static_kwarg == rhs_has_static_kwarg
                ):
                    self.assertEqual(lhs, rhs)
                else:
                    self.assertNotEqual(lhs, rhs)

            # Static arg mismatch
            for rhs_has_static_arg, rhs_has_static_kwarg, rhs in _get_sample_op_schemas(
                3, 2
            ):
                if (
                    lhs_has_static_arg
                    or rhs_has_static_arg
                    or lhs_has_static_kwarg != rhs_has_static_kwarg
                ):
                    self.assertNotEqual(lhs, rhs)
                else:
                    self.assertEqual(lhs, rhs)

            # Static kwarg mismatch
            for rhs_has_static_arg, rhs_has_static_kwarg, rhs in _get_sample_op_schemas(
                1, 3
            ):
                if (
                    lhs_has_static_kwarg
                    or rhs_has_static_kwarg
                    or lhs_has_static_arg != rhs_has_static_arg
                ):
                    self.assertNotEqual(lhs, rhs)
                else:
                    self.assertEqual(lhs, rhs)

            # Static arg/kwarg both mismatch
            for rhs_has_static_arg, rhs_has_static_kwarg, rhs in _get_sample_op_schemas(
                3, 4
            ):
                if (
                    lhs_has_static_arg
                    or rhs_has_static_arg
                    or lhs_has_static_kwarg
                    or rhs_has_static_kwarg
                ):
                    self.assertNotEqual(lhs, rhs)
                else:
                    self.assertEqual(lhs, rhs)

    def test_no_dtensor_spec_leak_in_comparison_key(self):
        # [FUNCTION DOC] Regression test: builds an OpSchema holding one DTensorSpec, drops the
        # local reference, and asserts a weak reference to the spec is dead, i.e. computing
        # the schema's comparison key must not leak a reference to the spec.
        """Regression test for https://github.com/pytorch/pytorch/issues/181761.

        DTensor_OpSchema_recompute_comparison_key_impl used to release() the
        inner args_to_hash tuple before passing it to PyTuple_Pack on the
        no-static_kwargkey path, leaking one ref to every DTensorSpec in
        args_schema for the lifetime of the process.
        """
        import weakref

        dts = DTensorSpec(mesh=None, placements=tuple(), tensor_meta=None)
        ref = weakref.ref(dts)
        OpSchema(op=None, args_schema=(dts,), kwargs_schema={})
        del dts
        self.assertIsNone(ref(), "DTensorSpec leaked via OpSchema._comparison_key")


if __name__ == "__main__":
    run_tests()
