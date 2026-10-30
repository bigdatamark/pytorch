# Owner(s): ["module: unknown"]
# [MODULE DOC] Tests for the per-overload operator API: OpOverloadPacket and OpOverload
# attributes, caching, deepcopy, repr, calling, out variants and decompose.
import copy

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestPerOverloadAPI(TestCase):
    # [CLASS DOC] Tests basic behavior of operator packets and overloads obtained via the
    # ops namespace.
    def test_basics_opoverloadpacket(self):
        # [FUNCTION DOC] Checks the add packet: name, string form, callable result,
        # module, identity caching, deepcopy returning the same object, repr, and
        # AttributeError on unknown attributes.
        # add is only used as an example here. It is ok to update the test
        # if the semantics of add are modified in the future.
        add_packet = torch.ops.aten.add

        # class attributes
        self.assertEqual(add_packet.__name__, "add")
        self.assertEqual(str(add_packet), "aten.add")

        # callable
        self.assertEqual(add_packet(torch.tensor(2), torch.tensor(3)), torch.tensor(5))

        # correct module
        self.assertEqual(add_packet.__module__, add_packet.op.__module__)

        # caching
        another_add_packet = torch.ops.aten.add
        self.assertEqual(id(add_packet), id(another_add_packet))

        # deepcopy is a no-op
        self.assertEqual(id(add_packet), id(copy.deepcopy(add_packet)))

        # pretty print
        self.assertEqual(repr(add_packet), "<OpOverloadPacket(op='aten.add')>")

        self.assertRaises(AttributeError, lambda: add_packet.foo)

    def test_basics_opoverload(self):
        # [FUNCTION DOC] Checks the add Tensor overload: names, parent packet, deepcopy
        # identity, caching, repr, call result, the out overload writing into a tensor,
        # and that passing out to the non-out overload raises a RuntimeError.
        add_packet = torch.ops.aten.add
        add_tensoroverload = add_packet.Tensor

        # class attributes
        self.assertEqual(str(add_tensoroverload), "aten.add.Tensor")
        self.assertEqual(add_tensoroverload.__name__, "add.Tensor")
        self.assertEqual(add_tensoroverload.overloadpacket, add_packet)

        # deepcopy is a no-op
        self.assertEqual(id(add_tensoroverload), id(copy.deepcopy(add_tensoroverload)))

        # caching
        another_add_tensoroverload = torch.ops.aten.add.Tensor
        self.assertEqual(id(add_tensoroverload), id(another_add_tensoroverload))

        # pretty print
        self.assertEqual(
            repr(add_tensoroverload), "<OpOverload(op='aten.add', overload='Tensor')>"
        )

        # callable
        self.assertEqual(
            add_tensoroverload(torch.tensor(2), torch.tensor(3)), torch.tensor(5)
        )

        a = torch.tensor(2)
        b = torch.tensor(0)
        torch.ops.aten.add.out(a, a, out=b)
        self.assertEqual(b, torch.tensor(4))

        self.assertRaises(RuntimeError, lambda: add_tensoroverload(a, a, out=b))

    def test_decompose(self):
        # [FUNCTION DOC] Checks that decompose on the linear overload matches calling it
        # directly.
        x = torch.randn(2, 3)
        y = torch.randn(5, 3)
        self.assertEqual(
            torch.ops.aten.linear.default.decompose(x, y),
            torch.ops.aten.linear.default(x, y),
        )


if __name__ == "__main__":
    run_tests()
