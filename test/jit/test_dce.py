# Owner(s): ["oncall: jit"]

# [MODULE DOC] JIT tests for dead code elimination interacting with attribute writes (SetAttr
# nodes) and tensor mutation: checks that mutating or attribute-setting nodes are kept or
# removed as appropriate. Running the file directly is refused in favor of test/test_jit.py.
import torch
from torch.testing import FileCheck
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase, make_global


class TestDCE(JitTestCase):
    # [CLASS DOC] Test case for the JIT dead-code-elimination pass on graphs containing SetAttr
    # and in-place mutation.
    def test_setattr_no_aliasdb(self):
        # [FUNCTION DOC] Scripts a module whose forward assigns a freshly created tensor to an
        # attribute without returning anything, and checks with FileCheck that the SetAttr node
        # is still present in the graph (the write must not be eliminated).
        class Net(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a tensor attribute x that forward overwrites with a
            # new random tensor.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes the base module and sets attribute x to an
                # empty 2x2 tensor.
                super().__init__()
                self.x = torch.empty([2, 2])

            def forward(self):
                # [FUNCTION DOC] HELPER: Creates a random 3x3 tensor and stores it in attribute x;
                # returns nothing.
                x = torch.rand([3, 3])
                self.x = x

        net = torch.jit.script(Net())

        FileCheck().check("prim::SetAttr").run(net.graph)

    def test_setattr_removed(self):
        # [FUNCTION DOC] Checks that writes to a script class instance which never escapes are
        # removed. Scripts a class Thing1 and a module that sets Thing1 attribute x but returns
        # another tensor, scripts it, sets eval mode, freezes it (which inlines the class init
        # so DCE can run), then asserts via FileCheck that no SetAttr node remains.
        @torch.jit.script
        class Thing1:
            # [CLASS DOC] HELPER: Torchscript class (made globally visible) with one attribute x
            # initialized to a 2x2 zeros tensor.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes attribute x to a 2x2 zeros tensor.
                self.x = torch.zeros([2, 2])

        make_global(Thing1)

        class Thing2(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward builds a Thing1, sets its attribute to a
            # random tensor, and returns a different random tensor, so the attribute write is
            # dead.
            def forward(self):
                # [FUNCTION DOC] HELPER: Creates two random 2x2 tensors, writes the first into a
                # new Thing1 instance's attribute x, and returns the second.
                x = torch.rand([2, 2])
                y = torch.rand([2, 2])
                t1 = Thing1()
                t1.x = x
                return y

        unscripted = Thing2()

        t2 = torch.jit.script(unscripted)
        t2.eval()

        # freezing inlines t1.__init__(), after which DCE can occur.
        t2 = torch.jit.freeze(t2)
        FileCheck().check_not("prim::SetAttr").run(t2.graph)

    def test_mutated_simple(self):
        # [FUNCTION DOC] Scripts a function that applies sin, mutates a strided slice in place
        # with add_, then returns cos of the base tensor. Runs the graph-level DCE pass and
        # checks via FileCheck that the aten::add_ node survives because it mutates a tensor
        # aliasing a live value.
        def fn(x: torch.Tensor):
            # [FUNCTION DOC] HELPER: Takes sin of x, adds a strided slice of x in place into a
            # strided view of the result, and returns the cosine of the (mutated) result.
            y = x.sin()
            y_slice = y[::2]
            y_slice.add_(x[::2])
            z = y.cos()
            return z

        fn_s = torch.jit.script(fn)
        torch._C._jit_pass_dce_graph(fn_s.graph)

        FileCheck().check("aten::add_").run(fn_s.graph)

    def test_mutated_loop(self):
        # [FUNCTION DOC] Like the simple mutation test but with a loop: scripts a function
        # that mutates a slice in place, then loops twice reassigning the slice view and
        # repeating the tensor. Runs the graph DCE pass and checks via FileCheck that
        # aten::add_ is still present.
        def fn(x: torch.Tensor):
            # [FUNCTION DOC] HELPER: Takes sin of x, mutates a strided view of it in place,
            # loops twice (taking a new strided view and repeating the tensor along its
            # dimension), and returns the cosine of the result.
            y = x.sin()
            y_slice = y[::2]
            y_slice.add_(x[::2])
            for _ in range(2):
                y_slice = y[::2]
                y = y.repeat(2)
            z = y.cos()
            return z

        fn_s = torch.jit.script(fn)
        torch._C._jit_pass_dce_graph(fn_s.graph)

        FileCheck().check("aten::add_").run(fn_s.graph)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
