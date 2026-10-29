# Owner(s): ["module: fx"]
# [MODULE DOC] Tests that FX tracing with parameter shapes treated as constants can follow
# Python control flow that depends on a parameter's shape, size, dimension count or element
# count. Defines a base module with a branching forward (matrix multiply with or without a
# relu) and subclasses that decide the branch from different parameter queries; a test class
# traces a pair of instances per query and checks branching differences.

import torch
import torch.fx
from torch.testing._internal.common_utils import raise_on_run_directly, TestCase


class MyModuleBase(torch.nn.Module):
    # [CLASS DOC] HELPER: Base module. Forward fetches a matrix via the overridable getter
    # and branches on the overridable no_relu predicate: matrix multiply alone if true,
    # matrix multiply followed by relu otherwise. Subclasses must provide the parameter and
    # predicate.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Multiplies x by the matrix; applies relu to the product
        # unless no_relu reports true.
        matrx = self.get_mul_matrix()
        if self.no_relu():
            return torch.mm(x, matrx)
        else:
            return torch.relu(torch.mm(x, matrx))

    def get_mul_matrix(self):
        # [FUNCTION DOC] HELPER: Returns the module's parameter attribute.
        return self.param

    def no_relu(self):
        # [FUNCTION DOC] HELPER: Placeholder predicate; always raises a generic exception
        # stating it is not implemented. Subclasses override.
        raise Exception("not implemented")  # noqa: TRY002


class MyModuleParamShape(MyModuleBase):
    # [CLASS DOC] HELPER: Module with a random parameter of in_channels rows and 3 columns;
    # branch decided by the first dimension of the parameter shape.
    def __init__(self, in_channels):
        # [FUNCTION DOC] HELPER: Creates the random parameter with the given row count and 3
        # columns.
        super().__init__()
        self.param = torch.nn.Parameter(torch.randn(in_channels, 3))

    def no_relu(self):
        # [FUNCTION DOC] HELPER: True if the first entry of the parameter shape is below 10.
        return self.param.shape[0] < 10


class MyModuleParamSize(MyModuleBase):
    # [CLASS DOC] HELPER: Same as the shape variant but queries the size method of the
    # parameter.
    def __init__(self, in_channels):
        # [FUNCTION DOC] HELPER: Creates the random parameter with the given row count and 3
        # columns.
        super().__init__()
        self.param = torch.nn.Parameter(torch.randn(in_channels, 3))

    def no_relu(self):
        # [FUNCTION DOC] HELPER: True if the first entry of the parameter size is below 10.
        return self.param.size()[0] < 10


class MyModuleParamDim(MyModuleBase):
    # [CLASS DOC] HELPER: Module that takes its parameter from the caller; branch decided by
    # the number of dimensions, via the dim method.
    def __init__(self, param):
        # [FUNCTION DOC] HELPER: Stores the supplied parameter.
        super().__init__()
        self.param = param

    def get_mul_matrix(self):
        # [FUNCTION DOC] HELPER: Returns the first slice of the parameter if it has three
        # dimensions, otherwise the parameter itself.
        return self.param[0] if (self.param.dim() == 3) else self.param

    def no_relu(self):
        # [FUNCTION DOC] HELPER: True if the parameter has three dimensions.
        return self.param.dim() == 3


class MyModuleParamNDim(MyModuleBase):
    # [CLASS DOC] HELPER: Like the dim variant but queries the ndim attribute.
    def __init__(self, param):
        # [FUNCTION DOC] HELPER: Stores the supplied parameter.
        super().__init__()
        self.param = param

    def get_mul_matrix(self):
        # [FUNCTION DOC] HELPER: Returns the first slice of the parameter if its ndim is
        # three, otherwise the parameter itself.
        return self.param[0] if (self.param.ndim == 3) else self.param

    def no_relu(self):
        # [FUNCTION DOC] HELPER: True if the parameter's ndim is three.
        return self.param.ndim == 3


class MyModuleParamNumEl(MyModuleBase):
    # [CLASS DOC] HELPER: Module with a random parameter of in_channels rows and 3 columns;
    # branch decided by the element count obtained with the numel method.
    def __init__(self, in_channels):
        # [FUNCTION DOC] HELPER: Creates the random parameter with the given row count and 3
        # columns.
        super().__init__()
        self.param = torch.nn.Parameter(torch.randn(in_channels, 3))

    def no_relu(self):
        # [FUNCTION DOC] HELPER: True if the parameter has fewer than 30 elements.
        return self.param.numel() < 10 * 3


class MyModuleParamNElement(MyModuleBase):
    # [CLASS DOC] HELPER: Same as the numel variant but queries the nelement method.
    def __init__(self, in_channels):
        # [FUNCTION DOC] HELPER: Creates the random parameter with the given row count and 3
        # columns.
        super().__init__()
        self.param = torch.nn.Parameter(torch.randn(in_channels, 3))

    def no_relu(self):
        # [FUNCTION DOC] HELPER: True if the parameter has fewer than 30 elements.
        return self.param.nelement() < 10 * 3


class TestConstParamShapeInControlFlow(TestCase):
    # [CLASS DOC] Tests that a tracer with constant parameter shapes resolves shape-
    # dependent branches at trace time. One shared verifier is used by tests for shape,
    # size, dim, ndim, numel and nelement queries. No state.
    def verify_mm_relu_mods(self, mm_only_mod, relu_mod):
        # [FUNCTION DOC] HELPER: Verifies one module takes the no-relu path and another
        # takes the relu path. For each module: checks eager output against the expected
        # formula, traces with a tracer configured to treat parameter shapes as constants,
        # wraps the graph in a graph module and checks its output with assert_close. Then
        # requires the matrix multiply function among node targets in both graphs and relu
        # only in the second, raising AssertionError otherwise. Inputs are random: 10 by 5
        # for the first module, 10 by 15 for the second.
        """
        Verify one module only does a mm op while the other
        performs both mm and relu ops in cascade
        """
        x = torch.randn(10, 5)
        torch.testing.assert_close(
            mm_only_mod(x), torch.mm(x, mm_only_mod.get_mul_matrix())
        )
        tracer = torch.fx.Tracer(param_shapes_constant=True)
        traced_graph = tracer.trace(mm_only_mod)

        # verify the graph module calculates the same result
        graph_mod_mm = torch.fx.GraphModule(mm_only_mod, traced_graph)
        torch.testing.assert_close(
            graph_mod_mm(x), torch.mm(x, mm_only_mod.get_mul_matrix())
        )

        # Make a new module with different parameter shape to go down the different
        # code path
        x = torch.randn(10, 15)
        torch.testing.assert_close(
            relu_mod(x), torch.relu(torch.mm(x, relu_mod.get_mul_matrix()))
        )

        tracer2 = torch.fx.Tracer(param_shapes_constant=True)
        traced_graph2 = tracer2.trace(relu_mod)

        # verify the graph module calculates the same result
        graph_mod_relu = torch.fx.GraphModule(relu_mod, traced_graph2)
        torch.testing.assert_close(
            graph_mod_relu(x), torch.relu(torch.mm(x, relu_mod.get_mul_matrix()))
        )

        graph1_node_targets = [n.target for n in traced_graph.nodes]
        graph2_node_targets = [n.target for n in traced_graph2.nodes]

        # the second graph has an extra relu function call node
        if not (torch.mm in graph1_node_targets and torch.mm in graph2_node_targets):
            raise AssertionError("Expected torch.mm in both graph node targets")
        if not (
            torch.relu not in graph1_node_targets and torch.relu in graph2_node_targets
        ):
            raise AssertionError(
                "Expected torch.relu not in graph1 but in graph2 node targets"
            )

    def test_param_shape_const(self):
        # [FUNCTION DOC] Builds two modules (5 versus 15 parameter rows) for the shape query
        # and runs the shared verifier.
        mymod = MyModuleParamShape(in_channels=5)
        mymod2 = MyModuleParamShape(in_channels=15)
        self.verify_mm_relu_mods(mymod, mymod2)

    def test_param_size_const(self):
        # [FUNCTION DOC] Builds two modules (5 versus 15 parameter rows) for the size query
        # and runs the shared verifier.
        mymod = MyModuleParamSize(in_channels=5)
        mymod2 = MyModuleParamSize(in_channels=15)
        self.verify_mm_relu_mods(mymod, mymod2)

    def test_param_dim_const(self):
        # [FUNCTION DOC] Builds two modules (three dimensional parameter versus a two
        # dimensional one) for the dim query and runs the shared verifier.
        mymod = MyModuleParamDim(torch.nn.Parameter(torch.randn(2, 5, 3)))
        mymod2 = MyModuleParamDim(torch.nn.Parameter(torch.randn(15, 3)))
        self.verify_mm_relu_mods(mymod, mymod2)

    def test_param_ndim_const(self):
        # [FUNCTION DOC] Builds two modules (three dimensional parameter versus a two
        # dimensional one) for the ndim query and runs the shared verifier.
        mymod = MyModuleParamNDim(torch.nn.Parameter(torch.randn(2, 5, 3)))
        mymod2 = MyModuleParamNDim(torch.nn.Parameter(torch.randn(15, 3)))
        self.verify_mm_relu_mods(mymod, mymod2)

    def test_param_numel_const(self):
        # [FUNCTION DOC] Builds two modules (5 versus 15 parameter rows) for the numel query
        # and runs the shared verifier.
        mymod = MyModuleParamNumEl(in_channels=5)
        mymod2 = MyModuleParamNumEl(in_channels=15)
        self.verify_mm_relu_mods(mymod, mymod2)

    def test_param_nelement_const(self):
        # [FUNCTION DOC] Builds two modules (5 versus 15 parameter rows) for the nelement
        # query and runs the shared verifier.
        mymod = MyModuleParamNElement(in_channels=5)
        mymod2 = MyModuleParamNElement(in_channels=15)
        self.verify_mm_relu_mods(mymod, mymod2)


if __name__ == "__main__":
    raise_on_run_directly("test/test_fx.py")
