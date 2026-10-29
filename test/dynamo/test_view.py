# Owner(s): ["module: dynamo"]
# [MODULE DOC] Dynamo tests for view and reshape with data-dependent or tensor-valued shape
#   arguments: sizes derived from scalar outputs of tensors, and tensor shape parameters passed
#   to the view operator, the view method and the reshape function. Compiled results are compared
#   with eager results. No import-time side effects.
import torch
import torch._dynamo
import torch._dynamo.test_case


@torch._dynamo.config.patch("capture_scalar_outputs", True)
class ViewTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case (with scalar-output capture enabled for the whole class) covering
    #   compiled view and reshape with unbacked sizes and tensor-valued shape parameters.
    def test_view_to_2d(self):
        # [FUNCTION DOC] Full-graph compile of a function that reads two sizes from a tensor,
        #   builds a random tensor of their product elements, and views it as two dimensions with
        #   a runtime second dimension. Passes if compilation and execution succeed; no result is
        #   compared.
        @torch.compile(fullgraph=True, backend="eager")
        def f(t, _u0):
            # [FUNCTION DOC] HELPER: Compiled body: take two scalar values from the tensor,
            #   create a random vector of length equal to their product, and view it with an
            #   inferred first dimension and the given second dimension.
            u0 = t[0].item()
            u1 = t[1].item()
            n = u0 * u1
            a = torch.randn(n)
            return a.view(-1, _u0)

        t = torch.tensor([2, 4], dtype=torch.int32)
        f(t, 2)

    def test_view_to_1d(self):
        # [FUNCTION DOC] Full-graph compile of a function that creates a two-dimensional random
        #   tensor with sizes taken from tensor scalars and views it as one dimension of the
        #   given length (8). Passes if compilation and execution succeed.
        @torch.compile(fullgraph=True, backend="eager")
        def f(t, _n):
            # [FUNCTION DOC] HELPER: Compiled body: build a random matrix with sizes from two
            #   tensor scalars and view it flat with the given length.
            u0 = t[0].item()
            u1 = t[1].item()
            a = torch.randn(u0, u1)
            return a.view(_n)

        t = torch.tensor([2, 4], dtype=torch.int32)
        f(t, 8)

    def test_view_with_tensor_shape_params(self):
        # [FUNCTION DOC] Regression test for a reported issue: the aten view operator called with
        #   a list of zero-dimensional int tensors as the shape must compile (eager backend) and
        #   match eager output.
        # Test for issue #156720: aten.view.default with tensor shape parameters
        class TestModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward calls the aten view operator with the
            #   given tensor and shape parameters.
            def forward(self, x, shape_params):
                # [FUNCTION DOC] HELPER: Return the aten view operator applied to the input with
                #   the shape parameter list.
                return torch.ops.aten.view.default(x, shape_params)

        x = torch.randn(24)
        shape_params = [
            torch.tensor(2, dtype=torch.int32),
            torch.tensor(3, dtype=torch.int32),
            torch.tensor(4, dtype=torch.int32),
        ]

        model = TestModel()
        expected = model(x, shape_params)

        compiled_model = torch.compile(model, backend="eager")
        result = compiled_model(x, shape_params)

        torch.testing.assert_close(result, expected)

    def test_tensor_view_with_tensor_shape_params(self):
        # [FUNCTION DOC] The view method given a tuple of zero-dimensional int tensors as shape
        #   must compile (eager backend) and match eager output.
        # Test tensor.view() method with tensor shape parameters (list version)
        class TestModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward calls the view method with a shape
            #   sequence.
            def forward(self, x, shape_params):
                # [FUNCTION DOC] HELPER: Return the input viewed with the shape sequence
                #   argument.
                return x.view(shape_params)

        x = torch.randn(24)
        shape_params = (
            torch.tensor(2, dtype=torch.int32),
            torch.tensor(3, dtype=torch.int32),
            torch.tensor(4, dtype=torch.int32),
        )

        model = TestModel()
        expected = model(x, shape_params)

        compiled_model = torch.compile(model, backend="eager")
        result = compiled_model(x, shape_params)

        torch.testing.assert_close(result, expected)

    def test_tensor_view_with_tensor_args(self):
        # [FUNCTION DOC] The view method given three separate zero-dimensional int tensors as
        #   dimension arguments must compile (eager backend) and match eager output.
        # Test tensor.view() method with individual tensor arguments
        class TestModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward calls the view method with three separate
            #   dimension arguments.
            def forward(self, x, dim1, dim2, dim3):
                # [FUNCTION DOC] HELPER: Return the input viewed with the three dimension
                #   arguments.
                return x.view(dim1, dim2, dim3)

        x = torch.randn(24)
        dim1 = torch.tensor(2, dtype=torch.int32)
        dim2 = torch.tensor(3, dtype=torch.int32)
        dim3 = torch.tensor(4, dtype=torch.int32)

        model = TestModel()
        expected = model(x, dim1, dim2, dim3)

        compiled_model = torch.compile(model, backend="eager")
        result = compiled_model(x, dim1, dim2, dim3)

        torch.testing.assert_close(result, expected)

    def test_torch_reshape_with_tensor_shape_params(self):
        # [FUNCTION DOC] The reshape function given a list of zero-dimensional int tensors as
        #   shape must compile (eager backend) and match eager output.
        # Test torch.reshape() function with tensor shape parameters
        def test_fn(x, shape_params):
            # [FUNCTION DOC] HELPER: Return the reshape function applied to the input with the
            #   shape parameter list.
            return torch.reshape(x, shape_params)

        x = torch.randn(24)
        shape_params = [
            torch.tensor(2, dtype=torch.int32),
            torch.tensor(3, dtype=torch.int32),
            torch.tensor(4, dtype=torch.int32),
        ]

        expected = test_fn(x, shape_params)

        compiled_fn = torch.compile(test_fn, backend="eager")
        result = compiled_fn(x, shape_params)

        torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
