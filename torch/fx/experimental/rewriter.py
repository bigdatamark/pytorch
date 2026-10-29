# [MODULE DOC]
# AST-based source rewriting so that FX symbolic tracing can handle constructs it
# otherwise cannot: assert statements become calls to the traceable torch._assert and
# annotated assignments become calls to an annotate function. Provides AST_Rewriter,
# RewritingTracer (a Tracer that rewrites its root before tracing) and the private
# _rewrite helper. No import-time side effects beyond imports.
import ast
import copy
import functools
import inspect
import textwrap
from collections.abc import Callable
from types import FunctionType
from typing import Any, cast

import torch
from torch._sources import normalize_source_lines
from torch.fx._symbolic_trace import Tracer
from torch.fx.graph import Graph


class AST_Rewriter(ast.NodeTransformer):
    # [CLASS DOC]
    # NodeTransformer that rewrites the source of a function (typically a forward method)
    # into an FX-traceable equivalent. Stateless; rewrite is the entry point and the
    # visit_Assert and visit_AnnAssign methods define the node replacements.
    """
    Take a FunctionType object representing a `forward` method, then
    perform an AST rewrite to swap out nodes that are not symbolically
    traceable with a callsite to the FX alternative.

    To support swapping out an AST node, define a new `visit` method on
    that node. For more details, see:
    https://docs.python.org/3/library/ast.html#ast.NodeTransformer
    """

    # This function checks for new keys added in the globals dict. TorchDynamo
    # can insert new keys in the global dict and upset the check. Therefore, put
    # a disable here. This function is an optimization pass and not really
    # suitable for dynamo tracing anyways.
    @torch._dynamo.disable
    def rewrite(self, fn: FunctionType) -> FunctionType:
        # [FUNCTION DOC]
        # Gets and dedents the source of fn, parses it, applies the node transformations,
        # compiles and executes the result in a copy of fn's globals, and returns the single
        # newly defined function rebuilt to use fn's original globals. Raises AssertionError if
        # executing the code does not define exactly one new global name; source lookup
        # errors from inspect propagate. Mutates only the copied globals mapping. Disabled
        # for dynamo tracing.
        # Normalize the source lines
        sourcelines, _ = inspect.getsourcelines(fn)
        sourcelines = normalize_source_lines(sourcelines)
        source = "".join(sourcelines)
        normalized_str = textwrap.dedent(source)

        # Rewrite the original AST
        source_ast = ast.parse(normalized_str)
        dest_ast = ast.fix_missing_locations(self.visit(source_ast))

        # Pull out the compiled function from the newly-created Module
        code = compile(dest_ast, "", "exec")
        globals_dict = copy.copy(fn.__globals__)
        keys_before = set(globals_dict.keys())
        exec(code, globals_dict)
        new_keys = list(set(globals_dict.keys()) - keys_before)
        if len(new_keys) != 1:
            raise AssertionError(f"Expected 1 new key, got {len(new_keys)}")
        fn_compiled = globals_dict[new_keys[0]]

        # return the compiled function with the original globals
        def change_func_globals(
            f: FunctionType, globals: dict[str, object]
        ) -> FunctionType:
            # [FUNCTION DOC] HELPER: Returns a copy of function f bound to the given globals
            # mapping, preserving name, defaults, closure, wrapper metadata and keyword defaults.
            """Based on https://stackoverflow.com/a/13503277/2988730 (@unutbu)"""
            # __globals__ is a private member of the function class
            # so we have to copy the function, f, all of its member, except f.__globals__
            g = FunctionType(
                f.__code__,
                globals,
                name=f.__name__,
                argdefs=f.__defaults__,
                closure=f.__closure__,
            )
            g = functools.update_wrapper(g, f)
            g.__kwdefaults__ = copy.copy(f.__kwdefaults__)  # type:ignore[attr-defined]
            return g  # pyrefly: ignore [bad-return]

        # Return the correct FunctionType object
        return change_func_globals(fn_compiled, globals=fn.__globals__)

    def visit_Assert(self, node: ast.Assert) -> ast.Expr:
        # [FUNCTION DOC]
        # Replaces an assert statement with an expression statement calling torch._assert
        # with the test and the message (an empty string constant when no message is given).
        # Returns the new node with the original location. Raises AssertionError if the
        # internally parsed template is not a call expression.
        """
        Swap out the Assert node (Python's `assert`) with a callsite to the
        symbolically-traceable torch._assert function
        """
        # Create the Call node
        n = ast.parse("torch._assert()", mode="eval")
        if not isinstance(n, ast.Expression):
            raise AssertionError(f"Expected ast.Expression, got {type(n)}")
        call_node = n.body
        if not isinstance(call_node, ast.Call):
            raise AssertionError(f"Expected ast.Call, got {type(call_node)}")
        msg = node.msg if node.msg else ast.Constant(value="", kind=None)
        call_node.args = [node.test, msg]

        # Ensure that the new node conforms to the Python AST grammar
        expr_wrapper = ast.Expr(value=call_node)

        # Return the new Call node to signify that we want to use it as
        # a replacement for the original _assert node
        return ast.copy_location(expr_wrapper, node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.Assign:
        # [FUNCTION DOC]
        # Replaces an annotated assignment with a plain assignment of a call to a name
        # annotate, passing the assigned value and the annotation expression.
        # KNOWN ISSUE: an annotation without a value (node.value is None) is not handled and
        # yields an invalid argument; no location is copied to the new node (callers rely on
        # fix_missing_locations); the docstring example output shows a different tuple form.
        """
        Swap out Python's AnnAssign with an Assign node where the annotation function is called.
        Example:
             Original:
             y: Tensor_Type(1,2,3, Dyn) = f2(x)
            Output:
             y = annotate(f2(x),Tensor_Type((1,2,3,Dyn)))
        """
        return ast.Assign(
            targets=[node.target],
            value=ast.Call(
                func=ast.Name(id="annotate", ctx=ast.Load()),
                # pyrefly: ignore [bad-argument-type]
                args=[node.value, node.annotation],
                keywords=[],
            ),
        )


class RewritingTracer(Tracer):
    # [CLASS DOC]
    # Tracer that first rewrites the root module or function's source using AST_Rewriter
    # (via _rewrite), then traces the rewritten object with the standard Tracer logic.
    def trace(
        self,
        root: torch.nn.Module | Callable[..., Any],
        concrete_args: dict[str, Any] | None = None,
    ) -> Graph:
        # [FUNCTION DOC] Rewrites root with _rewrite and traces it, forwarding concrete_args;
        # returns the resulting Graph.
        return super().trace(_rewrite(root), concrete_args)


def _rewrite(
    fn: torch.nn.Module | Callable[..., Any],
) -> torch.nn.Module | Callable[..., Any]:
    # [FUNCTION DOC] HELPER: For a Module, returns a new module hierarchy (shallow copies of
    # attributes, recursively rewriting submodules) whose forward methods are rewritten;
    # for any other callable returns the rewritten function. Does not modify the input.
    if isinstance(fn, torch.nn.Module):
        # Rewrite this module's `forward` as well as the `forward`s of
        # all of this module's recursive descendants. Return the new,
        # rewritten module hierarchy.
        def rewrite_module(m: torch.nn.Module) -> torch.nn.Module:
            # [FUNCTION DOC] HELPER: Builds a new Module subclass instance copying m's
            # attributes with a rewritten forward; submodule attributes are rewritten recursively.
            class RewrittenModule(torch.nn.Module):
                # [CLASS DOC] HELPER: Dynamically created Module copy of an original module; its forward is replaced after class creation.
                def __init__(self, orig: torch.nn.Module) -> None:
                    # [FUNCTION DOC] HELPER: Copies every entry of orig's attribute dictionary, recursively rewriting Module values.
                    super().__init__()
                    for k, v in orig.__dict__.items():
                        if isinstance(v, torch.nn.Module):
                            self.__dict__[k] = copy.copy(rewrite_module(v))
                        else:
                            self.__dict__[k] = copy.copy(v)

            RewrittenModule.forward = AST_Rewriter().rewrite(
                cast(FunctionType, m.forward)
            )
            return RewrittenModule(m)

        return rewrite_module(fn)
    else:
        # Rewrite this single free function
        return AST_Rewriter().rewrite(cast(FunctionType, fn))
