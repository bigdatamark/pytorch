# [MODULE DOC] Pass management infrastructure for FX/nn.Module transformations.
# Provides pass_result_wrapper (adapts legacy passes to return PassResult),
# this_before_that_pass_constraint (builds ordering constraints), and the
# PassManager class that orders passes by constraints, runs them repeatedly
# up to a step limit, and optionally runs checks after each pass.
# Import-time side effect: sets this module's logger level to WARNING.
import inspect
import logging
from collections.abc import Callable
from functools import wraps
from queue import Queue
from typing import Any

import torch.nn as nn
from torch.fx._compatibility import compatibility
from torch.fx.graph_module import GraphModule
from torch.fx.passes.infra.pass_base import PassResult


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

__all__ = ["pass_result_wrapper", "this_before_that_pass_constraint", "PassManager"]


@compatibility(is_backward_compatible=False)
def pass_result_wrapper(fn: Callable[..., Any]) -> Callable[..., PassResult | None]:
    """
    Wrapper for passes which currently do not return a PassResult.
    This wrapper makes them return a PassResult containing the modified object
    and True for the "modified" flag.

    Args:
        fn (Callable[Module, Any])

    Returns:
        wrapped_fn (Callable[Module, PassResult])
    """
    # [FUNCTION DOC] Wrap a pass so it returns a PassResult. Returns None if fn is
    # None. The wrapper calls fn on the module: a None result becomes a PassResult
    # of the input module flagged modified, a PassResult is passed through, and a
    # returned module is wrapped as modified. Any other result type yields None.
    # For non-function callables the wrapper name is set to the type name of fn.
    if fn is None:
        # pyrefly: ignore [bad-return]
        return None

    @wraps(fn)
    def wrapped_fn(gm: nn.Module) -> PassResult | None:
        # [FUNCTION DOC] HELPER: Run the wrapped pass and normalize its result to a
        # PassResult (see pass_result_wrapper); returns None for unsupported types.
        res = fn(gm)
        if res is None:
            return PassResult(gm, True)
        if isinstance(res, PassResult):
            return res
        elif isinstance(res, nn.Module):
            return PassResult(res, True)

    if not inspect.isfunction(fn):
        wrapped_fn.__name__ = type(fn).__name__

    return wrapped_fn


def _validate_pass_schedule_constraint(
    constraint: Callable[[Callable[..., Any], Callable[..., Any]], bool],
    passes: list[Callable[..., Any]],
) -> None:
    # [FUNCTION DOC] HELPER: Check every ordered pair (earlier, later) in the pass
    # list against the constraint. Raises RuntimeError naming the offending pair
    # on the first violation. No mutation.
    # KNOWN ISSUE: the index reported for the second pass is relative to the slice
    # after the first pass, not its absolute position in the list.
    for i, a in enumerate(passes):
        for j, b in enumerate(passes[i + 1 :]):
            if constraint(a, b):
                continue
            raise RuntimeError(
                f"pass schedule constraint violated. Expected {a} before {b}"
                f" but found {a} at index {i} and {b} at index {j} in pass"
                f" list."
            )


def _topological_sort_passes(
    passes: list[Callable[..., Any]], constraints: list[Callable[..., Any]]
) -> list[Callable[..., Any]]:
    """
    Args
        passes: Passes that we are ordering
        constraints: Constraints applied on these passes

    Returns
        A sorted list of callables and a boolean of if a circular dependency
        existed
    """
    # [FUNCTION DOC] HELPER: Order passes using a queue-based topological sort
    # (Kahn style). For each ordered pair where a constraint returns False, the
    # first pass gets an edge from the second and its in-degree is incremented.
    # Returns the input list unchanged if there are no constraints. Raises
    # RuntimeError listing passes left over if a circular dependency exists.
    # KNOWN ISSUE: the docstring says a boolean about cycles is returned, but only
    # the sorted list is returned.
    if len(constraints) == 0:
        return passes

    # Construct a graph mapping nodes to a list of their users
    graph: dict[Callable[..., Any], list[Callable[..., Any]]] = {p: [] for p in passes}
    indegree_map: dict[Callable[..., Any], int] = dict.fromkeys(passes, 0)
    candidates: Queue[Callable[..., Any]] = Queue()
    for a in passes:
        for b in passes:
            if a == b:
                continue

            for constraint in constraints:
                if not constraint(a, b):
                    graph[b].append(a)
                    indegree_map[a] += 1

        if indegree_map[a] == 0:
            candidates.put(a)

    visited: dict[Callable[..., Any], bool] = dict.fromkeys(passes, False)
    sorted_passes: list[Callable[..., Any]] = []

    while not candidates.empty():
        p = candidates.get()
        sorted_passes.append(p)
        visited[p] = True

        for n in graph[p]:
            if not visited[n]:
                indegree_map[n] -= 1
                if indegree_map[n] == 0:
                    candidates.put(n)

    # Check if there are unvisited nodes (aka cycles in the graph)
    cycle_passes = list(filter(lambda p: indegree_map[p] != 0, indegree_map.keys()))
    if len(cycle_passes) != 0:
        error = (
            f"Circular dependency detected within the following passes: {cycle_passes}"
        )
        raise RuntimeError(error)

    return sorted_passes


@compatibility(is_backward_compatible=False)
def this_before_that_pass_constraint(
    this: Callable[..., Any], that: Callable[..., Any]
) -> Callable[[Callable[..., Any], Callable[..., Any]], bool]:
    """
    Defines a partial order ('depends on' function) where ``this`` must occur
    before ``that``.

    For example, the following pass list and constraint list would be invalid::

        passes = [pass_b, pass_a]

        constraints = [this_before_that_pass_constraint(pass_a, pass_b)]

    Args:
        this (Callable): pass which should occur first
        that (Callable): pass which should occur later

    Returns:
        depends_on (Callable[[Object, Object], bool])
    """
    # [FUNCTION DOC] Build a constraint callable enforcing that pass this precedes
    # pass that. The returned callable takes two passes and returns False only
    # when the first is that and the second is this, True otherwise.

    def depends_on(a: Callable[..., Any], b: Callable[..., Any]) -> bool:
        # [FUNCTION DOC] HELPER: Constraint predicate; False only for the pair
        # (that, this), meaning that-before-this is a violation.
        return a != that or b != this

    return depends_on


@compatibility(is_backward_compatible=False)
class PassManager:
    # [CLASS DOC] Callable that holds an ordered list of passes plus ordering
    # constraints, and applies the passes to a module. State: passes, constraints,
    # steps (max repetitions, default 1), run_checks_after_each_pass,
    # suppress_check_failures (stored but never consulted), and a _validated flag
    # reset whenever passes or constraints change. Key methods: add_pass,
    # add_constraint, validate_constraints, solve_constraints, add_checks, __call__.
    """
    Construct a PassManager.

    Collects passes and constraints. This defines the pass schedule, manages
    pass constraints and pass execution.

    Args:
        passes (Optional[List[Callable]]): List of passes. A pass is a
            callable which modifies an object and returns a PassResult
        constraint (Optional[List[Callable]]): List of constraints. A
            constraint is a callable which takes two passes (A, B) and returns
            True if A depends on B and False otherwise. See implementation of
            `this_before_that_pass_constraint` for example.
        steps (int): Max number of times we run the passes (default = 1).
        run_checks_after_each_pass (bool): Whether to run checks and linting
            after each pass
        suppress_check_failures (bool): Whether to raise errors when running
            checks
    """

    passes: list[Callable[..., PassResult | None]]
    constraints: list[Callable[[Callable[..., Any], Callable[..., Any]], bool]]
    _validated: bool = False
    steps: int = 1

    def __init__(
        self,
        passes: list[Callable[..., PassResult | None]] | None = None,
        constraints: list[Callable[[Callable[..., Any], Callable[..., Any]], bool]]
        | None = None,
        steps: int | None = None,
        run_checks_after_each_pass: bool = False,
        suppress_check_failures: bool = False,
    ) -> None:
        # [FUNCTION DOC] Initialize with optional passes and constraints (default to
        # empty lists; the given lists are stored, not copied). A falsy steps keeps
        # the class default of 1. Stores the check-related flags.
        self.passes = passes or []
        self.constraints = constraints or []
        if steps:
            self.steps = steps

        self.run_checks_after_each_pass = run_checks_after_each_pass
        self.suppress_check_failures = suppress_check_failures

    def add_pass(self, _pass: Callable[..., PassResult | None]) -> None:
        """
        Adds a pass into the current list of passes.
        """
        # [FUNCTION DOC] Append a pass to the list and mark the schedule unvalidated.
        self.passes.append(_pass)
        self._validated = False

    def add_constraint(
        self, constraint: Callable[[Callable[..., Any], Callable[..., Any]], bool]
    ) -> None:
        """
        Adds a constraint into the current list of constraints.
        """
        # [FUNCTION DOC] Append a constraint and mark the schedule unvalidated.
        self.constraints.append(constraint)
        self._validated = False

    def validate_constraints(self) -> None:
        """
        Validates that current pass schedule defined by `self.passes` is valid
        according to all constraints in `self.constraints`
        """
        # [FUNCTION DOC] Verify the current pass order against every constraint.
        # Skips work if already validated. Raises RuntimeError on a violation.
        # Sets the validated flag on success.
        if self._validated:
            return
        for constraint in self.constraints:
            _validate_pass_schedule_constraint(constraint, self.passes)
        self._validated = True

    def solve_constraints(self) -> None:
        """
        Finds a valid traversal order based on the given constraints and orders
        the passes based on this order.

        If a circular dependency exists between the constraints and steps = 1,
        then we will raise an error because if steps != 1 this means that we
        will re-run the passes, allowing for circular dependencies.
        """
        # [FUNCTION DOC] Reorder the passes by topological sort over the constraints
        # and mark the schedule validated. Raises RuntimeError on circular
        # dependencies.
        # KNOWN ISSUE: the docstring says a cycle is only an error when steps is 1,
        # but the sort raises on a cycle regardless of steps.
        self.passes = _topological_sort_passes(self.passes, self.constraints)
        self._validated = True

    def add_checks(self, check: Callable[[nn.Module], None]) -> None:
        """
        Adds a function which takes runs various checks on a given graph module.
        This function is run before and after each pass if the
        `run_checks_after_each_pass` flag is enabled.
        """
        # [FUNCTION DOC] Install a check callable as this instance's check method.
        # Raises TypeError unless the callable takes exactly one parameter.
        # Side effect: sets an instance attribute shadowing the default no-op check.
        # KNOWN ISSUE: the docstring says checks run before and after each pass, but
        # the call method runs them once up front and after each pass only when
        # run_checks_after_each_pass is set.
        sig = inspect.signature(check)

        if len(list(sig.parameters.values())) != 1:
            raise TypeError(
                "PassManager check function should only take in one variable, a module"
            )

        setattr(self, "check", check)  # noqa: B010

    def check(self, module: nn.Module) -> None:
        # [FUNCTION DOC] Default no-op check hook; replaced per instance by add_checks.
        pass

    def __call__(self, module: nn.Module) -> PassResult:
        """
        Runs a list of passes in the order based on `self.passes` on the given
        graph module. Each time a pass is run, checks and linting will be run on
        the graph module if `run_checks_after_each_pass` is set.

        If the module is a graph module, we will run the list of passes until
        the graph stops changing, or until `steps` number of times.
        """
        # [FUNCTION DOC] Run all passes on the module. Solves constraints first if
        # not yet validated, runs check once, then loops up to steps times, applying
        # each pass in order, taking the result's module, recompiling GraphModules
        # and optionally checking after each pass. Stops early when a full round
        # reports no modification. Returns a PassResult with the final module and
        # whether anything changed. Raises TypeError (wrapped) if a pass returns a
        # non-PassResult; any pass failure is re-raised as a generic Exception that
        # names the failing pass and the preceding passes.
        # Order the passes based on the constraints
        if not self._validated:
            self.solve_constraints()

        # Check graph invariants
        self.check(module)

        # Run the set of passes `steps` number of times or until the graph stops
        # changing
        overall_modified = False
        for _ in range(self.steps):
            modified = False

            # Run the set of passes on the graph module
            for i, fn in enumerate(self.passes):
                fn_name = fn.__name__ if inspect.isfunction(fn) else type(fn).__name__
                logger.debug("Running pass '%s'", fn_name)

                try:
                    res = fn(module)

                    if not isinstance(res, PassResult) and not hasattr(
                        res, "graph_module"
                    ):
                        raise TypeError(
                            f"The result of the pass {fn_name} should be type PassResult."
                            + "Please wrap it with pass_result_wrapper()"
                        )
                    # pyrefly: ignore[missing-attribute]
                    module = res.graph_module
                    # pyrefly: ignore[missing-attribute]
                    modified = modified or res.modified

                    if isinstance(module, GraphModule):
                        logger.debug("Graph after pass '%s': %s", fn_name, module.graph)
                        module.recompile()

                    # Check graph invariants
                    if self.run_checks_after_each_pass:
                        self.check(module)

                except Exception as e:
                    prev_pass_names = [
                        p.__name__ if inspect.isfunction(p) else type(p).__name__
                        for p in self.passes[:i]
                    ]
                    msg = f"An error occurred when running the '{fn_name}' pass after the following passes: {prev_pass_names}"
                    raise Exception(msg) from e  # noqa: TRY002

            # If the graph no longer changes, then we can stop running these passes
            overall_modified = overall_modified or modified
            if not modified:
                break

        return PassResult(module, overall_modified)
