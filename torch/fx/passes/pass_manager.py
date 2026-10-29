# [MODULE DOC] Older, simpler pass manager for arbitrary objects (not just FX).
# Provides wrappers (inplace_wrapper, log_hook, loop_pass), ordering-constraint
# builders (this_before_that_pass_constraint, these_before_those_pass_constraint),
# and a PassManager that validates constraints and applies passes sequentially,
# feeding each pass output into the next. No import-time side effects besides
# creating a module logger.
import logging
from collections.abc import Callable
from functools import wraps
from inspect import unwrap
from typing import Any, Concatenate, ParamSpec, TypeVar


_P = ParamSpec("_P")
_R = TypeVar("_R")
_T = TypeVar("_T")


logger = logging.getLogger(__name__)

__all__ = [
    "PassManager",
    "inplace_wrapper",
    "log_hook",
    "loop_pass",
    "this_before_that_pass_constraint",
    "these_before_those_pass_constraint",
]


# for callables which modify object inplace and return something other than
# the object on which they act
def inplace_wrapper(
    fn: Callable[Concatenate[_T, _P], Any],
) -> Callable[Concatenate[_T, _P], _T]:
    """
    Convenience wrapper for passes which modify an object inplace. This
    wrapper makes them return the modified object instead.

    Args:
        fn (Callable[Object, Any])

    Returns:
        wrapped_fn (Callable[Object, Object])
    """
    # [FUNCTION DOC] Adapt a pass that mutates its first argument in place so that
    # it returns that argument. Returns the wrapper; the wrapper discards the
    # original pass return value.

    @wraps(fn)
    def wrapped_fn(gm: _T, *args: _P.args, **kwargs: _P.kwargs) -> _T:
        # [FUNCTION DOC] HELPER: Call the pass with all arguments, then return the
        # first argument.
        fn(gm, *args, **kwargs)
        return gm

    return wrapped_fn


def log_hook(fn: Callable[_P, _R], level: int = logging.INFO) -> Callable[_P, _R]:
    """
    Logs callable output.

    This is useful for logging output of passes. Note ``inplace_wrapper`` replaces
    the pass output with the modified object. If we want to log the original
    output, apply this wrapper before ``inplace_wrapper``.

    Example::

        def my_pass(d: Dict) -> bool:
            changed = False
            if "foo" in d:
                d["foo"] = "bar"
                changed = True
            return changed


        pm = PassManager(passes=[inplace_wrapper(log_hook(my_pass))])

    Args:
        fn (Callable[Type1, Type2])
        level: logging level (e.g. logging.INFO)

    Returns:
        wrapped_fn (Callable[Type1, Type2])
    """
    # [FUNCTION DOC] Wrap a callable so its return value is logged at the given
    # level (default INFO) through the module logger, then returned unchanged.
    # Returns the wrapper.

    @wraps(fn)
    def wrapped_fn(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        # [FUNCTION DOC] HELPER: Call fn, log the callable and its result, and
        # return the result.
        val = fn(*args, **kwargs)
        logger.log(level, "Ran pass %s\t Return value: %s", fn, val)
        return val

    return wrapped_fn


def loop_pass(
    base_pass: Callable[[_T], _T],
    n_iter: int | None = None,
    predicate: Callable[[_T], bool] | None = None,
) -> Callable[[_T], _T]:
    """
    Convenience wrapper for passes which need to be applied multiple times.

    Exactly one of `n_iter`or `predicate` must be specified.

    Args:
        base_pass (Callable[Object, Object]): pass to be applied in loop
        n_iter (int, optional): number of times to loop pass
        predicate (Callable[Object, bool], optional):

    """
    # [FUNCTION DOC] Build a pass that applies base_pass repeatedly, feeding each
    # output into the next call. Exactly one of n_iter (fixed count) or predicate
    # (loop while it is true on the current output) must be given, otherwise an
    # AssertionError is raised immediately. Returns the looping pass.
    if not ((n_iter is not None) ^ (predicate is not None)):
        raise AssertionError("Exactly one of `n_iter`or `predicate` must be specified.")

    @wraps(base_pass)
    def new_pass(source: _T) -> _T:
        # [FUNCTION DOC] HELPER: Apply base_pass n_iter times, or while predicate
        # holds. Raises RuntimeError if n_iter is not positive and no predicate was
        # given. Returns the final output.
        # KNOWN ISSUE: with n_iter of zero or negative and no predicate this raises
        # at call time rather than at construction time.
        output = source
        if n_iter is not None and n_iter > 0:
            for _ in range(n_iter):
                output = base_pass(output)
        elif predicate is not None:
            while predicate(output):
                output = base_pass(output)
        else:
            raise RuntimeError(
                f"loop_pass must be given positive int n_iter (given "
                f"{n_iter}) xor predicate (given {predicate})"
            )
        return output

    return new_pass


# Pass Schedule Constraints:
#
# Implemented as 'depends on' operators. A constraint is satisfied iff a list
# has a valid partial ordering according to this comparison operator.
def _validate_pass_schedule_constraint(
    constraint: Callable[[Callable[..., Any], Callable[..., Any]], bool],
    passes: list[Callable[..., Any]],
) -> None:
    # [FUNCTION DOC] HELPER: Check every ordered pair (earlier, later) in the pass
    # list against the constraint; raises RuntimeError naming the first violating
    # pair. No mutation.
    # KNOWN ISSUE: the index reported for the second pass is relative to the slice
    # after the first pass, and the message text lacks a space before it.
    for i, a in enumerate(passes):
        for j, b in enumerate(passes[i + 1 :]):
            if constraint(a, b):
                continue
            raise RuntimeError(
                f"pass schedule constraint violated. Expected {a} before {b}"
                f" but found {a} at index {i} and {b} at index{j} in pass"
                f" list."
            )


def this_before_that_pass_constraint(
    this: Callable[..., Any], that: Callable[..., Any]
) -> Callable[[Callable[..., Any], Callable[..., Any]], bool]:
    """
    Defines a partial order ('depends on' function) where `this` must occur
    before `that`.
    """
    # [FUNCTION DOC] Build a constraint requiring this to come before that, by direct
    # comparison of the passes (no unwrapping). Returns the predicate.

    def depends_on(a: Callable[..., Any], b: Callable[..., Any]) -> bool:
        # [FUNCTION DOC] HELPER: False only for the pair (that, this).
        return a != that or b != this

    return depends_on


def these_before_those_pass_constraint(
    these: Callable[..., Any], those: Callable[..., Any]
) -> Callable[[Callable[..., Any], Callable[..., Any]], bool]:
    """
    Defines a partial order ('depends on' function) where ``these`` must occur
    before ``those``. Where the inputs are 'unwrapped' before comparison.

    For example, the following pass list and constraint list would be invalid::

        passes = [
            loop_pass(pass_b, 3),
            loop_pass(pass_a, 5),
        ]

        constraints = [these_before_those_pass_constraint(pass_a, pass_b)]

    Args:
        these (Callable): pass which should occur first
        those (Callable): pass which should occur later

    Returns:
        depends_on (Callable[[Object, Object], bool])
    """
    # [FUNCTION DOC] Build a constraint requiring these to come before those, where
    # both candidate passes are first unwrapped through their wrapper chain (so
    # wrapped passes such as looped ones match). Returns the predicate.

    def depends_on(a: Callable[..., Any], b: Callable[..., Any]) -> bool:
        # [FUNCTION DOC] HELPER: False only when the unwrapped first pass is those
        # and the unwrapped second pass is these.
        return unwrap(a) != those or unwrap(b) != these

    return depends_on


class PassManager:
    # [CLASS DOC] Ordered container of passes and constraints. Each pass takes an
    # object and returns the next object. State: passes list, constraints list, and
    # a _validated flag cleared by any mutation. Key methods: add_pass,
    # add_constraint, remove_pass, replace_pass, validate, call (runs the chain).
    """
    Construct a PassManager.

    Collects passes and constraints. This defines the pass schedule, manages
    pass constraints and pass execution.

    Args:
        passes (Optional[List[Callable]]): list of passes. A pass is a
            callable which modifies an object and returns modified object
        constraint (Optional[List[Callable]]): list of constraints. A
            constraint is a callable which takes two passes (A, B) and returns
            True if A depends on B and False otherwise. See implementation of
            `this_before_that_pass_constraint` for example.
    """

    passes: list[Callable[..., Any]]
    constraints: list[Callable[..., Any]]
    _validated: bool = False

    def __init__(
        self,
        passes: list[Callable[..., Any]] | None = None,
        constraints: list[Callable[..., Any]] | None = None,
    ) -> None:
        # [FUNCTION DOC] Store the given passes and constraints (default empty
        # lists; the given lists are stored, not copied).
        self.passes = passes or []
        self.constraints = constraints or []

    @classmethod
    def build_from_passlist(cls, passes: list[Callable[..., Any]]) -> "PassManager":
        # [FUNCTION DOC] Class-level factory: create a PassManager from a list of
        # passes with no constraints and return it.
        pm = PassManager(passes)
        # TODO(alexbeloi): add constraint management/validation
        return pm

    def add_pass(self, _pass: Callable[..., Any]) -> None:
        # [FUNCTION DOC] Append a pass and mark the schedule unvalidated.
        self.passes.append(_pass)
        self._validated = False

    def add_constraint(self, constraint: Callable[..., Any]) -> None:
        # [FUNCTION DOC] Append a constraint and mark the schedule unvalidated.
        self.constraints.append(constraint)
        self._validated = False

    def remove_pass(self, _passes: list[str]) -> None:
        # [FUNCTION DOC] Remove all passes whose function name is in the given list
        # of names (does nothing if None). Replaces the passes list and marks the
        # schedule unvalidated. Passes lacking a name attribute raise AttributeError.
        if _passes is None:
            return
        passes_left = [ps for ps in self.passes if ps.__name__ not in _passes]
        self.passes = passes_left
        self._validated = False

    def replace_pass(
        self, _target: Callable[..., Any], _replacement: Callable[..., Any]
    ) -> None:
        # [FUNCTION DOC] Replace every pass whose name equals the target's name with
        # the replacement, keeping order. Replaces the passes list and marks the
        # schedule unvalidated.
        passes_left = []
        for ps in self.passes:
            if ps.__name__ == _target.__name__:
                passes_left.append(_replacement)
            else:
                passes_left.append(ps)
        self.passes = passes_left
        self._validated = False

    def validate(self) -> None:
        """
        Validates that current pass schedule defined by `self.passes` is valid
        according to all constraints in `self.constraints`
        """
        # [FUNCTION DOC] Check the pass order against all constraints unless already
        # validated; raises RuntimeError on violation; sets the validated flag.
        if self._validated:
            return
        for constraint in self.constraints:
            _validate_pass_schedule_constraint(constraint, self.passes)
        self._validated = True

    def __call__(self, source: Any) -> Any:
        # [FUNCTION DOC] Validate constraints, then apply each pass in order to the
        # running object and return the final result. Raises RuntimeError on a
        # constraint violation; passes may raise anything.
        self.validate()
        out = source
        for _pass in self.passes:
            out = _pass(out)
        return out
