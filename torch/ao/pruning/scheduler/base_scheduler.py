# mypy: allow-untyped-defs
# [MODULE DOC] Defines BaseScheduler, the base class for schedulers that change the
# sparsity level of each group of a BaseSparsifier over epochs. Exports BaseScheduler.
# No import-time side effects.

import warnings
import weakref
from functools import wraps

from torch.ao.pruning.sparsifier.base_sparsifier import BaseSparsifier


__all__ = ["BaseScheduler"]


class BaseScheduler:
    # [CLASS DOC] Base scheduler wrapping a sparsifier. State: sparsifier, base_sl (list of
    # each group's initial sparsity level, treated as the target), last_epoch, verbose,
    # step counters and _last_sl (levels from the latest step). Subclasses implement
    # get_sl, returning one new level per group; step advances the epoch and writes the
    # levels into the sparsifier groups.
    def __init__(self, sparsifier, last_epoch=-1, verbose=False):
        # [FUNCTION DOC] Validates that sparsifier is a BaseSparsifier (TypeError otherwise),
        # records each group's sparsity level as base_sl, wraps the sparsifier's step
        # method with a call counter (mutating the sparsifier), initializes counters and
        # flags, and performs an initial step.
        # Attach sparsifier
        if not isinstance(sparsifier, BaseSparsifier):
            raise TypeError(
                f"{type(sparsifier).__name__} is not an instance of torch.ao.pruning.BaseSparsifier"
            )
        self.sparsifier = sparsifier

        # Initialize epoch and base sparsity levels

        self.base_sl = [group["sparsity_level"] for group in sparsifier.groups]
        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `scheduler.step()` is called after
        # `sparsifier.step()`
        def with_counter(method):
            # [FUNCTION DOC] HELPER: Returns method unchanged if it is already counted;
            # otherwise returns a wrapper that holds only a weak reference to the instance,
            # increments the instance's step count on each call, then calls the original.
            if getattr(method, "_with_counter", False):
                # `sparsifier.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the sparsifier instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Bumps the sparsifier's step count, then forwards all
                # arguments to the original step method and returns its result.
                instance = instance_ref()
                instance._step_count += 1  # type: ignore[union-attr]
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True  # type: ignore[attr-defined]
            return wrapper

        self.sparsifier.step = with_counter(self.sparsifier.step)  # type: ignore[assignment]
        self.sparsifier._step_count = 0  # type: ignore[attr-defined]
        self._step_count: int = 0
        self.verbose = verbose

        # Housekeeping
        self._get_sl_called_within_step: bool = False

        self.step()

    def state_dict(self):
        # [FUNCTION DOC] Returns a new mapping of every instance attribute except the sparsifier.
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the sparsifier.
        """
        return {
            key: value for key, value in self.__dict__.items() if key != "sparsifier"
        }

    def load_state_dict(self, state_dict):
        # [FUNCTION DOC] Updates the instance attributes with the entries of state_dict.
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_sl(self):
        # [FUNCTION DOC] Returns the list of sparsity levels computed by the latest step.
        """Return last computed sparsity level by current scheduler."""
        return self._last_sl

    def get_sl(self):
        # [FUNCTION DOC] Abstract hook for subclasses: should return one new sparsity level per
        # group. Warns if called outside of step, then always raises NotImplementedError.
        # Compute sparsity level using chainable form of the scheduler
        # Note: This method is not intended to be called directly, and is only
        #       used by the ".step" method. Use .get_last_sl() instead.
        if not self._get_sl_called_within_step:
            warnings.warn(
                "To get the last sparsity level computed by the scheduler, "
                "please use `get_last_sl()`.",
                stacklevel=2,
            )
        raise NotImplementedError

    def print_sl(self, is_verbose, group, sl, epoch=None):
        # [FUNCTION DOC] When is_verbose is true, prints a message about group's new level sl
        # (including the epoch when given) to standard output. Otherwise does nothing.
        """Display the current sparsity level."""
        if is_verbose:
            if epoch is None:
                print(f"Adjusting sparsity level of group {group} to {sl:.4e}.")
            else:
                print(
                    f"Epoch {epoch:5d}: adjusting sparsity level of group {group} to {sl:.4e}."
                )

    def __repr__(self):
        # [FUNCTION DOC] Returns a multi-line string with the class name, the sparsifier and base_sl.
        format_string = self.__class__.__name__ + " ("
        format_string += "\n"
        format_string += f"Sparsifier {self.sparsifier}\n"
        format_string += f"    base_sl: {self.base_sl}\n"
        format_string += ")"
        return format_string

    def step(self, epoch=None):
        # [FUNCTION DOC] Advances the schedule by one epoch. On the second call overall, warns
        # if the sparsifier step was overridden or has not run yet. Increments the
        # step count and last_epoch, calls get_sl with the in-step flag set, writes each
        # resulting level into the matching sparsifier group (optionally printing), saves
        # the levels as _last_sl, and sets enable_mask_update on the sparsifier to True.
        # The epoch argument is only used for printing.
        # Raise warning if trying to call scheduler step before the sparsifier.
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.sparsifier.step, "_with_counter"):
                warnings.warn(
                    "Seems like `sparsifier.step()` has been overridden after sparsity scheduler "
                    "initialization. Please, make sure to call `sparsifier.step()` before "
                    "`scheduler.step()`.",
                    UserWarning,
                    stacklevel=2,
                )

            # Just check if there were two first scheduler.step() calls before sparsifier.step()
            elif self.sparsifier._step_count < 1:  # type: ignore[attr-defined]
                warnings.warn(
                    "Detected call of `scheduler.step()` before `sparsifier.step()`. "
                    "You have to make sure you run the sparsifier.step() BEFORE any "
                    "calls to the scheduler.step().",
                    UserWarning,
                    stacklevel=2,
                )
        self._step_count += 1

        class _enable_get_sl_call:
            # [CLASS DOC] HELPER: Context manager that sets the scheduler's in-step flag on
            # entry and clears it on exit, so get_sl does not warn.
            def __init__(self, o):
                # [FUNCTION DOC] HELPER: Stores the scheduler o.
                self.o = o

            def __enter__(self):
                # [FUNCTION DOC] HELPER: Sets the in-step flag and returns self.
                self.o._get_sl_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                # [FUNCTION DOC] HELPER: Clears the in-step flag; exceptions propagate.
                self.o._get_sl_called_within_step = False

        with _enable_get_sl_call(self):
            self.last_epoch += 1
            values = self.get_sl()

        for i, data in enumerate(zip(self.sparsifier.groups, values)):
            param_group, sl = data
            param_group["sparsity_level"] = sl
            self.print_sl(self.verbose, i, sl, epoch)

        self._last_sl = [group["sparsity_level"] for group in self.sparsifier.groups]
        self.sparsifier.enable_mask_update = True

    def _make_sure_a_list(self, var):
        # [FUNCTION DOC] HELPER: Returns a list with one value per sparsifier group. A scalar is
        # repeated; a list or tuple is copied into a list. Raises ValueError if its length
        # differs from the number of groups.
        r"""Utility that extends it to the same length as the .groups, ensuring it is a list"""
        n = len(self.sparsifier.groups)
        if not isinstance(var, (list, tuple)):
            return [var] * n
        else:
            if len(var) != n:
                raise ValueError(f"Expected variable of length {n}, but got {len(var)}")
            return list(var)  # We want the result to be in a list, not tuple
