# mypy: allow-untyped-defs
# [MODULE DOC] Defines BaseDataScheduler, a learning-rate-scheduler-like base class that varies one
# hyperparameter of a BaseDataSparsifier over time. Subclasses implement get_schedule_param. No
# import-time side effects.
import abc
import warnings
import weakref
from functools import wraps

from torch.ao.pruning._experimental.data_sparsifier import BaseDataSparsifier


__all__ = ["BaseDataScheduler"]


class BaseDataScheduler:
    # [CLASS DOC] Base scheduler for a BaseDataSparsifier. State: data_sparsifier, schedule_param (name of
    # the config key being scheduled), base_param (initial value per data group), last_epoch, step counter,
    # verbose flag, and the last applied values. On construction it wraps the sparsifier step method with
    # a call counter (to warn if scheduler step is called first) and performs one initial step.
    # Subclasses must override get_schedule_param (not enforced, the class does not derive from ABC).
    r"""
    The BaseDataScheduler is the abstract scheduler class specifically for the
    BaseDataSparsifier class. This class controls a specific hyperparameter of
    the sparsifier class and varies it across the training process (or across time).

    Args:
        data_sparsifier (instance of BaseDataSparsifier)
            Implemented class data sparsifier class wherein the update_mask is implemented
        schedule_param (str)
            A specific hyperparameter of the passed sparsifier that needs to be scheduled/varied
        last_epoch (int, default=-1)
            This is specifically passed when training needs to be resumed from a particular
            point.
        verbose (bool, default=False)
            Verbosity of the BaseDataScheduler

    The *get_hyperparam()* function needs to be implemented by the user.
    """

    def __init__(
        self, data_sparsifier, schedule_param: str, last_epoch=-1, verbose=False
    ):
        # [FUNCTION DOC] Attach to a data sparsifier and schedule the config key schedule_param.
        # Raises TypeError if data_sparsifier is not a BaseDataSparsifier. Records base values per group,
        # replaces the sparsifier step method with a counting wrapper (once), sets its step count to 0,
        # then calls step once, which applies the initial schedule (and increments last_epoch from its
        # starting value, default -1).
        # Attach sparsifier
        if not isinstance(data_sparsifier, BaseDataSparsifier):
            raise TypeError(
                f"{type(data_sparsifier).__name__} is not an instance of torch.ao.pruning.BaseDataSparsifier"
            )
        self.data_sparsifier = data_sparsifier
        self.schedule_param = schedule_param

        # Initialize epoch and base hyper-params
        self.base_param = {
            name: config.get(schedule_param, None)
            for name, config in self.data_sparsifier.data_groups.items()
        }

        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `scheduler.step()` is called after
        # `sparsifier.step()`
        def with_counter(method):
        # [FUNCTION DOC] HELPER: Wrap a bound sparsifier step method so each call increments the instance step
        # count. Returns it unchanged if already wrapped. Keeps only a weak reference to the instance.
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
            # [FUNCTION DOC] HELPER: Increment the sparsifier step count then call the original step method.
                instance = instance_ref()
                instance._step_count += 1  # type: ignore[union-attr]
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True  # type: ignore[attr-defined]
            return wrapper

        self.data_sparsifier.step = with_counter(self.data_sparsifier.step)  # type: ignore[assignment]
        self.data_sparsifier._step_count = 0  # type: ignore[attr-defined]
        self._step_count: int = 0
        self.verbose = verbose

        # Housekeeping
        self._get_sp_called_within_step: bool = False  # sp -> schedule parameter
        self.step()

    @abc.abstractmethod
    def get_schedule_param(self):
        # [FUNCTION DOC] Abstract: return a mapping from data group name to the new value of the scheduled
        # parameter. Called by step; the base implementation raises NotImplementedError.
        r"""
        Abstract method that needs to be implemented by the child class.
        The expected return type should be a dictionary of name to schedule_param value
        The returned values will be updated in sparsifier when the scheduler step() function
        is called.

        Example:
            >>> def get_schedule_param(self):
            ...     new_param = {}
            ...     for name in self.sparsifier.data_groups.keys():
            ...         new_param[name] = (
            ...             self.sparsifier.data_groups[name][self.schedule_param] * 0.5
            ...         )
            ...     return new_param

        When the step() function is called, the value in self.sparsifier.data_groups[name][self.schedule_param]
        would be halved
        """
        raise NotImplementedError

    def __repr__(self):
        # [FUNCTION DOC] Return a string with the class name, the sparsifier and the base parameter values.
        format_string = self.__class__.__name__ + " ("
        format_string += "\n"
        format_string += f"Data Sparsifier {self.data_sparsifier}\n"
        format_string += f"    {self.schedule_param}: {self.base_param}\n"
        format_string += ")"
        return format_string

    def state_dict(self):
        # [FUNCTION DOC] Return a mapping of all instance attributes except the sparsifier (which is not saved).
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the sparsifier.

        Note:
            The scheduler class does not track the state of the data_sparsifier.
            Make sure to store the state of the sparsifier before storing the
            state of the scheduler
        """
        return {
            key: value
            for key, value in self.__dict__.items()
            if key != "data_sparsifier"
        }

    def load_state_dict(self, state_dict):
        # [FUNCTION DOC] Restore scheduler attributes by updating the instance attribute table from the given mapping.
        """Loads the schedulers state.

        Note:
            Remember to restore the state of the data_sparsifier before the scheduler.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_param(self):
        # [FUNCTION DOC] Return the parameter values applied in the last step (mapping group name -> value).
        # Raises AttributeError if step has never completed.
        return self._last_param

    def step(self):
        # [FUNCTION DOC] Advance the schedule one step. On the first call after construction, warns (UserWarning)
        # if the sparsifier step was overridden or has not been called yet. Increments counters and last_epoch,
        # calls get_schedule_param (with an internal flag set), writes each returned value into the sparsifier
        # data group config (printing a message if verbose), records the applied values, and sets the
        # sparsifier enable_mask_update flag to True.
        # Raise warning if trying to call scheduler step before the sparsifier.
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.data_sparsifier.step, "_with_counter"):
                warnings.warn(
                    "Seems like `data_sparsifier.step()` has been overridden after sparsity scheduler "
                    "initialization. Please, make sure to call `data_sparsifier.step()` before "
                    "`scheduler.step()`.",
                    UserWarning,
                    stacklevel=2,
                )

            # Just check if there were two first scheduler.step() calls before sparsifier.step()
            elif self.data_sparsifier._step_count < 1:  # type: ignore[attr-defined]
                warnings.warn(
                    "Detected call of `scheduler.step()` before `data_sparsifier.step()`. "
                    "You have to make sure you run the data_sparsifier.step() BEFORE any "
                    "calls to the scheduler.step().",
                    UserWarning,
                    stacklevel=2,
                )
        self._step_count += 1

        class _enable_get_sp_call:
        # [CLASS DOC] HELPER: Context manager that sets the scheduler flag marking that the schedule parameter is
        # being computed inside step, and clears it on exit.
            def __init__(self, o):
            # [FUNCTION DOC] HELPER: Remember the scheduler object.
                self.o = o

            def __enter__(self):
            # [FUNCTION DOC] HELPER: Set the flag on the scheduler and return the manager.
                self.o._get_sp_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
            # [FUNCTION DOC] HELPER: Clear the flag on the scheduler; exceptions are not suppressed.
                self.o._get_sp_called_within_step = False

        with _enable_get_sp_call(self):
            self.last_epoch += 1
            updated_scheduler_params = self.get_schedule_param()

        for name, param in updated_scheduler_params.items():
            self.data_sparsifier.data_groups[name][self.schedule_param] = param
            if self.verbose:
                print(f"Adjusting {self.schedule_param} for group {name} to {param}")

        self._last_param = {
            name: config.get(self.schedule_param, None)
            for name, config in self.data_sparsifier.data_groups.items()
        }
        self.data_sparsifier.enable_mask_update = True
