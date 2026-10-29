# mypy: allow-untyped-defs

# [MODULE DOC]
# Helpers for registering an out-of-tree backend on the PrivateUse1 dispatch key from Python.
# Public API: rename_privateuse1_backend (give the backend a device name) and
# generate_methods_for_privateuse1_backend (monkey patch Tensor, Module, PackedSequence and
# storage classes with is_<name> properties and <name>() conversion methods).
# Private helpers cover registration-once checks, device normalization, a dummy Python backend
# module/hook/device guard, and a lookup of functions in the custom backend module.
# Import-time side effect: defines the module-level global _privateuse1_backend_name.

import torch
from torch._C import _get_privateuse1_backend_name, _rename_privateuse1_backend
from torch.overrides import handle_torch_function, has_torch_function_unary


__all__ = [
    "rename_privateuse1_backend",
    "generate_methods_for_privateuse1_backend",
]

# TODO: Should use `torch._C._get_privateuse1_backend_name()` to get
# renamed-backend name for `privateuse1`, but the func will cause an
# error with torch.jit.script, so we use the global variable named
# `_privateuse1_backend_name`.
_privateuse1_backend_name = "privateuseone"


def _rename_profiler_activity(backend_name: str) -> None:
    # [FUNCTION DOC] HELPER: Adds an upper-cased alias for the PrivateUse1 member on the
    # profiler activity enum and overrides its repr, str and name so that the member shows the
    # new backend name. Mutates the enum class globally (monkey patching). Takes the new backend
    # name and returns nothing.
    """Mirror the privateuse1 rename in ProfilerActivity so users can write e.g.
    ProfilerActivity.<backend_name> instead of ProfilerActivity.PrivateUse1."""
    from torch._C._profiler import ProfilerActivity

    alias = backend_name.upper()
    setattr(ProfilerActivity, alias, ProfilerActivity.PrivateUse1)

    pu1 = ProfilerActivity.PrivateUse1
    original_repr = ProfilerActivity.__repr__
    original_str = ProfilerActivity.__str__

    def custom_repr(self):
        # [FUNCTION DOC] HELPER: Repr that shows the renamed alias for the PrivateUse1 member
        # and defers to the original repr for every other member.
        if self == pu1:
            return f"<ProfilerActivity.{alias}: {pu1.value}>"
        return original_repr(self)

    def custom_str(self):
        # [FUNCTION DOC] HELPER: Str that shows the renamed alias for the PrivateUse1 member
        # and defers to the original str for every other member.
        if self == pu1:
            return f"ProfilerActivity.{alias}"
        return original_str(self)

    ProfilerActivity.__repr__ = custom_repr
    ProfilerActivity.__str__ = custom_str
    ProfilerActivity.name = property(  # type: ignore[assignment]
        lambda self: alias if self == pu1 else original_str(self).split(".")[-1]
    )


def _rename_device_type(backend_name: str) -> None:
    # [FUNCTION DOC] HELPER: Same as the profiler activity rename but for the autograd
    # DeviceType enum: adds an upper-cased alias for the PrivateUse1 member and overrides repr,
    # str and name for it. Mutates the enum class globally; returns nothing.
    from torch._C._autograd import DeviceType

    alias = backend_name.upper()
    setattr(DeviceType, alias, DeviceType.PrivateUse1)

    pu1 = DeviceType.PrivateUse1
    original_repr = DeviceType.__repr__
    original_str = DeviceType.__str__

    def custom_repr(self):
        # [FUNCTION DOC] HELPER: Repr showing the renamed alias for the PrivateUse1 member;
        # other members use the original repr.
        if self == pu1:
            return f"<DeviceType.{alias}: {pu1.value}>"
        return original_repr(self)

    def custom_str(self):
        # [FUNCTION DOC] HELPER: Str showing the renamed alias for the PrivateUse1 member;
        # other members use the original str.
        if self == pu1:
            return f"DeviceType.{alias}"
        return original_str(self)

    DeviceType.__repr__ = custom_repr
    DeviceType.__str__ = custom_str
    DeviceType.name = property(  # type: ignore[assignment]
        lambda self: alias if self == pu1 else original_str(self).split(".")[-1]
    )


def rename_privateuse1_backend(backend_name: str) -> None:
    # [FUNCTION DOC] Renames the PrivateUse1 backend to the given name so it can be used as an
    # ordinary device string. Calls the C++ rename (which errors if a rename already happened),
    # stores the name in the module-level global, then renames the profiler activity and the
    # autograd device type enum members. Side effects: global state and enum classes are mutated.
    r"""
    Rename the privateuse1 backend device to make it more convenient to use as a device name within PyTorch APIs.

    The steps are:

    (1) (In C++) implement kernels for various torch operations, and register them
        to the PrivateUse1 dispatch key.
    (2) (In python) call torch.utils.rename_privateuse1_backend("foo")

    You can now use "foo" as an ordinary device string in python.

    Note: this API can only be called once per process. Attempting to change
    the external backend after it's already been set will result in an error.

    Note(AMP): If you want to support AMP on your device, you can register a custom backend module.
    The backend must register a custom backend module with ``torch._register_device_module("foo", BackendModule)``.
    BackendModule needs to have the following API's:

    (1) ``get_amp_supported_dtype() -> List[torch.dtype]``
        get the supported dtypes on your "foo" device in AMP, maybe the "foo" device supports one more dtype.

    Note(random): If you want to support to set seed for your device, BackendModule needs to have the following API's:

    (1) ``_is_in_bad_fork() -> bool``
        Return ``True`` if now it is in bad_fork, else return ``False``.

    (2) ``manual_seed_all(seed int) -> None``
        Sets the seed for generating random numbers for your devices.

    (3) ``device_count() -> int``
        Returns the number of "foo"s available.

    (4) ``get_rng_state(device: Union[int, str, torch.device] = 'foo') -> Tensor``
        Returns a list of ByteTensor representing the random number states of all devices.

    (5) ``set_rng_state(new_state: Tensor, device: Union[int, str, torch.device] = 'foo') -> None``
        Sets the random number generator state of the specified "foo" device.

    And there are some common funcs:

    (1) ``is_available() -> bool``
        Returns a bool indicating if "foo" is currently available.

    (2) ``current_device() -> int``
        Returns the index of a currently selected device.

    For more details, see https://pytorch.org/tutorials/advanced/extend_dispatcher.html#get-a-dispatch-key-for-your-backend
    For an existing example, see https://github.com/bdhirsh/pytorch_open_registration_example

    Example::

        >>> # xdoctest: +SKIP("failing")
        >>> torch.utils.rename_privateuse1_backend("foo")
        # This will work, assuming that you've implemented the right C++ kernels
        # to implement torch.ones.
        >>> a = torch.ones(2, device="foo")

    """
    _rename_privateuse1_backend(backend_name)
    global _privateuse1_backend_name
    _privateuse1_backend_name = backend_name
    _rename_profiler_activity(backend_name)
    _rename_device_type(backend_name)


def _check_register_once(module, attr) -> None:
    # [FUNCTION DOC] HELPER: Raises a runtime error if the given object already has the named
    # attribute, guarding against registering a custom backend method twice. Returns nothing.
    if hasattr(module, attr):
        raise RuntimeError(
            f"The custom device module of {module} has already been registered with {attr}"
        )


def _normalization_device(
    custom_backend_name: str, device: int | str | torch.device | None = None
) -> int:
    # [FUNCTION DOC] HELPER: Converts a device given as nothing, string, torch.device or integer
    # into an integer device index for the custom backend. A missing device or a device without
    # index maps to the backend's current device. Raises a runtime error if a device object or
    # string has a type different from the custom backend name. An integer is returned as is.
    def _get_current_device_index():
        # [FUNCTION DOC] HELPER: Returns the current device index from the backend module in
        # torch if it defines current_device, otherwise the default index zero.
        _get_device_index = "current_device"
        if hasattr(torch, custom_backend_name) and hasattr(
            getattr(torch, custom_backend_name), _get_device_index
        ):
            return getattr(getattr(torch, custom_backend_name), _get_device_index)()
        else:
            # The default device index is 0.
            return 0

    if device is None:
        return _get_current_device_index()
    # if isinstance(device, str), this means that the parameter passed in is in the string format "foo:0"
    # convert str object to torch.device object, and then process it uniformly
    elif isinstance(device, str):
        device = torch.device(device)

    # variable device can only be torch.device type or int type
    if isinstance(device, torch.device):
        if device.type != custom_backend_name:
            raise RuntimeError(f"Invalid device, must be {custom_backend_name} device")
        elif device.index is None:
            device_idx = _get_current_device_index()
        else:
            device_idx = device.index
    # if isinstance(device, int), we can take the index number directly
    else:
        device_idx = device
    return device_idx


def _generate_tensor_methods_for_privateuse1_backend(custom_backend_name: str) -> None:
    # [FUNCTION DOC] HELPER: Attaches two members to torch.Tensor: an is_<name> property that
    # tells whether the tensor lives on the custom backend, and a <name>() method that moves the
    # tensor to the backend device. Raises a runtime error if either attribute already exists.
    @property  # type: ignore[misc]
    def wrap_tensor_backend(self: torch.Tensor) -> bool:
        # [FUNCTION DOC] HELPER: Property body; true when the tensor's device type equals the
        # custom backend name. Dispatches through the torch function protocol for subclasses.
        if has_torch_function_unary(self):
            # TODO mypy doesn't support @property, see: https://github.com/python/mypy/issues/6185
            return handle_torch_function(wrap_tensor_backend.__get__, (self,), self)  # type: ignore[attr-defined]
        return self.device.type == custom_backend_name

    _check_register_once(torch.Tensor, f"is_{custom_backend_name}")
    wrap_tensor_backend.fget.__name__ = f"is_{custom_backend_name}"  # type: ignore[attr-defined]
    setattr(torch.Tensor, f"is_{custom_backend_name}", wrap_tensor_backend)

    def wrap_tensor_to(
        self: torch.Tensor,
        device: int | torch.device | None = None,
        non_blocking=False,
        **kwargs,
    ) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Generated <name>() method on Tensor. Normalizes the optional
        # device to an index on the custom backend and calls the regular tensor conversion with
        # the extra keyword arguments. Returns the converted tensor (self if already there).
        # KNOWN ISSUE: the torch function protocol path passes non_blocking as false instead of
        # forwarding the caller's non_blocking value.
        r"""Perform Tensor device conversion. Call the to operator implementation.

        .. note::
            If the ``self`` Tensor already
            has the correct :class:`torch.device`, then ``self`` is returned.
            Otherwise, the returned tensor is a copy of ``self`` with the desired :class:`torch.device`.

        Args:
            device (int, optional): if specified, all parameters will be copied to that device
            non_blocking (bool): If ``True`` and the source is in pinned memory,
                the copy will be asynchronous with respect to the host. Otherwise,
                the argument has no effect.
            **kwargs (dict): For compatibility, may contain the key ``memory_format`` argument.
        """
        if has_torch_function_unary(self):
            return handle_torch_function(
                wrap_tensor_to,
                (self,),
                self,
                device=device,
                non_blocking=False,
                **kwargs,
            )
        device_idx = _normalization_device(custom_backend_name, device)
        return self.to(
            device=torch.device(f"{custom_backend_name}:{device_idx}"),
            non_blocking=non_blocking,
            **kwargs,
        )

    _check_register_once(torch.Tensor, custom_backend_name)
    wrap_tensor_to.__name__ = custom_backend_name
    setattr(torch.Tensor, custom_backend_name, wrap_tensor_to)


def _generate_module_methods_for_privateuse1_backend(custom_backend_name: str) -> None:
    # [FUNCTION DOC] HELPER: Attaches a <name>() method to torch.nn.Module that moves all
    # parameters and buffers to the custom device. Raises a runtime error if Tensor lacks the
    # corresponding method or if Module already has the attribute.
    # Generate Module attributes and methods depends on Tensor methods,
    # so we need to check whether Tensor methods is already registered.
    if not hasattr(torch.Tensor, custom_backend_name):
        raise RuntimeError(
            f"Can not automatically generate {custom_backend_name}() method for torch.nn.Module."
            f"Because torch.Tensor doesn't have the method {custom_backend_name}()."
            f"For this error, you can try setting for_tensor=True."
        )

    def wrap_module_to(
        # pyrefly: ignore [invalid-type-var]
        self: torch.nn.modules.module.T,
        device: int | torch.device | None = None,
    ) -> torch.nn.modules.module.T:  # pyrefly: ignore [invalid-type-var]
        # [FUNCTION DOC] HELPER: Generated Module method; applies the tensor-level backend
        # conversion to every parameter and buffer in place and returns the module.
        r"""Move all model parameters and buffers to the custom device.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on device while being optimized.

        .. note::
            This method modifies the module in-place.

        Args:
            device (int, optional): if specified, all parameters will be copied to that device
        """
        # pyrefly: ignore [missing-attribute]
        return self._apply(lambda t: getattr(t, custom_backend_name)(device))

    _check_register_once(torch.nn.Module, custom_backend_name)
    setattr(torch.nn.Module, custom_backend_name, wrap_module_to)


def _generate_packed_sequence_methods_for_privateuse1_backend(
    custom_backend_name: str,
) -> None:
    # [FUNCTION DOC] HELPER: Attaches an is_<name> property and a <name>() method to
    # PackedSequence. Raises a runtime error if Tensor lacks the matching members or if
    # PackedSequence already has the attributes.
    # Generate PackedSequence Module attributes and methods depends on Tensor methods,
    # so we need to check whether Tensor methods is already registered.
    if not hasattr(torch.Tensor, f"is_{custom_backend_name}") or not hasattr(
        torch.Tensor, custom_backend_name
    ):
        raise RuntimeError(
            f"Can not automatically generate is_{custom_backend_name}() or "
            f"{custom_backend_name}() method for torch.nn.utils.rnn.PackedSequence."
            f"Because torch.Tensor doesn't have the method is_{custom_backend_name}()"
            f"or {custom_backend_name}()."
            f"For this error, you can try setting for_tensor=True."
        )

    @property  # type: ignore[misc]
    def wrap_tensor_backend(self: torch.nn.utils.rnn.PackedSequence) -> bool:
        # [FUNCTION DOC] HELPER: Property body; true when the packed data tensor is on the
        # custom backend.
        return self.data.device.type == custom_backend_name

    _check_register_once(torch.nn.utils.rnn.PackedSequence, f"is_{custom_backend_name}")
    setattr(
        torch.nn.utils.rnn.PackedSequence,
        f"is_{custom_backend_name}",
        wrap_tensor_backend,
    )

    def wrap_module_to(
        self: torch.nn.utils.rnn.PackedSequence, *args, **kwargs
    ) -> torch.nn.utils.rnn.PackedSequence:
        # [FUNCTION DOC] HELPER: Generated PackedSequence method. Probes with an empty tensor
        # whether the given arguments already target the custom backend; if so forwards them to
        # the regular conversion, otherwise forces the device keyword to the backend name first.
        # Mutates the caller's keyword mapping. Returns the converted packed sequence.
        r"""Move all model parameters and buffers to the custom device.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on device while being optimized.

        .. note::
            This method modifies the module in-place.

        Args:
            device (int, optional): if specified, all parameters will be copied to that device
        """
        ex = torch.tensor((), dtype=self.data.dtype, device=self.data.device).to(
            *args,
            **kwargs,
        )
        if ex.device.type == custom_backend_name:
            return self.to(*args, **kwargs)
        kwargs.update({"device": custom_backend_name})

        return self.to(*args, **kwargs)

    _check_register_once(torch.nn.utils.rnn.PackedSequence, custom_backend_name)
    setattr(torch.nn.utils.rnn.PackedSequence, custom_backend_name, wrap_module_to)


def _generate_storage_methods_for_privateuse1_backend(
    custom_backend_name: str, unsupported_dtype: list[torch.dtype] | None = None
) -> None:
    # [FUNCTION DOC] HELPER: Attaches is_<name> properties and <name>() conversion methods to the
    # untyped storage base class and to TypedStorage. The typed conversion raises a runtime error
    # for dtypes listed in unsupported_dtype. Raises a runtime error if any attribute already
    # exists. Side effect: mutates the storage classes.
    # Attribute is registered in the _StorageBase class
    # and UntypedStorage obtains through inheritance.
    @property  # type: ignore[misc]
    def wrap_storage_backend(self: torch.storage._StorageBase) -> bool:
        # [FUNCTION DOC] HELPER: Property body; true when the storage device type equals the
        # custom backend name.
        # KNOWN ISSUE: the docstring says it returns the internal untyped storage, but the code
        # returns a boolean.
        r"""Return the internal :class:`torch.UntypedStorage`."""
        return self.device.type == custom_backend_name

    _check_register_once(torch.storage._StorageBase, f"is_{custom_backend_name}")
    setattr(
        torch.storage._StorageBase, f"is_{custom_backend_name}", wrap_storage_backend
    )

    def wrap_storage_to(self, device=None, non_blocking=False):
        # [FUNCTION DOC] HELPER: Generated storage method. Returns self if already on the target
        # backend device; raises a runtime error for sparse storage; otherwise allocates a new
        # untyped storage on the target device, copies the data into it and returns it.
        r"""Return a copy of this object in custom device memory.

        If this object is already in device memory and on the correct device, then
        no copy is performed and the original object is returned.

        Args:
            device (int): The destination device id. Defaults to the current device.
            non_blocking (bool): If ``True`` and the source is in pinned memory,
            the copy will be asynchronous with respect to the host. Otherwise,
            the argument has no effect.
        """
        # There should be a judgment related to storage device and a judgment related to storage type,
        # but it depends on the extended function, so this part is temporarily omitted in the automatic generation.
        device_idx = _normalization_device(custom_backend_name, device)

        if getattr(self, f"is_{custom_backend_name}"):
            # storage has already on expected device.
            if self.get_device() == device_idx:
                return self
        # For sparse storage, custom need to extend the implementation by themselves.
        if self.is_sparse:
            raise RuntimeError(
                f"Can not support a sparse storage move to {custom_backend_name} backend"
            )
        # create untyped_storage and copy data
        untyped_storage = torch.UntypedStorage(
            self.size(), device=torch.device(f"{custom_backend_name}:{device_idx}")
        )
        untyped_storage.copy_(self, non_blocking)
        return untyped_storage

    _check_register_once(torch.storage._StorageBase, custom_backend_name)
    setattr(torch.storage._StorageBase, custom_backend_name, wrap_storage_to)

    # Register the corresponding attribute for the TypedStorage class.
    # When the TypedStorage class is removed, the registration is also removed.

    @property  # type: ignore[misc]
    def wrap_typed_storage_backend(self: torch.storage.TypedStorage) -> bool:
        # [FUNCTION DOC] HELPER: Property body; emits the typed-storage deprecation warning and
        # tells whether the underlying untyped storage is on the custom backend.
        torch.storage._warn_typed_storage_removal()
        return self._untyped_storage.device.type == custom_backend_name

    _check_register_once(torch.TypedStorage, f"is_{custom_backend_name}")
    setattr(
        torch.storage.TypedStorage,
        f"is_{custom_backend_name}",
        wrap_typed_storage_backend,
    )

    def wrap_typed_storage_to(
        self: torch.storage.TypedStorage, device=None, non_blocking=False, **kwargs
    ) -> torch.storage.TypedStorage:
        # [FUNCTION DOC] HELPER: Generated TypedStorage method. Warns about typed-storage
        # removal, raises a runtime error if the dtype is unsupported, converts the underlying
        # untyped storage via the backend method and wraps the result as a new typed storage.
        torch.storage._warn_typed_storage_removal()
        if unsupported_dtype and self.dtype in unsupported_dtype:
            raise RuntimeError(
                f"Cannot create {custom_backend_name} storage "
                f"as {self.dtype} dtype is not supported by this backend"
            )
        custom_backend_storage: torch.UntypedStorage = getattr(
            self._untyped_storage, custom_backend_name
        )(device, non_blocking, **kwargs)
        return self._new_wrapped_storage(custom_backend_storage)

    _check_register_once(torch.TypedStorage, custom_backend_name)
    setattr(torch.TypedStorage, custom_backend_name, wrap_typed_storage_to)


def generate_methods_for_privateuse1_backend(
    for_tensor: bool = True,
    for_module: bool = True,
    for_packed_sequence: bool = True,
    for_storage: bool = False,
    unsupported_dtype: list[torch.dtype] | None = None,
) -> None:
    # [FUNCTION DOC] Looks up the renamed PrivateUse1 backend name and, depending on the flags,
    # installs the Tensor, Module, storage and PackedSequence members for it. Returns nothing;
    # raises a runtime error if members already exist or prerequisites are missing.
    # KNOWN ISSUE: the docstring does not document for_packed_sequence, which defaults to true.
    r"""
    Automatically generate attributes and methods for the custom backend after rename privateuse1 backend.

    In the default scenario, storage-related methods will not be generated automatically.

    When you implement kernels for various torch operations, and register them to the PrivateUse1 dispatch key.
    And call the function torch.rename_privateuse1_backend("foo") to rename your backend name.
    At this point, you can easily register specific methods and attributes by calling this function.
    Just like torch.Tensor.foo(), torch.Tensor.is_foo, torch.Storage.foo(), torch.Storage.is_foo.

    Note: We recommend you use generic functions (check devices are equal or to(device=)).
    We provide these methods for convenience only and they will be "monkey patched" onto the objects
    and so will not be properly typed. For Storage methods generate, if you need to support sparse data storage,
    you need to extend the implementation yourself.

    Args:
        for_tensor (bool): whether register related methods for torch.Tensor class.
        for_module (bool): whether register related methods for torch.nn.Module class.
        for_storage (bool): whether register related methods for torch.Storage class.
        unsupported_dtype (List[torch.dtype]): takes effect only when the storage method needs to be generated,
            indicating that the storage does not support the torch.dtype type.

    Example::

        >>> # xdoctest: +SKIP("failing")
        >>> torch.utils.rename_privateuse1_backend("foo")
        >>> torch.utils.generate_methods_for_privateuse1_backend()
        # Then automatically generate backend-related attributes and methods.
        >>> a = torch.tensor(2).foo()
        >>> a.is_foo
        >>> hasattr(torch.nn.Module, 'foo')
    """
    custom_backend_name = _get_privateuse1_backend_name()

    if for_tensor:
        _generate_tensor_methods_for_privateuse1_backend(custom_backend_name)

    if for_module:
        _generate_module_methods_for_privateuse1_backend(custom_backend_name)

    if for_storage:
        _generate_storage_methods_for_privateuse1_backend(
            custom_backend_name, unsupported_dtype
        )

    if for_packed_sequence:
        _generate_packed_sequence_methods_for_privateuse1_backend(custom_backend_name)


def _get_custom_mod_func(func_name: str):
    # [FUNCTION DOC] HELPER: Returns the named callable from the torch submodule registered for
    # the renamed PrivateUse1 backend. Raises an assertion error if the name is not a string and
    # a runtime error if the backend module or the function is missing.
    r"""
    Return the func named `func_name` defined in custom device module. If not defined,
    return `None`. And the func is registered with `torch.utils.rename_privateuse1_backend('foo')`
    and `torch._register_device_module('foo', BackendModule)`.
    If the custom device module or the func is not defined, it will give warning or error message.
    Args:
        func_name (str): return the callable func named func_name defined in custom device module.
    Example::
        class DummyfooModule:
            @staticmethod
            def is_available():
                return True
            @staticmethod
            def func_name(*args, **kwargs):
                ....
        torch.utils.rename_privateuse1_backend("foo")
        torch._register_device_module("foo", DummyfooModule)
        foo_is_available_func = torch.utils.backend_registration._get_custom_mod_func("is_available")
        if foo_is_available_func:
            foo_is_available = foo_is_available_func()
        func_ = torch.utils.backend_registration._get_custom_mod_func("func_name")
        if func_:
            result = func_(*args, **kwargs)
    Attention: This function is not meant to be used directly by users, which is why
    it is marked as private. It is a convenience function for backend implementers to
    more easily call the hooks into their backend extensions.
    """
    if not isinstance(func_name, str):
        raise AssertionError(f"func_name must be `str`, but got `{type(func_name)}`.")
    backend_name = _get_privateuse1_backend_name()
    custom_device_mod = getattr(torch, backend_name, None)
    function = getattr(custom_device_mod, func_name, None)
    if custom_device_mod is None or function is None:
        message = f"Try to call torch.{backend_name}.{func_name}. The backend must register a custom backend "
        message += f"module with `torch._register_device_module('{backend_name}', BackendModule)`. And "
        message += f"BackendModule needs to have the following API's:\n `{func_name}(*args, **kwargs)`. \n"
        raise RuntimeError(message)
    return function


class _DummyBackendModule:
    # [CLASS DOC] HELPER: Minimal stand-in backend module for a Python-only PrivateUse1 backend.
    # Stateless; reports one always-available device with index zero and ignores seeding.
    def is_initialized(self) -> bool:
        # [FUNCTION DOC] HELPER: Always reports the backend as initialized.
        return True

    def is_available(self) -> bool:
        # [FUNCTION DOC] HELPER: Always reports the backend as available.
        return True

    def current_device(self) -> int:
        # [FUNCTION DOC] HELPER: Always returns device index zero.
        return 0

    def _is_in_bad_fork(self) -> bool:
        # [FUNCTION DOC] HELPER: Always reports not being in a bad fork.
        return False

    def manual_seed_all(self, seed: int) -> None:
        # [FUNCTION DOC] HELPER: No-op; the seed is ignored.
        pass

    def device_count(self) -> int:
        # [FUNCTION DOC] HELPER: Always reports a single device.
        return 1


class _DummyPrivateUse1Hook(torch._C._acc.PrivateUse1Hooks):
    # [CLASS DOC] HELPER: Default accelerator hooks object for a Python PrivateUse1 backend;
    # reports the backend as built, available and having a primary context.
    def is_available(self) -> bool:
        # [FUNCTION DOC] HELPER: Always true.
        return True

    def has_primary_context(self, dev_id) -> bool:
        # [FUNCTION DOC] HELPER: Always true regardless of the device id.
        return True

    def is_built(self) -> bool:
        # [FUNCTION DOC] HELPER: Always true.
        return True


class _DummyDeviceGuard(torch._C._acc.DeviceGuard):
    # [CLASS DOC] HELPER: Default device guard for a Python PrivateUse1 backend; only reports
    # its device type.
    def type_(self):
        # [FUNCTION DOC] HELPER: Returns the PrivateUse1 device type enum member.
        return torch._C._autograd.DeviceType.PrivateUse1


def _setup_privateuseone_for_python_backend(
    rename=None, backend_module=None, hook=None, device_guard=None
) -> None:
    # [FUNCTION DOC] HELPER: Experimental setup of PrivateUse1 as a Python backend. Optionally
    # renames the backend, generates the default Tensor/Module/PackedSequence methods, fills in
    # dummy backend module, hook and device guard when not supplied, registers the module under
    # the backend name and registers the hook and guard with the C++ accelerator layer. The
    # order of these steps matters. Side effects: global registration state changes.
    """This function will prepare the PrivateUse1 dispatch key to be used as a python backend.

    WARNING: this API is experimental and might change without notice.

    Formally, this registers things that Pytorch expects a registered backend
    in C++ to have: including device guards, hooks, and backend modules and what not.

    after this call, one can use `torch.library` to write Ops for this dispatch key
    and expect it to behave like a backend registered in C++.

    See the unit test at test/test_privateuseone_python_backend.py for more details.

    Args:
        rename: str | None, if passed in, we will rename privateuseone backend to
           the name given.
        backend_module: object | None, if passed in None, we will use DummyBackendModule
        hook: object | None, if passed in None, we will use DummyPrivateUse1Hook
        device_guard: object | None, if passed in None, we will use DummyDeviceGuard
    """
    # NOTE: the ordering of which these functions are called is important.
    if rename is not None:
        torch.utils.rename_privateuse1_backend(rename)
    else:
        rename = "privateuseone"
    torch.utils.generate_methods_for_privateuse1_backend()
    if backend_module is None:
        backend_module = _DummyBackendModule()
    if hook is None:
        hook = _DummyPrivateUse1Hook()
    if device_guard is None:
        device_guard = _DummyDeviceGuard()
    torch._register_device_module(rename, backend_module)  # type: ignore[bad-argument-type]
    torch._C._acc.register_python_privateuseone_hook(hook)
    torch._C._acc.register_python_privateuseone_device_guard(device_guard)
