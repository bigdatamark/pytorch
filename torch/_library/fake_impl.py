# mypy: allow-untyped-defs
# [MODULE DOC] Support for registering "fake" (abstract/meta) implementations of custom
# operators: FakeImplHolder stores and registers fake kernels, a meta kernel wrapper
# guards against ctx use with plain meta tensors, and FakeImplCtx lets a fake impl
# create data-dependent symbolic sizes. Import-time side effect: creates a logger and
# the module-global context getter (defaults to returning None).
import contextlib
import functools
import logging
from collections.abc import Callable
from typing_extensions import deprecated

import torch
from torch._library.utils import Kernel, RegistrationHandle


log = logging.getLogger(__name__)


class FakeImplHolder:
    # [CLASS DOC] Per-operator holder of fake kernels. Attributes: qualname and kernels, a
    # list of Kernel objects in registration order; the newest (last) one is active. The
    # kernel property is read-only. register() adds a kernel and installs a Meta-key
    # wrapper into a library, returning a handle that removes the kernel.
    """A holder where one can register a fake impl to."""

    def __init__(self, qualname: str):
        # [FUNCTION DOC] Stores qualname and initializes an empty kernel list.
        self.qualname: str = qualname
        # kernels stores all registered fake kernels, ordered by registration
        # time ascendingly (newer registration after older registration). If an
        # operator library gets loaded that overrides an existing fake kernel,
        # both kernels will be in the list, but the newest one will be the one
        # that is run. If the library is unloaded, we will remove the kernel
        # from this list.
        self.kernels: list[Kernel] = []

    @property
    def kernel(self):
        # [FUNCTION DOC] Property getter: the most recently registered Kernel, or None.
        if len(self.kernels) == 0:
            return None
        return self.kernels[-1]

    @kernel.setter
    def kernel(self, value):
        # [FUNCTION DOC] Property setter: always raises RuntimeError.
        raise RuntimeError("Unable to directly set kernel.")

    def register(
        self, func: Callable, source: str, lib, *, allow_override=True
    ) -> RegistrationHandle:
        # [FUNCTION DOC] Registers func (with a source description string) as the fake impl.
        # If allow_override is false, raises RuntimeError when a fake kernel, a Meta kernel
        # or a CompositeImplicitAutograd kernel already exists. Appends a Kernel, then
        # registers a wrapped meta kernel on the Meta key via lib.impl; if that fails, the
        # kernel is removed again (logged) and the error re-raised. Returns a handle whose
        # deregistration removes the kernel from the list (it does not unregister the Meta impl).
        """Register a fake impl.

        Returns a RegistrationHandle that one can use to de-register this
        fake impl.
        """

        if not allow_override:
            if self.kernel is not None:
                raise RuntimeError(
                    f"register_fake(...): the operator {self.qualname} "
                    f"already has a fake impl registered at "
                    f"{self.kernel.source}."
                )
            if torch._C._dispatch_has_kernel_for_dispatch_key(self.qualname, "Meta"):
                raise RuntimeError(
                    f"register_fake(...): the operator {self.qualname} "
                    f"already has a DispatchKey::Meta implementation via a "
                    f"pre-existing torch.library or TORCH_LIBRARY registration. "
                    f"Please either remove that registration or don't call "
                    f"register_fake."
                )

            if torch._C._dispatch_has_kernel_for_dispatch_key(
                self.qualname, "CompositeImplicitAutograd"
            ):
                raise RuntimeError(
                    f"register_fake(...): the operator {self.qualname} "
                    f"already has an implementation for this device type via a "
                    f"pre-existing registration to "
                    f"DispatchKey::CompositeImplicitAutograd."
                    f"CompositeImplicitAutograd operators do not need a fake "
                    f"impl; "
                    f"instead, the operator will decompose into its constituents "
                    f"and those "
                    f"can have fake impls defined on them."
                )

        # Store the kernel in this holder
        kernel = Kernel(func, source)
        self.kernels.append(kernel)

        def deregister_fake_kernel():
            # [FUNCTION DOC] HELPER: Removes the registered kernel from the holder's list.
            self.kernels.remove(kernel)

        meta_kernel = construct_meta_kernel(self.qualname, self)
        try:
            lib.impl(self.qualname, meta_kernel, "Meta", allow_override=allow_override)
        except Exception:
            log.info(
                "Failed to register fake_impl '%s':",
                self.qualname,
            )
            self.kernels.remove(kernel)
            raise

        handle = RegistrationHandle(deregister_fake_kernel)
        return handle


def construct_meta_kernel(qualname: str, fake_impl_holder: FakeImplHolder) -> Callable:
    # [FUNCTION DOC] Returns a Meta-key kernel that runs the holder's current fake kernel
    # with the library context getter replaced by one that raises RuntimeError (because
    # meta tensors cannot support data-dependent output shapes). Raises AssertionError if
    # the holder has no kernel, at construction time or at call time.
    if fake_impl_holder.kernel is None:
        raise AssertionError("fake_impl_holder.kernel must not be None")

    @functools.wraps(fake_impl_holder.kernel.func)
    def meta_kernel(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Calls the holder's latest fake kernel under a ctx getter
        # that raises; returns its result. Looks up the kernel at call time.
        if fake_impl_holder.kernel is None:
            raise AssertionError("fake_impl_holder.kernel must not be None")
        source = fake_impl_holder.kernel.source

        def error_on_ctx():
            # [FUNCTION DOC] HELPER: Always raises RuntimeError explaining that get_ctx
            # cannot be used with meta tensors; suggests FakeTensors.
            raise RuntimeError(
                f"{qualname} ({source}): You're trying to run this operator "
                f"with meta Tensors (as opposed to FakeTensors), but this "
                f"operator may return an output Tensor with data-dependent shape. Meta "
                f"Tensors don't support operators with outputs that have data-dependent shapes "
                f"but FakeTensors do. "
                f"If your operator does not return an output with data-dependent shape, "
                f"make sure the FakeTensor and/or meta kernel does not call "
                f"torch.library.get_ctx(). Otherwise, please use FakeTensors."
            )

        with set_ctx_getter(error_on_ctx):
            return fake_impl_holder.kernel(*args, **kwargs)

    return meta_kernel


def get_none():
    # [FUNCTION DOC] HELPER: Default context getter; returns None (no active ctx).
    return None


global_ctx_getter: Callable = get_none


@contextlib.contextmanager
def set_ctx_getter(ctx_getter):
    # [FUNCTION DOC] Context manager: temporarily replaces the module-global ctx getter
    # (used by get_ctx) with ctx_getter and restores the previous one on exit, even on error.
    global global_ctx_getter
    prev = global_ctx_getter
    try:
        global_ctx_getter = ctx_getter
        yield
    finally:
        global_ctx_getter = prev


class FakeImplCtx:
    # [CLASS DOC] Context passed to fake impls (via get_ctx). Holds the active fake mode,
    # its shape environment (may be None) and the op being faked. Provides
    # new_dynamic_size to create unbacked symbolic sizes; create_unbacked_symint is a
    # deprecated alias.
    """
    Context object for writing fake implementations for custom operators.
    """

    def __init__(self, _fake_mode, _op):
        # [FUNCTION DOC] Stores the fake mode, its shape_env and the op.
        self._fake_mode = _fake_mode
        self._shape_env = _fake_mode.shape_env
        self._op = _op

    @deprecated(
        "`create_unbacked_symint` is deprecated, please use `new_dynamic_size` instead",
        category=FutureWarning,
    )
    def create_unbacked_symint(self, *, min=2, max=None) -> torch.SymInt:
        # [FUNCTION DOC] Deprecated alias (emits FutureWarning) for new_dynamic_size.
        # KNOWN ISSUE: default min is 2 here but 0 in new_dynamic_size, so behavior differs
        # from the replacement when called without arguments.
        return self.new_dynamic_size(min=min, max=max)

    def new_dynamic_size(self, *, min=0, max=None) -> torch.SymInt:
        # [FUNCTION DOC] Creates a new unbacked symbolic int in [min, max] for a
        # data-dependent size. Raises DynamicOutputShapeException if there is no shape env
        # or dynamic output shape ops are disallowed; ValueError if min/max are SymInts or
        # min is negative. Side effect: allocates a symbol in the shape env.
        # KNOWN ISSUE: the docstring says the default min is 2, but the signature default is 0.
        """Constructs a new symint (symbolic int) representing a data-dependent value.

        This is useful for writing the fake implementation (which is necessary
        for torch.compile) for a CustomOp where an output Tensor has a size
        that depends on the data of the input Tensors.

        Args:
            min (int): A statically known inclusive lower bound for this symint. Default: 0
            max (Optional[int]): A statically known inclusive upper bound for this
                symint. Default: None

        .. warning:

            It is important that the ``min`` and ``max`` (if not None) values are set
            correctly, otherwise, there will be undefined behavior under
            torch.compile. The default value of ``min`` is 2 due to torch.compile
            specializing on 0/1 sizes.

            You must also verify that your implementation on concrete Tensors
            (e.g. CPU/CUDA) only returns Tensors where the size that corresponds
            to the symint also has respects these constraint.
            The easiest way to do this is to add an assertion in the CPU/CUDA/etc
            implementation that the size follows these bounds.

        Example::

            >>> # An operator with data-dependent output shape
            >>> lib = torch.library.Library("mymodule", "FRAGMENT")
            >>> lib.define("mymodule::custom_nonzero(Tensor x) -> Tensor")
            >>>
            >>> @torch.library.register_fake("mymodule::custom_nonzero")
            >>> def _(x):
            >>>     # Number of nonzero-elements is data-dependent.
            >>>     # Since we cannot peek at the data in a fake impl,
            >>>     # we use the ctx object to construct a new symint that
            >>>     # represents the data-dependent size.
            >>>     ctx = torch.library.get_ctx()
            >>>     nnz = ctx.new_dynamic_size()
            >>>     shape = [nnz, x.dim()]
            >>>     result = x.new_empty(shape, dtype=torch.int64)
            >>>     return result
            >>>
            >>> @torch.library.impl(lib, "custom_nonzero", "CPU")
            >>> def _(x):
            >>>     x_np = x.numpy()
            >>>     res = np.stack(np.nonzero(x_np), axis=1)
            >>>     return torch.tensor(res, device=x.device)

        """
        if (
            self._shape_env is None
            or not self._shape_env.allow_dynamic_output_shape_ops
        ):
            raise torch._subclasses.fake_tensor.DynamicOutputShapeException(self._op)

        if isinstance(min, torch.SymInt) or isinstance(max, torch.SymInt):
            raise ValueError(
                f"ctx.new_dynamic_size(min={min}, max={max}): expected "
                f"min and max to be statically known ints but got SymInt. "
                f"This is not supported."
            )

        if min < 0:
            raise ValueError(
                f"ctx.new_dynamic_size(min={min}, ...): expected min to be "
                f"greater than or equal to 0: this API can only create "
                f"non-negative sizes."
            )

        return allocate_size(self._shape_env, min, max)


def allocate_size(shape_env, min_val=0, max_val=None):
    # [FUNCTION DOC] HELPER: Creates an unbacked symint in the shape env and constrains its
    # range as a size to [min_val, max_val]; returns it. Mutates the shape env.
    result = shape_env.create_unbacked_symint()
    torch.fx.experimental.symbolic_shapes._constrain_range_for_size(
        result, min=min_val, max=max_val
    )
    return result
