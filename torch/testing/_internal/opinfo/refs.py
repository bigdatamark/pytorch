# mypy: ignore-errors
# [MODULE DOC] OpInfo subclasses describing Python references (implementations of existing
# operations in terms of torch._prims). Each reference OpInfo finds an existing OpInfo by
# name and variant and reuses that OpInfo's original constructor arguments, with overrides.
# Contains the helper functions that find the parent and merge arguments, and four classes:
# PythonRefInfo, ReductionPythonRefInfo, ElementwiseUnaryPythonRefInfo and
# ElementwiseBinaryPythonRefInfo. No import-time side effects.

from torch.testing._internal.opinfo.core import (
    BinaryUfuncInfo,
    OpInfo,
    ReductionOpInfo,
    UnaryUfuncInfo,
)


# NOTE [Python References]
# Python References emulate existing PyTorch operations, but can ultimately
#   be expressed in terms of "primitive" operations from torch._prims.
#
# These references are experimental.
# See https://dev-discuss.pytorch.org/t/tracing-with-primitives-update-0/577
#   for additional context.
#
# Python Reference OpInfos should be added to the python_ref_db list below.
#   Tests can opt-into running on these references by including
#   that list in the Sequence they pass to the @ops decorator.
#
# When a Python Reference OpInfo is constructed a pointer to an
#   existing OpInfo must be provided using the torch_opinfo_name kwarg.
#   The existing OpInfo with that name and no variant will be found
#   to inherit from.
#
# Instead of just inheriting the existing OpInfo's metadata, the
#   Python Reference OpInfos inherit the existing OpInfo's
#   construction arguments. These arguments can be overridden
#   by adding kwargs to the constructor.


def _find_referenced_opinfo(referenced_name, variant_name, *, op_db=None):
    # [FUNCTION DOC] HELPER: Searches the given op_db (default: the global op_db, imported
    # lazily) for the OpInfo with the given name and variant test name and returns it.
    # Returns None if no entry matches.
    """
    Finds the OpInfo with the given name that has no variant name.
    """
    # NOTE: searching the global op_db doesn't work when OpInfos are split into
    # different modules, as otherwise the op_db will not be fully constructed
    # yet. So, instead the local op_db must be passed in explicitly.
    if op_db is None:
        from torch.testing._internal.common_methods_invocations import op_db

    for opinfo in op_db:
        if opinfo.name == referenced_name and opinfo.variant_test_name == variant_name:
            return opinfo


def _inherit_constructor_args(name, op, inherited, overrides):
    # [FUNCTION DOC] HELPER: Builds the constructor keyword mapping for a reference OpInfo.
    # Copies the inherited arguments, flattens a nested kwargs entry, drops self, class,
    # skips and decorators entries, then applies fixed values (name, op, no aliases, no
    # method or inplace variants, scripting unsupported), then the caller's overrides, and
    # finally forces all autograd related support flags to false. Returns the mapping; the
    # inputs are not mutated.
    # inherits metadata
    common_kwargs = {
        "name": name,
        "op": op,
        "aliases": None,  # TODO add a check for alias coverage
        "method_variant": None,
        "inplace_variant": None,  # TODO: add a check for inplace coverage
        "supports_scripting": False,
    }

    # Acquires inherited kwargs
    kwargs = inherited.copy()

    # Fixes metadata
    if "kwargs" in kwargs:
        kwargs.update(kwargs["kwargs"])
        del kwargs["kwargs"]
    if "self" in kwargs:
        del kwargs["self"]
    if "__class__" in kwargs:
        del kwargs["__class__"]
    if "skips" in kwargs:
        del kwargs["skips"]
    if "decorators" in kwargs:
        del kwargs["decorators"]

    # Overrides metadata
    kwargs.update(common_kwargs)
    kwargs.update(overrides)

    # At the moment no prims support autograd, so we must not run autograd
    # tests e.g. when testing dtype support.  Once we start writing autograd
    # formulas for prims this can be removed.
    kwargs["supports_autograd"] = False
    kwargs["supports_gradgrad"] = False
    kwargs["supports_fwgrad_bwgrad"] = False
    kwargs["supports_inplace_autograd"] = False
    kwargs["supports_forward_ad"] = False

    return kwargs


class PythonRefInfo(OpInfo):
    # [CLASS DOC] OpInfo for a Python reference of a generic operation. Attributes:
    # torch_opinfo_name, torch_opinfo_variant_name, torch_opinfo (the parent OpInfo found
    # in the database) and validate_view_consistency. Remaining state is inherited from the
    # parent's original constructor arguments.
    """
    An OpInfo for a Python reference of an OpInfo base class operation.
    """

    def __init__(
        self,
        name,  # the stringname of the callable Python reference
        *,
        op=None,  # the function variant of the operation, populated as torch.<name> if None
        op_db=None,  # The database of opinfos to search for the parent opinfo
        torch_opinfo_name,  # the string name of the corresponding torch opinfo
        torch_opinfo_variant_name="",  # the variant name for corresponding torch opinfo
        validate_view_consistency=True,
        **kwargs,
    ):  # additional kwargs override kwargs inherited from the torch opinfo
        # [FUNCTION DOC] Looks up the parent OpInfo by name and variant in op_db, records
        # the lookup keys and the view consistency flag, merges the parent's original
        # arguments with kwargs overrides, and initializes the base OpInfo. Raises
        # AssertionError if the parent is not found or is not an OpInfo.
        self.torch_opinfo_name = torch_opinfo_name
        self.torch_opinfo_variant_name = torch_opinfo_variant_name
        self.torch_opinfo = _find_referenced_opinfo(
            torch_opinfo_name, torch_opinfo_variant_name, op_db=op_db
        )
        self.validate_view_consistency = validate_view_consistency
        if not isinstance(self.torch_opinfo, OpInfo):
            raise AssertionError(
                f"Expected torch_opinfo to be OpInfo, got {type(self.torch_opinfo)}"
            )

        inherited = self.torch_opinfo._original_opinfo_args
        ukwargs = _inherit_constructor_args(name, op, inherited, kwargs)
        super().__init__(**ukwargs)


class ReductionPythonRefInfo(ReductionOpInfo):
    # [CLASS DOC] OpInfo for a Python reference of a reduction operation. Same attributes
    # as PythonRefInfo, but view consistency validation is always disabled (see the linked
    # issue) and the parent must be a ReductionOpInfo.
    # KNOWN ISSUE: the docstring below says elementwise unary operation, but this class is
    # for reductions.
    """
    An OpInfo for a Python reference of an elementwise unary operation.
    """

    def __init__(
        self,
        name,  # the stringname of the callable Python reference
        *,
        op=None,  # the function variant of the operation, populated as torch.<name> if None
        op_db=None,  # The database of opinfos to search for the parent opinfo
        torch_opinfo_name,  # the string name of the corresponding torch opinfo
        torch_opinfo_variant_name="",  # the variant name for corresponding torch opinfo
        **kwargs,
    ):  # additional kwargs override kwargs inherited from the torch opinfo
        # [FUNCTION DOC] Finds the parent ReductionOpInfo, merges its original reduction
        # arguments with the kwargs overrides, sets validate_view_consistency to false and
        # initializes the base class. Raises AssertionError if the parent is missing or is
        # not a ReductionOpInfo.
        self.torch_opinfo_name = torch_opinfo_name
        self.torch_opinfo_variant_name = torch_opinfo_variant_name
        self.torch_opinfo = _find_referenced_opinfo(
            torch_opinfo_name, torch_opinfo_variant_name, op_db=op_db
        )
        if not isinstance(self.torch_opinfo, ReductionOpInfo):
            raise AssertionError(
                f"Expected torch_opinfo to be ReductionOpInfo, got {type(self.torch_opinfo)}"
            )

        inherited = self.torch_opinfo._original_reduction_args
        ukwargs = _inherit_constructor_args(name, op, inherited, kwargs)

        # See https://github.com/pytorch/pytorch/issues/77216
        self.validate_view_consistency = False

        super().__init__(**ukwargs)


class ElementwiseUnaryPythonRefInfo(UnaryUfuncInfo):
    # [CLASS DOC] OpInfo for a Python reference of an elementwise unary operation. Same
    # attributes as PythonRefInfo; the parent must be a UnaryUfuncInfo.
    """
    An OpInfo for a Python reference of an elementwise unary operation.
    """

    def __init__(
        self,
        name,  # the stringname of the callable Python reference
        *,
        op=None,  # the function variant of the operation, populated as torch.<name> if None
        op_db=None,  # The database of opinfos to search for the parent opinfo
        torch_opinfo_name,  # the string name of the corresponding torch opinfo
        torch_opinfo_variant_name="",  # the variant name for corresponding torch opinfo
        validate_view_consistency=True,
        **kwargs,
    ):  # additional kwargs override kwargs inherited from the torch opinfo
        # [FUNCTION DOC] Finds the parent UnaryUfuncInfo, merges its original unary ufunc
        # arguments with the kwargs overrides and initializes the base class. Raises
        # AssertionError if the parent is missing or is not a UnaryUfuncInfo.
        self.torch_opinfo_name = torch_opinfo_name
        self.torch_opinfo_variant_name = torch_opinfo_variant_name
        self.torch_opinfo = _find_referenced_opinfo(
            torch_opinfo_name, torch_opinfo_variant_name, op_db=op_db
        )
        self.validate_view_consistency = validate_view_consistency
        if not isinstance(self.torch_opinfo, UnaryUfuncInfo):
            raise AssertionError(
                f"Expected torch_opinfo to be UnaryUfuncInfo, got {type(self.torch_opinfo)}"
            )

        inherited = self.torch_opinfo._original_unary_ufunc_args
        ukwargs = _inherit_constructor_args(name, op, inherited, kwargs)

        super().__init__(**ukwargs)


class ElementwiseBinaryPythonRefInfo(BinaryUfuncInfo):
    # [CLASS DOC] OpInfo for a Python reference of an elementwise binary operation. Same
    # lookup attributes as PythonRefInfo; the parent must be a BinaryUfuncInfo.
    """
    An OpInfo for a Python reference of an elementwise binary operation.
    """

    def __init__(
        self,
        name,  # the stringname of the callable Python reference
        *,
        op=None,  # the function variant of the operation, populated as torch.<name> if None
        op_db=None,  # The database of opinfos to search for the parent opinfo
        torch_opinfo_name,  # the string name of the corresponding torch opinfo
        torch_opinfo_variant_name="",  # the variant name for corresponding torch opinfo
        **kwargs,
    ):  # additional kwargs override kwargs inherited from the torch opinfo
        # [FUNCTION DOC] Finds the parent BinaryUfuncInfo, merges its original binary ufunc
        # arguments with the kwargs overrides and initializes the base class. Raises
        # AssertionError if the parent is missing or is not a BinaryUfuncInfo.
        self.torch_opinfo_name = torch_opinfo_name
        self.torch_opinfo_variant_name = torch_opinfo_variant_name
        self.torch_opinfo = _find_referenced_opinfo(
            torch_opinfo_name, torch_opinfo_variant_name, op_db=op_db
        )
        if not isinstance(self.torch_opinfo, BinaryUfuncInfo):
            raise AssertionError(
                f"Expected torch_opinfo to be BinaryUfuncInfo, got {type(self.torch_opinfo)}"
            )

        inherited = self.torch_opinfo._original_binary_ufunc_args
        ukwargs = _inherit_constructor_args(name, op, inherited, kwargs)

        super().__init__(**ukwargs)
