# mypy: allow-untyped-defs
# [MODULE DOC] Defines QConfigMultiMapping, a container holding several QConfigMapping
# objects so each qconfig matching style/key can carry a list of qconfigs (used by the
# n-shadows model preparation API). Also holds the style-name to setter-method-name table.
# No import-time side effects beyond the constant definitions.
from __future__ import annotations

import copy
from typing import Any, TYPE_CHECKING

import torch
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.qconfig_mapping import _QCONFIG_STYLE_ORDER


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.ao.quantization.qconfig import QConfigAny

__all__ = ["QConfigMultiMapping"]

_QCONFIG_STYLE_TO_METHOD: dict[str, str] = {
    "global_qconfig": "set_global",
    "object_type_qconfigs": "set_object_type",
    "module_name_regex_qconfigs": "set_module_name_regex",
    "module_name_qconfigs": "set_module_name",
    "module_name_object_type_order_qconfigs": "set_module_name_object_type_order",
}


def _remove_duplicates_and_none(qconfig_list: list[QConfigAny]) -> None:
    # [FUNCTION DOC] HELPER: Mutates qconfig_list in place, removing None and duplicate qconfigs.
    # Duplicates are detected with qconfig_equals against earlier entries. Returns nothing.
    # KNOWN ISSUE: the scan stops (break) at the first None, so entries after it are never
    # checked for duplicates, and only that first None is removed.
    to_remove = []
    for index, cur_qconfig in enumerate(qconfig_list):
        if cur_qconfig is None:
            to_remove.append(index)
            break
        for checked_qconfig in qconfig_list[:index]:
            if torch.ao.quantization.qconfig_equals(cur_qconfig, checked_qconfig):
                to_remove.append(index)
                break
    for index in to_remove[::-1]:
        qconfig_list.pop(index)


class QConfigMultiMapping:
    # [CLASS DOC] Holds a list of QConfigMapping objects (attribute qconfig_mappings_list,
    # starts with one empty mapping) so multiple qconfigs can be given per style and key.
    # Invariant: if any mapping has a qconfig for a style+key, every mapping has an entry
    # (a qconfig or None) for it, so shorter lists never leak global-style matches.
    # Setter methods mirror QConfigMapping but take lists and return self for chaining;
    # from_list_qconfig_mapping builds an instance from existing mappings.
    """
    This class, used with the prepare_n_shadows_model API, stores a list of :class:`torch.ao.quantization.QConfigMapping`s
    so that multiple QConfigs can be specified for each QConfig matching style.

    The user can specify QConfigs using the following methods (in increasing match priority):

        ``set_global`` : sets the global (default) QConfigs

        ``set_object_type`` : sets the QConfigs for a given module type, function, or method name

        ``set_module_name_regex`` : sets the QConfigs for modules matching the given regex string

        ``set_module_name`` : sets the QConfigs for modules matching the given module name

        ``set_module_name_object_type_order`` : sets the QConfigs for modules matching a combination
        of the given module name, object type, and the index at which the module appears

    Note: Usage of set methods is the same as in QConfigMapping except with a passed in list of QConfigs rather than a
    single QConfig.

    Example usage::

        qconfig_mapping = QConfigMultiMapping()
            .set_global([qconfig1, qconfig2])
            .set_object_type(torch.nn.Linear, [qconfig2, qconfig3])
            .set_object_type(torch.nn.ReLU, [qconfig1])
            .set_module_name_regex("foo.*bar.*conv[0-9]+", [qconfig2])
            .set_module_name_regex("foo.*", [qconfig1, qconfig2, qconfig3])
            .set_module_name("module1", [None])
            .set_module_name("module2", [qconfig2])
            .set_module_name_object_type_order("foo.bar", torch.nn.functional.linear, 0, [qconfig3])

    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Creates the instance with a single default QConfigMapping in the list.
        # initialize this with 1 QConfigMapping to avoid corner cases
        self.qconfig_mappings_list: list[QConfigMapping] = [QConfigMapping()]

    def _handle_list_size_mismatch(
        self, qconfig_list: list[QConfigAny], style: str
    ) -> None:
        # [FUNCTION DOC] HELPER: Reconciles qconfig_list length with the number of stored mappings.
        # If the list is longer, appends deep copies of a new mapping whose non-global style
        # keys (taken from the first existing mapping) are set to None, until sizes match.
        # If shorter, pads qconfig_list in place with None. Mutates self and qconfig_list.
        # The style argument is unused.
        # this method handles cases where the size of qconfig_list does not match
        # the size of qconfig_mappings_list.
        # Issue: Consider a user inserting global_qconfig A and B first, then inserting
        # qconfig C as an object_type_qconfig for conv ops. If we internally store
        # 1 QConfigMapping with A and C and another with just B, then the
        # second QConfigMapping will match B to conv ops (which is not wanted), since B is global.

        # we avoid this by maintaining the invariant that if any QConfigMapping
        # has a qconfig style+key with a qconfig in it, all QConfigMappings must
        # have either a qconfig or None for that same style+key. In the above
        # example, a None qconfig would prevent the unwanted match in the
        # second QConfigMapping

        if len(qconfig_list) > len(self.qconfig_mappings_list):
            # Case: we have more qconfigs (in qconfig_list) than QConfigMappings

            # Add new QConfigMappings (initialized so we maintain the `invariant`)

            new_qconfig_mapping = QConfigMapping()
            # searches other QConfigMappings for qconfig style+keys
            # that need to be inserted as `None` into the new QConfigMapping
            for qconfig_mapping in self.qconfig_mappings_list:
                # global_qconfig has None by default
                for check_style in _QCONFIG_STYLE_ORDER[1:]:
                    qconfigs_dict = getattr(qconfig_mapping, check_style)
                    target_qconfigs_dict = getattr(new_qconfig_mapping, check_style)
                    for key in qconfigs_dict:
                        target_qconfigs_dict[key] = None
                break

            # insert copies of this new QConfigMapping until all entries
            # in qconfig_list can fit among the QConfigMappings
            while len(qconfig_list) > len(self.qconfig_mappings_list):
                self.qconfig_mappings_list.append(copy.deepcopy(new_qconfig_mapping))
        else:
            # Case: we have fewer qconfigs in qconfig_list than QConfigMappings

            # pad qconfig_list with `None` until length is same
            while len(qconfig_list) < len(self.qconfig_mappings_list):
                qconfig_list.append(None)

    # this function applies the insertion method across each QConfigMapping
    def _insert_qconfig_list(
        self,
        style: str,
        args: list[str | int | Callable],
        qconfig_list: list[QConfigAny],
    ) -> None:
        # [FUNCTION DOC] HELPER: Inserts a list of qconfigs for one style across all mappings.
        # Strips None/duplicates from qconfig_list, balances sizes, then calls the matching
        # QConfigMapping setter (looked up from the style) on each mapping with args plus
        # the corresponding qconfig. Mutates qconfig_list and self; raises KeyError for unknown style.
        # we remove duplicates and None to make the ordering of qconfigs
        # deterministic upon insertion.
        _remove_duplicates_and_none(qconfig_list)

        self._handle_list_size_mismatch(qconfig_list, style)
        method_name = _QCONFIG_STYLE_TO_METHOD[style]
        for qconfig_mapping, qconfig in zip(self.qconfig_mappings_list, qconfig_list):
            # uses QConfigMapping set method to insert qconfig
            set_method = getattr(qconfig_mapping, method_name)
            set_method(*args, qconfig)

    def set_global(self, global_qconfig_list: list[QConfigAny]) -> QConfigMultiMapping:
        # [FUNCTION DOC] Sets the global qconfigs from a list; returns self for chaining.
        """
        Set global QConfigs
        see :func:`~torch.ao.quantization.QConfigMapping.set_global()` for more info
        """
        self._insert_qconfig_list("global_qconfig", [], global_qconfig_list)
        return self

    def set_object_type(
        self, object_type: Callable | str, qconfig_list: list[QConfigAny]
    ) -> QConfigMultiMapping:
        # [FUNCTION DOC] Sets the qconfigs for an object type (module class, function or method name); returns self.
        """
        Set object type QConfigs
        see :func:`~torch.ao.quantization.QConfigMapping.set_object_type()` for more info
        """
        self._insert_qconfig_list("object_type_qconfigs", [object_type], qconfig_list)
        return self

    def set_module_name_regex(
        self, module_name_regex: str, qconfig_list: list[QConfigAny]
    ) -> QConfigMultiMapping:
        # [FUNCTION DOC] Sets the qconfigs for modules whose names match a regex string; returns self.
        """
        Set module_name_regex QConfigs
        see :func:`~torch.ao.quantization.QConfigMapping.set_module_name_regex()` for more info
        """
        self._insert_qconfig_list(
            "module_name_regex_qconfigs", [module_name_regex], qconfig_list
        )
        return self

    def set_module_name(
        self, module_name: str, qconfig_list: list[QConfigAny]
    ) -> QConfigMultiMapping:
        # [FUNCTION DOC] Sets the qconfigs for the module with the given name; returns self.
        """
        Set module_name QConfigs
        see :func:`~torch.ao.quantization.QConfigMapping.set_module_name()` for more info
        """
        self._insert_qconfig_list("module_name_qconfigs", [module_name], qconfig_list)
        return self

    def set_module_name_object_type_order(
        self,
        module_name: str,
        object_type: Callable,
        index: int,
        qconfig_list: list[QConfigAny],
    ) -> QConfigMultiMapping:
        # [FUNCTION DOC] Sets qconfigs for a module name, object type and occurrence index combination; returns self.
        """
        Set module_name QConfigs
        see :func:`~torch.ao.quantization.QConfigMapping.set_module_name_object_type_order()` for more info
        """
        self._insert_qconfig_list(
            "module_name_object_type_order_qconfigs",
            [module_name, object_type, index],
            qconfig_list,
        )
        return self

    def __repr__(self):
        # [FUNCTION DOC] Returns a string: the class name followed by the repr of each stored mapping, one per line, in brackets.
        return (
            self.__class__.__name__
            + " ["
            + "".join(
                f"\n{qconfig_mapping.__repr__()},"
                for qconfig_mapping in self.qconfig_mappings_list
            )
            + "\n]"
        )

    @classmethod
    def from_list_qconfig_mapping(
        cls, qconfig_mapping_list: list[QConfigMapping]
    ) -> QConfigMultiMapping:
        # [FUNCTION DOC] Alternate constructor: builds a QConfigMultiMapping from a list of QConfigMapping.
        # Deep-copies the list into the new instance, then, for every non-global style, gathers
        # the qconfigs per key across the mappings and re-inserts them through the setters so
        # the size-balancing invariant holds. Tuple keys are unpacked as positional arguments.
        # Does not modify the input list; returns the new instance.
        """
        Creates a QConfigMultiMapping from a list of QConfigMappings
        """
        new_qconfig_multi_mapping = cls()

        new_qconfig_multi_mapping.qconfig_mappings_list = copy.deepcopy(
            qconfig_mapping_list
        )

        # we need to avoid the issue described in _handle_list_size_mismatch,
        # so we reinsert all the qconfigs using the QConfigMultiMapping
        # set methods

        # go through all qconfig styles
        # note: global can be ignored since it is None by default
        for style in _QCONFIG_STYLE_ORDER[1:]:
            # gather all key+qconfigs for current style
            # into qconfig_dict_list
            qconfig_dict_list: dict[Any, list[QConfigAny]] = {}
            for qconfig_mapping in qconfig_mapping_list:
                qconfig_dict = getattr(qconfig_mapping, style)
                for key, qconfig in qconfig_dict.items():
                    if key not in qconfig_dict_list:
                        qconfig_dict_list[key] = []
                    qconfig_dict_list[key].append(qconfig)

            # reinsert all gathered key+qconfigs
            set_method_name = _QCONFIG_STYLE_TO_METHOD[style]
            set_method = getattr(new_qconfig_multi_mapping, set_method_name)
            for key, qconfig_list in qconfig_dict_list.items():
                if isinstance(key, tuple):
                    set_method(*key, qconfig_list)
                else:
                    set_method(key, qconfig_list)

        return new_qconfig_multi_mapping
