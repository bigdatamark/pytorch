# [MODULE DOC] Records and replays Python execution state: ModuleRecord, DummyModule,
# ExecutionRecord (dill-serializable snapshot of code, closure, globals, locals, builtins)
# and ExecutionRecorder (collects them while tracing). Import-time side effect: imports
# dill via import_dill (may be None if not installed).
from __future__ import annotations


"""
Python execution state recording and replay functionality.

This module provides mechanisms for capturing and replaying Python execution state:

- ModuleRecord: Tracks module access patterns and attribute usage
- DummyModule: Lightweight module substitute for replay
- ExecutionRecord: Manages execution context including globals, locals and builtins
- ExecutionRecorder: Records variable states and module access during execution

The module enables serialization and reproduction of Python execution environments,
particularly useful for debugging and testing frameworks that need to capture
and recreate specific program states.
"""

import dataclasses
from dataclasses import field
from types import CellType, CodeType, ModuleType
from typing import Any, cast, IO, TYPE_CHECKING
from typing_extensions import Self

from torch.utils._import_utils import import_dill


if TYPE_CHECKING:
    from io import BufferedReader, BufferedWriter

    from .output_graph import CodeOptions


dill = import_dill()


@dataclasses.dataclass
class ModuleRecord:
    # [CLASS DOC] Tracks a real module and the attributes accessed on it (name to value),
    # so a stand-in module can be rebuilt later.
    module: ModuleType
    accessed_attrs: dict[str, Any] = field(default_factory=dict)


@dataclasses.dataclass
class DummyModule:
    # [CLASS DOC] Lightweight stand-in for a module, with name, is_torch flag and an
    # optional value; attributes accessed during recording are set on it dynamically.
    name: str
    is_torch: bool = False
    value: object = None

    @property
    def __name__(self) -> str:
        # [FUNCTION DOC] Property returning the stand-in module's name.
        return self.name


@dataclasses.dataclass
class ExecutionRecord:
    # [CLASS DOC] Serializable snapshot of an execution: code object, closure cells, globals,
    # locals, builtins and code options. Can be dumped to and loaded from a file with dill.
    code: CodeType
    closure: tuple[CellType]
    globals: dict[str, Any] = field(default_factory=dict)
    locals: dict[str, Any] = field(default_factory=dict)
    builtins: dict[str, Any] = field(default_factory=dict)
    # The replay record starts empty and gets populated by the translator before use.
    code_options: CodeOptions = field(default_factory=lambda: cast("CodeOptions", {}))

    def dump(self, f: IO[str] | BufferedWriter) -> None:
        # [FUNCTION DOC] Serializes this record to the open file with dill. Raises
        # AssertionError if dill is not installed.
        if dill is None:
            raise AssertionError("replay_record requires `pip install dill`")
        dill.dump(self, f)

    @classmethod
    def load(cls, f: IO[bytes] | BufferedReader) -> Self:
        # [FUNCTION DOC] Deserializes and returns a record from the open file with dill.
        # Raises AssertionError if dill is not installed.
        if dill is None:
            raise AssertionError("replay_record requires `pip install dill`")
        return dill.load(f)


@dataclasses.dataclass
class ExecutionRecorder:
    # [CLASS DOC] Collects execution state during tracing: code, closure, globals, locals,
    # builtins, code options and name_to_modrec (module name to ModuleRecord). Modules stored
    # as variables are wrapped in ModuleRecords; get_record produces an ExecutionRecord with
    # modules replaced by DummyModules.
    LOCAL_MOD_PREFIX = "___local_mod_"

    code: CodeType
    closure: tuple[CellType]
    globals: dict[str, Any] = field(default_factory=dict)
    locals: dict[str, Any] = field(default_factory=dict)
    builtins: dict[str, Any] = field(default_factory=dict)
    # The recorder starts empty and gets populated by the translator before use.
    code_options: CodeOptions = field(default_factory=lambda: cast("CodeOptions", {}))
    name_to_modrec: dict[str, ModuleRecord] = field(default_factory=dict)

    def add_local_var(self, name: str, var: Any) -> None:
        # [FUNCTION DOC] Records a local variable; module values are stored as module records.
        if isinstance(var, ModuleType):
            self.locals[name] = self._add_mod(var)
        else:
            self.locals[name] = var

    def add_global_var(self, name: str, var: Any) -> None:
        # [FUNCTION DOC] Records a global variable; module values are stored as module records.
        if isinstance(var, ModuleType):
            self.globals[name] = self._add_mod(var)
        else:
            self.globals[name] = var

    def add_local_mod(self, name: str, mod: ModuleType) -> None:
        # [FUNCTION DOC] Records a module under the given name. Raises AssertionError if mod
        # is not a module.
        # KNOWN ISSUE: despite the name, it registers the module as a global, not a local.
        if not isinstance(mod, ModuleType):
            raise AssertionError(f"Expected ModuleType, got {type(mod)}")
        self.add_global_var(name, mod)

    def record_module_access(self, mod: ModuleType, name: str, val: Any) -> None:
        # [FUNCTION DOC] Notes that attribute name of module mod had value val. Module-valued
        # attributes are wrapped as module records; other values are stored only if mod is
        # already tracked.
        # KNOWN ISSUE: for module-valued attributes the owning module must already be tracked,
        # otherwise a KeyError is raised.
        if isinstance(val, ModuleType):
            self.name_to_modrec[mod.__name__].accessed_attrs[name] = self._add_mod(val)
            return

        if mod.__name__ in self.name_to_modrec:
            self.name_to_modrec[mod.__name__].accessed_attrs[name] = val

    def get_record(self) -> ExecutionRecord:
        # [FUNCTION DOC] Builds an ExecutionRecord from the current state, resolving module
        # records into dummy modules and copying builtins and code options.
        return ExecutionRecord(
            self.code,
            self.closure,
            ExecutionRecorder._resolve_modules(self.globals),
            ExecutionRecorder._resolve_modules(self.locals),
            self.builtins.copy(),
            self.code_options.copy(),
        )

    def _add_mod(self, mod: ModuleType) -> ModuleRecord:
        # [FUNCTION DOC] HELPER: returns the module record for mod, creating and registering
        # it by module name if needed.
        if mod.__name__ not in self.name_to_modrec:
            self.name_to_modrec[mod.__name__] = ModuleRecord(mod)

        return self.name_to_modrec[mod.__name__]

    @classmethod
    def _resolve_modules(cls, vars: dict[str, Any]) -> dict[str, Any]:
        # [FUNCTION DOC] HELPER: returns a copy of the mapping with every module record
        # replaced by a dummy module carrying its accessed attributes (resolved recursively).
        def resolve_module(var: Any) -> Any:
            # [FUNCTION DOC] HELPER: non-records pass through; a record becomes a dummy module
            # with each accessed attribute set to its resolved value.
            if not isinstance(var, ModuleRecord):
                return var

            dummy_mod = DummyModule(var.module.__name__)
            for attr_name, attr_value in var.accessed_attrs.items():
                attr_value = resolve_module(attr_value)
                dummy_mod.__setattr__(attr_name, attr_value)

            return dummy_mod

        return {k: resolve_module(v) for k, v in vars.items()}
