# [MODULE DOC] Static typing check of the return types of the script entry point of the jit
# package. It defines a ParamSpec and a covariant TypeVar (both unused later) and a small Enum,
# then asserts the inferred type for scripting an Enum class, a dict, a list, a module and a
# function. The final block expects scripting a module class (not an instance) to raise a
# RuntimeError, via pytest raises, and the assert_never inside it marks the call as returning a
# bottom type. All of these scripting calls execute at import time.
from enum import Enum
from typing import TypeVar
from typing_extensions import assert_never, assert_type, ParamSpec

import pytest

from torch import jit, nn, ScriptDict, ScriptFunction, ScriptList


P = ParamSpec("P")
R = TypeVar("R", covariant=True)


class Color(Enum):
    # [CLASS DOC] HELPER: Plain Enum with three members (RED, GREEN, BLUE with values 1 to 3),
    # used only as the argument whose scripted type is checked.
    RED = 1
    GREEN = 2
    BLUE = 3


# Script Enum
assert_type(jit.script(Color), type[Color])

# ScriptDict
assert_type(jit.script({1: 1}), ScriptDict)

# ScriptList
assert_type(jit.script([0]), ScriptList)

# ScriptModule
scripted_module = jit.script(nn.Linear(2, 2))
assert_type(scripted_module, jit.RecursiveScriptModule)

# ScripFunction
# NOTE: can't use assert_type because of parameter names
# NOTE: Generic usage only possible with Python 3.9
relu: ScriptFunction = jit.script(nn.functional.relu)

# can't script nn.Module class
with pytest.raises(RuntimeError):
    assert_never(jit.script(nn.Linear))
