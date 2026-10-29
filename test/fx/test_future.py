# Owner(s): ["module: fx"]
# [MODULE DOC] Script-style test (no test class) of FX symbolic tracing in a file that uses
# postponed evaluation of annotations. Defines a callable class and four modules whose
# forward signatures use annotations (some referring to the callable class, some with a
# parameterized list type), traces each at import time and raises AssertionError if the
# traced output differs from a reference tensor sum. All work happens at import time.

from __future__ import annotations  # type: ignore[attr-defined]

import torch
from torch.fx import symbolic_trace


class A:
    # [CLASS DOC] HELPER: Plain callable class used as an annotation target and as a call
    # argument.
    def __call__(self, x: torch.Tensor):
        # [FUNCTION DOC] HELPER: Callable test object. Returns the input tensor added to
        # itself.
        return torch.add(x, x)


# No forward references
class M1(torch.nn.Module):
    # [CLASS DOC] HELPER: Module with forward annotations using only tensor and the callable
    # class.
    def forward(self, x: torch.Tensor, a: A) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Calls the supplied callable on the tensor and returns the
        # result. Annotations are plain, with no forward references.
        return a(x)


# Forward references
class M2(torch.nn.Module):
    # [CLASS DOC] HELPER: Module whose comment says forward references; structurally
    # identical to the first module.
    def forward(self, x: torch.Tensor, a: A) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Same behavior as the first module; the section comment
        # labels it as the forward references case. Calls the callable on the tensor and
        # returns the result.
        return a(x)


# Non-torch annotation with no internal forward references
class M3(torch.nn.Module):
    # [CLASS DOC] HELPER: Module with a list-of-tensors annotation (non-torch generic type).
    def forward(self, x: list[torch.Tensor], a: A) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Takes a list of tensors and a callable; calls the callable
        # on the first tensor and returns the result.
        return a(x[0])


# Non-torch annotation with internal forward references
class M4(torch.nn.Module):
    # [CLASS DOC] HELPER: Module with a list-of-tensors annotation, labelled as having
    # internal forward references; structurally identical to the third module.
    def forward(self, x: list[torch.Tensor], a: A) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Same behavior as the third module; takes a list of tensors
        # and a callable, calls it on the first tensor and returns the result.
        return a(x[0])


x = torch.rand(2, 3)

ref = torch.add(x, x)

traced1 = symbolic_trace(M1())
res1 = traced1(x, A())
if not torch.all(torch.eq(ref, res1)):
    raise AssertionError("res1 does not match ref")

traced2 = symbolic_trace(M2())
res2 = traced2(x, A())
if not torch.all(torch.eq(ref, res2)):
    raise AssertionError("res2 does not match ref")

traced3 = symbolic_trace(M3())
res3 = traced3([x], A())
if not torch.all(torch.eq(ref, res3)):
    raise AssertionError("res3 does not match ref")

traced4 = symbolic_trace(M4())
res4 = traced4([x], A())
if not torch.all(torch.eq(ref, res4)):
    raise AssertionError("res4 does not match ref")
