# [MODULE DOC] Defines the abstract Check interface: an execution check that produces the lines
# of Python code appended to a generated fuzz program. Concrete checks live in device plugins.
"""Check abstractions for different execution modes and validations.

Concrete :class:`Check` subclasses are owned by device plugins (see
``torchfuzz/cuda/_checks.py`` for the CUDA reference implementation).
"""

from abc import ABC, abstractmethod


class Check(ABC):
    # [CLASS DOC] Abstract base of execution checks; subclasses implement codegen to emit the
    # code that runs a generated program (for example eagerly and compiled) and validates it.
    # Stateless.
    """Base class for execution checks."""

    @abstractmethod
    def codegen(self, args_tuple: str) -> list[str]:
        # [FUNCTION DOC] Abstract: returns the lines of generated code for this check, given the
        # source text of the tuple expression holding the program arguments.
        """Generate code lines for this check."""
