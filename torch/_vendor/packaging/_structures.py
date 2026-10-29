# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.

# [MODULE DOC] HELPER: Sentinel types used by the vendored version parser for
# ordering. Defines InfinityType and NegativeInfinityType plus their singleton
# instances Infinity and NegativeInfinity, which compare greater / smaller than
# everything. No import-time side effects beyond creating the two singletons.


class InfinityType:
    # [CLASS DOC] HELPER: Sentinel that compares greater than every other object
    # and equal only to another InfinityType. Stateless; negation yields NegativeInfinity.
    def __repr__(self) -> str:
        # [FUNCTION DOC] HELPER: Returns the text "Infinity".
        return "Infinity"

    def __hash__(self) -> int:
        # [FUNCTION DOC] HELPER: Hash derived from the repr text.
        return hash(repr(self))

    def __lt__(self, other: object) -> bool:
        # [FUNCTION DOC] HELPER: Always false (never less than anything).
        return False

    def __le__(self, other: object) -> bool:
        # [FUNCTION DOC] HELPER: Always false, even against another Infinity.
        # KNOWN ISSUE: less-or-equal is false even for equal instances, inconsistent with equality.
        return False

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] HELPER: True only if other is an instance of this class.
        return isinstance(other, self.__class__)

    def __gt__(self, other: object) -> bool:
        # [FUNCTION DOC] HELPER: Always true.
        return True

    def __ge__(self, other: object) -> bool:
        # [FUNCTION DOC] HELPER: Always true.
        return True

    def __neg__(self: object) -> "NegativeInfinityType":
        # [FUNCTION DOC] HELPER: Returns the NegativeInfinity singleton.
        return NegativeInfinity


Infinity = InfinityType()


class NegativeInfinityType:
    # [CLASS DOC] HELPER: Sentinel that compares smaller than every other object
    # and equal only to another NegativeInfinityType. Stateless; negation yields Infinity.
    def __repr__(self) -> str:
        # [FUNCTION DOC] HELPER: Returns the text "-Infinity".
        return "-Infinity"

    def __hash__(self) -> int:
        # [FUNCTION DOC] HELPER: Hash derived from the repr text.
        return hash(repr(self))

    def __lt__(self, other: object) -> bool:
        # [FUNCTION DOC] HELPER: Always true.
        return True

    def __le__(self, other: object) -> bool:
        # [FUNCTION DOC] HELPER: Always true.
        return True

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] HELPER: True only if other is an instance of this class.
        return isinstance(other, self.__class__)

    def __gt__(self, other: object) -> bool:
        # [FUNCTION DOC] HELPER: Always false.
        return False

    def __ge__(self, other: object) -> bool:
        # [FUNCTION DOC] HELPER: Always false, even against another NegativeInfinity.
        # KNOWN ISSUE: greater-or-equal is false even for equal instances, inconsistent with equality.
        return False

    def __neg__(self: object) -> InfinityType:
        # [FUNCTION DOC] HELPER: Returns the Infinity singleton.
        return Infinity


NegativeInfinity = NegativeInfinityType()
