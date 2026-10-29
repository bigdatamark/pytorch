# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# [MODULE DOC] Test module for Python structural pattern matching (match/case statements),
# copied from the CPython 3.13 test suite and adapted to run under the dynamo CPython test
# harness. Holds the Point dataclass fixture and test classes: TestCompiler (compiles this
# file), TestInheritance (sequence vs mapping classification), TestPatma (behavioral cases
# numbered test_patma_000 to 255 plus protocol cases), TestSyntaxErrors (invalid patterns must
# fail to compile), TestTypeErrors and TestValueErrors (runtime errors from bad class or
# mapping patterns), TestSourceLocations (jump instructions keep line numbers) and TestTracing
# (line events while matching). Running the file directly calls run_tests.
# KNOWN ISSUE: the header comment names test_collections.py as the source, but the content is
# CPython's test_patma.py.

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_collections.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

# ======= END DYNAMO PATCH =======

import array
import collections
import dataclasses
import dis
import enum
import inspect
import sys
import unittest


@dataclasses.dataclass
class Point:
    # [CLASS DOC] Dataclass fixture with integer fields x and y. The generated __match_args__
    # is (x, y), so it supports positional and keyword class patterns in the match tests
    # below.
    x: int
    y: int


class TestCompiler(CPythonTestCase):
    # [CLASS DOC] Test case with a single test that exercises the compiler on this very file.

    @torch._dynamo.error_on_graph_break(False)
    def test_refleaks(self):
        # [FUNCTION DOC] Reads the source of this file and compiles it to a code object in
        # exec mode, discarding the result. Guards against reference leaks inside the pattern
        # compiler, which refleak runs would otherwise miss. Wrapped so that graph breaks do
        # not raise under dynamo. Raises only if compilation fails.
        # Hunting for leaks using -R doesn't catch leaks in the compiler itself,
        # just the code under test. This test ensures that if there are leaks in
        # the pattern compiler, those runs will fail:
        with open(__file__) as file:
            compile(file.read(), __file__, "exec")


class TestInheritance(CPythonTestCase):
    # [CLASS DOC] Tests how class patterns classify subclasses of Sequence and Mapping
    # abstract bases: sequence and mapping patterns are chosen from inheritance flags,
    # regardless of pattern order or registration time.

    @staticmethod
    def check_sequence_then_mapping(x):
        # [FUNCTION DOC] HELPER: Static helper. Matches the argument against a sequence
        # pattern first and a mapping pattern second; returns "seq" for sequences, "map" for
        # mappings, and None if neither applies.
        match x:
            case [*_]:
                return "seq"
            case {}:
                return "map"

    @staticmethod
    def check_mapping_then_sequence(x):
        # [FUNCTION DOC] HELPER: Static helper, the mirror of the other checker: tries the
        # empty mapping pattern first and then the star sequence pattern, returning "map" or
        # "seq" accordingly, else None.
        match x:
            case {}:
                return "map"
            case [*_]:
                return "seq"

    def test_multiple_inheritance_mapping(self):
        # [FUNCTION DOC] Builds four classes that mix a mapping base (UserDict or dict) with
        # Sequence or list and a plain class in different orders. Asserts every instance is
        # classified as a mapping by both checkers, so the first matching base flag decides
        # the kind.
        class C:
            # [CLASS DOC] HELPER: plain base class used only to vary the method resolution
            # order of the mixed classes.
            pass
        class M1(collections.UserDict, collections.abc.Sequence):
            # [CLASS DOC] HELPER: mapping-first mixed class (UserDict then Sequence), empty
            # body.
            pass
        class M2(C, collections.UserDict, collections.abc.Sequence):
            # [CLASS DOC] HELPER: plain class first, then UserDict then Sequence; checks MRO
            # position does not hide the mapping flag.
            pass
        class M3(collections.UserDict, C, list):
            # [CLASS DOC] HELPER: UserDict, plain class and list; mapping base listed first.
            pass
        class M4(dict, collections.abc.Sequence, C):
            # [CLASS DOC] HELPER: dict, Sequence and plain class; dict base wins.
            pass
        self.assertEqual(self.check_sequence_then_mapping(M1()), "map")
        self.assertEqual(self.check_sequence_then_mapping(M2()), "map")
        self.assertEqual(self.check_sequence_then_mapping(M3()), "map")
        self.assertEqual(self.check_sequence_then_mapping(M4()), "map")
        self.assertEqual(self.check_mapping_then_sequence(M1()), "map")
        self.assertEqual(self.check_mapping_then_sequence(M2()), "map")
        self.assertEqual(self.check_mapping_then_sequence(M3()), "map")
        self.assertEqual(self.check_mapping_then_sequence(M4()), "map")

    def test_multiple_inheritance_sequence(self):
        # [FUNCTION DOC] Builds four classes mixing sequence bases (UserList or list) with
        # Mapping or dict and a plain class. Asserts both checkers classify every instance as
        # a sequence, the mirror of the mapping test.
        class C:
            # [CLASS DOC] HELPER: plain base class used only to vary the method resolution
            # order of the mixed classes.
            pass
        class S1(collections.UserList, collections.abc.Mapping):
            # [CLASS DOC] HELPER: sequence-first mixed class (UserList then Mapping), empty
            # body.
            pass
        class S2(C, collections.UserList, collections.abc.Mapping):
            # [CLASS DOC] HELPER: plain class first, then UserList then Mapping.
            pass
        class S3(list, C, collections.abc.Mapping):
            # [CLASS DOC] HELPER: list, plain class and Mapping; list base wins.
            pass
        class S4(collections.UserList, dict, C):
            # [CLASS DOC] HELPER: UserList, dict and plain class; sequence base listed first.
            pass
        self.assertEqual(self.check_sequence_then_mapping(S1()), "seq")
        self.assertEqual(self.check_sequence_then_mapping(S2()), "seq")
        self.assertEqual(self.check_sequence_then_mapping(S3()), "seq")
        self.assertEqual(self.check_sequence_then_mapping(S4()), "seq")
        self.assertEqual(self.check_mapping_then_sequence(S1()), "seq")
        self.assertEqual(self.check_mapping_then_sequence(S2()), "seq")
        self.assertEqual(self.check_mapping_then_sequence(S3()), "seq")
        self.assertEqual(self.check_mapping_then_sequence(S4()), "seq")

    def test_late_registration_mapping(self):
        # [FUNCTION DOC] Registers a parent class as a virtual Mapping after one child and
        # grandchild already exist, then creates more descendants. Asserts parent, pre-
        # registration and post-registration descendants are all classified as mappings by
        # both checkers.
        class Parent:
            # [CLASS DOC] HELPER: parent that is later registered as a virtual ABC; empty
            # body.
            pass
        class ChildPre(Parent):
            # [CLASS DOC] HELPER: child created before the parent is registered.
            pass
        class GrandchildPre(ChildPre):
            # [CLASS DOC] HELPER: grandchild created before the parent is registered.
            pass
        collections.abc.Mapping.register(Parent)
        class ChildPost(Parent):
            # [CLASS DOC] HELPER: child created after the parent is registered.
            pass
        class GrandchildPost(ChildPost):
            # [CLASS DOC] HELPER: grandchild created after the parent is registered.
            pass
        self.assertEqual(self.check_sequence_then_mapping(Parent()), "map")
        self.assertEqual(self.check_sequence_then_mapping(ChildPre()), "map")
        self.assertEqual(self.check_sequence_then_mapping(GrandchildPre()), "map")
        self.assertEqual(self.check_sequence_then_mapping(ChildPost()), "map")
        self.assertEqual(self.check_sequence_then_mapping(GrandchildPost()), "map")
        self.assertEqual(self.check_mapping_then_sequence(Parent()), "map")
        self.assertEqual(self.check_mapping_then_sequence(ChildPre()), "map")
        self.assertEqual(self.check_mapping_then_sequence(GrandchildPre()), "map")
        self.assertEqual(self.check_mapping_then_sequence(ChildPost()), "map")
        self.assertEqual(self.check_mapping_then_sequence(GrandchildPost()), "map")

    def test_late_registration_sequence(self):
        # [FUNCTION DOC] Same scenario as the mapping version but registers the parent as a
        # virtual Sequence. Asserts all five instances are classified as sequences.
        class Parent:
            # [CLASS DOC] HELPER: parent that is later registered as a virtual Sequence; empty
            # body.
            pass
        class ChildPre(Parent):
            # [CLASS DOC] HELPER: child created before the registration.
            pass
        class GrandchildPre(ChildPre):
            # [CLASS DOC] HELPER: grandchild created before the registration.
            pass
        collections.abc.Sequence.register(Parent)
        class ChildPost(Parent):
            # [CLASS DOC] HELPER: child created after the registration.
            pass
        class GrandchildPost(ChildPost):
            # [CLASS DOC] HELPER: grandchild created after the registration.
            pass
        self.assertEqual(self.check_sequence_then_mapping(Parent()), "seq")
        self.assertEqual(self.check_sequence_then_mapping(ChildPre()), "seq")
        self.assertEqual(self.check_sequence_then_mapping(GrandchildPre()), "seq")
        self.assertEqual(self.check_sequence_then_mapping(ChildPost()), "seq")
        self.assertEqual(self.check_sequence_then_mapping(GrandchildPost()), "seq")
        self.assertEqual(self.check_mapping_then_sequence(Parent()), "seq")
        self.assertEqual(self.check_mapping_then_sequence(ChildPre()), "seq")
        self.assertEqual(self.check_mapping_then_sequence(GrandchildPre()), "seq")
        self.assertEqual(self.check_mapping_then_sequence(ChildPost()), "seq")
        self.assertEqual(self.check_mapping_then_sequence(GrandchildPost()), "seq")


class TestPatma(CPythonTestCase):
    # [CLASS DOC] Main behavioral suite for match statements. Each numbered test runs a small
    # match and asserts which case was taken, the variables bound, and that the subject is
    # unchanged; later tests check bindings via locals, guards, class patterns, protocols and
    # name mangling.

    def test_patma_000(self):
        # [FUNCTION DOC] Literal pattern 0 matches subject 0 and the body sets x to True.
        match 0:
            case 0:
                x = True
        self.assertIs(x, True)

    def test_patma_001(self):
        # [FUNCTION DOC] A first case whose guard is False is skipped; the second case with a
        # True guard runs.
        match 0:
            case 0 if False:
                x = False
            case 0 if True:
                x = True
        self.assertIs(x, True)

    def test_patma_002(self):
        # [FUNCTION DOC] Two identical literal cases: only the first one runs, so x is True.
        match 0:
            case 0:
                x = True
            case 0:
                x = False
        self.assertIs(x, True)

    def test_patma_003(self):
        # [FUNCTION DOC] Or-pattern of 0, 1, 2, 3 matches subject 0.
        x = False
        match 0:
            case 0 | 1 | 2 | 3:
                x = True
        self.assertIs(x, True)

    def test_patma_004(self):
        # [FUNCTION DOC] Or-pattern of 0, 1, 2, 3 matches subject 1.
        x = False
        match 1:
            case 0 | 1 | 2 | 3:
                x = True
        self.assertIs(x, True)

    def test_patma_005(self):
        # [FUNCTION DOC] Or-pattern of 0, 1, 2, 3 matches subject 2.
        x = False
        match 2:
            case 0 | 1 | 2 | 3:
                x = True
        self.assertIs(x, True)

    def test_patma_006(self):
        # [FUNCTION DOC] Or-pattern of 0, 1, 2, 3 matches subject 3.
        x = False
        match 3:
            case 0 | 1 | 2 | 3:
                x = True
        self.assertIs(x, True)

    def test_patma_007(self):
        # [FUNCTION DOC] Or-pattern of 0, 1, 2, 3 does not match subject 4, so x stays False.
        x = False
        match 4:
            case 0 | 1 | 2 | 3:
                x = True
        self.assertIs(x, False)

    def test_patma_008(self):
        # [FUNCTION DOC] Value pattern on a dotted name A.y (value 1) with an as-capture does
        # not match subject 0; no case body runs and neither x nor A.y change.
        x = 0
        class A:
            # [CLASS DOC] HELPER: local class holding class attribute y equal to 1, used as a
            # dotted value pattern.
            y = 1
        match x:
            case A.y as z:
                pass
        self.assertEqual(x, 0)
        self.assertEqual(A.y, 1)

    def test_patma_009(self):
        # [FUNCTION DOC] Subject 0 with a capture and falsy guard, then a wildcard-as with a
        # failing guard, then the dotted value pattern A.B which matches. Asserts names bound
        # by failed guards (x, y) stay bound and z is 2.
        class A:
            # [CLASS DOC] HELPER: local class holding class attribute B equal to 0, used as a
            # dotted value pattern.
            B = 0
        match 0:
            case x if x:
                z = 0
            case _ as y if y == x and y:
                z = 1
            case A.B:
                z = 2
        self.assertEqual(A.B, 0)
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)
        self.assertEqual(z, 2)

    def test_patma_010(self):
        # [FUNCTION DOC] The empty tuple matches an empty sequence pattern.
        match ():
            case []:
                x = 0
        self.assertEqual(x, 0)

    def test_patma_011(self):
        # [FUNCTION DOC] Star-only sequence pattern captures all of a tuple as a list.
        match (0, 1, 2):
            case [*x]:
                y = 0
        self.assertEqual(x, [0, 1, 2])
        self.assertEqual(y, 0)

    def test_patma_012(self):
        # [FUNCTION DOC] Literal then star: captures the remaining items 1 and 2 as a list.
        match (0, 1, 2):
            case [0, *x]:
                y = 0
        self.assertEqual(x, [1, 2])
        self.assertEqual(y, 0)

    def test_patma_013(self):
        # [FUNCTION DOC] Two literals then star with trailing comma: the star captures just
        # the last item.
        match (0, 1, 2):
            case [0, 1, *x,]:
                y = 0
        self.assertEqual(x, [2])
        self.assertEqual(y, 0)

    def test_patma_014(self):
        # [FUNCTION DOC] Literals consume the whole tuple so the trailing star captures an
        # empty list.
        match (0, 1, 2):
            case [0, 1, 2, *x]:
                y = 0
        self.assertEqual(x, [])
        self.assertEqual(y, 0)

    def test_patma_015(self):
        # [FUNCTION DOC] Star then final literal with trailing comma: captures the first two
        # items.
        match (0, 1, 2):
            case [*x, 2,]:
                y = 0
        self.assertEqual(x, [0, 1])
        self.assertEqual(y, 0)

    def test_patma_016(self):
        # [FUNCTION DOC] Star then two literals: captures the first item only.
        match (0, 1, 2):
            case [*x, 1, 2]:
                y = 0
        self.assertEqual(x, [0])
        self.assertEqual(y, 0)

    def test_patma_017(self):
        # [FUNCTION DOC] Star then three literals: star captures an empty list.
        match (0, 1, 2):
            case [*x, 0, 1, 2,]:
                y = 0
        self.assertEqual(x, [])
        self.assertEqual(y, 0)

    def test_patma_018(self):
        # [FUNCTION DOC] Star in the middle between literals captures the single middle item.
        match (0, 1, 2):
            case [0, *x, 2]:
                y = 0
        self.assertEqual(x, [1])
        self.assertEqual(y, 0)

    def test_patma_019(self):
        # [FUNCTION DOC] Star in the middle with trailing comma captures an empty list.
        match (0, 1, 2):
            case [0, 1, *x, 2,]:
                y = 0
        self.assertEqual(x, [])
        self.assertEqual(y, 0)

    def test_patma_020(self):
        # [FUNCTION DOC] Literal, star, then two literals: the star captures an empty list.
        match (0, 1, 2):
            case [0, *x, 1, 2]:
                y = 0
        self.assertEqual(x, [])
        self.assertEqual(y, 0)

    def test_patma_021(self):
        # [FUNCTION DOC] Star-only pattern with trailing comma captures the full tuple as a
        # list.
        match (0, 1, 2):
            case [*x,]:
                y = 0
        self.assertEqual(x, [0, 1, 2])
        self.assertEqual(y, 0)

    def test_patma_022(self):
        # [FUNCTION DOC] Empty mapping pattern matches an empty dict.
        x = {}
        match x:
            case {}:
                y = 0
        self.assertEqual(x, {})
        self.assertEqual(y, 0)

    def test_patma_023(self):
        # [FUNCTION DOC] Empty mapping pattern also matches a non-empty dict and leaves it
        # unchanged.
        x = {0: 0}
        match x:
            case {}:
                y = 0
        self.assertEqual(x, {0: 0})
        self.assertEqual(y, 0)

    def test_patma_024(self):
        # [FUNCTION DOC] Mapping pattern requiring key 0 does not match an empty dict; y stays
        # None.
        x = {}
        y = None
        match x:
            case {0: 0}:
                y = 0
        self.assertEqual(x, {})
        self.assertIs(y, None)

    def test_patma_025(self):
        # [FUNCTION DOC] Mapping with an or-pattern value and as-capture: value 0 matches and
        # z is 0.
        x = {0: 0}
        match x:
            case {0: (0 | 1 | 2 as z)}:
                y = 0
        self.assertEqual(x, {0: 0})
        self.assertEqual(y, 0)
        self.assertEqual(z, 0)

    def test_patma_026(self):
        # [FUNCTION DOC] Same mapping pattern with value 1; z is 1.
        x = {0: 1}
        match x:
            case {0: (0 | 1 | 2 as z)}:
                y = 0
        self.assertEqual(x, {0: 1})
        self.assertEqual(y, 0)
        self.assertEqual(z, 1)

    def test_patma_027(self):
        # [FUNCTION DOC] Same mapping pattern with value 2; z is 2.
        x = {0: 2}
        match x:
            case {0: (0 | 1 | 2 as z)}:
                y = 0
        self.assertEqual(x, {0: 2})
        self.assertEqual(y, 0)
        self.assertEqual(z, 2)

    def test_patma_028(self):
        # [FUNCTION DOC] Same mapping pattern with value 3 does not match; y stays None.
        x = {0: 3}
        y = None
        match x:
            case {0: (0 | 1 | 2 as z)}:
                y = 0
        self.assertEqual(x, {0: 3})
        self.assertIs(y, None)

    def test_patma_029(self):
        # [FUNCTION DOC] An empty dict fails all three cases (required keys missing, and a
        # sequence pattern never matches a mapping); y stays None.
        x = {}
        y = None
        match x:
            case {0: [1, 2, {}]}:
                y = 0
            case {0: [1, 2, {}], 1: [[]]}:
                y = 1
            case []:
                y = 2
        self.assertEqual(x, {})
        self.assertIs(y, None)

    def test_patma_030(self):
        # [FUNCTION DOC] Mapping with key False and a tuple value matches the first case
        # because False equals 0, True equals 1 and 2.0 equals 2; y is 0.
        x = {False: (True, 2.0, {})}
        match x:
            case {0: [1, 2, {}]}:
                y = 0
            case {0: [1, 2, {}], 1: [[]]}:
                y = 1
            case []:
                y = 2
        self.assertEqual(x, {False: (True, 2.0, {})})
        self.assertEqual(y, 0)

    def test_patma_031(self):
        # [FUNCTION DOC] Mapping with extra keys still matches the first case, since mapping
        # patterns ignore unlisted keys.
        x = {False: (True, 2.0, {}), 1: [[]], 2: 0}
        match x:
            case {0: [1, 2, {}]}:
                y = 0
            case {0: [1, 2, {}], 1: [[]]}:
                y = 1
            case []:
                y = 2
        self.assertEqual(x, {False: (True, 2.0, {}), 1: [[]], 2: 0})
        self.assertEqual(y, 0)

    def test_patma_032(self):
        # [FUNCTION DOC] First case fails on sequence length (two items versus three); second
        # case needing key 1 as well matches, y is 1.
        x = {False: (True, 2.0, {}), 1: [[]], 2: 0}
        match x:
            case {0: [1, 2]}:
                y = 0
            case {0: [1, 2, {}], 1: [[]]}:
                y = 1
            case []:
                y = 2
        self.assertEqual(x, {False: (True, 2.0, {}), 1: [[]], 2: 0})
        self.assertEqual(y, 1)

    def test_patma_033(self):
        # [FUNCTION DOC] An empty list does not match mapping patterns but matches the empty
        # sequence pattern; y is 2.
        x = []
        match x:
            case {0: [1, 2, {}]}:
                y = 0
            case {0: [1, 2, {}], 1: [[]]}:
                y = 1
            case []:
                y = 2
        self.assertEqual(x, [])
        self.assertEqual(y, 2)

    def test_patma_034(self):
        # [FUNCTION DOC] Subject {0: 0} fails the first case; the second case's or-pattern
        # matches via its first alternative where value 0 equals False; y is 1.
        x = {0: 0}
        match x:
            case {0: [1, 2, {}]}:
                y = 0
            case {0: ([1, 2, {}] | False)} | {1: [[]]} | {0: [1, 2, {}]} | [] | "X" | {}:
                y = 1
            case []:
                y = 2
        self.assertEqual(x, {0: 0})
        self.assertEqual(y, 1)

    def test_patma_035(self):
        # [FUNCTION DOC] Same shape with True in place of False; value 0 does not equal True,
        # so the match comes from the last alternative, the empty mapping pattern; y is 1.
        x = {0: 0}
        match x:
            case {0: [1, 2, {}]}:
                y = 0
            case {0: [1, 2, {}] | True} | {1: [[]]} | {0: [1, 2, {}]} | [] | "X" | {}:
                y = 1
            case []:
                y = 2
        self.assertEqual(x, {0: 0})
        self.assertEqual(y, 1)

    def test_patma_036(self):
        # [FUNCTION DOC] Or-pattern 0, 1, 2 matches subject 0.
        x = 0
        match x:
            case 0 | 1 | 2:
                y = 0
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_037(self):
        # [FUNCTION DOC] Or-pattern 0, 1, 2 matches subject 1.
        x = 1
        match x:
            case 0 | 1 | 2:
                y = 0
        self.assertEqual(x, 1)
        self.assertEqual(y, 0)

    def test_patma_038(self):
        # [FUNCTION DOC] Or-pattern 0, 1, 2 matches subject 2.
        x = 2
        match x:
            case 0 | 1 | 2:
                y = 0
        self.assertEqual(x, 2)
        self.assertEqual(y, 0)

    def test_patma_039(self):
        # [FUNCTION DOC] Or-pattern 0, 1, 2 does not match 3; y stays None.
        x = 3
        y = None
        match x:
            case 0 | 1 | 2:
                y = 0
        self.assertEqual(x, 3)
        self.assertIs(y, None)

    def test_patma_040(self):
        # [FUNCTION DOC] Or-pattern with as-captures and a guard comparing z to subject modulo
        # 2; subject 0 matches, y and z are 0.
        x = 0
        match x:
            case (0 as z) | (1 as z) | (2 as z) if z == x % 2:
                y = 0
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)
        self.assertEqual(z, 0)

    def test_patma_041(self):
        # [FUNCTION DOC] Same pattern with subject 1; guard holds, z is 1.
        x = 1
        match x:
            case (0 as z) | (1 as z) | (2 as z) if z == x % 2:
                y = 0
        self.assertEqual(x, 1)
        self.assertEqual(y, 0)
        self.assertEqual(z, 1)

    def test_patma_042(self):
        # [FUNCTION DOC] Subject 2: the pattern binds z to 2 but the guard fails, so y stays
        # None while z remains 2.
        x = 2
        y = None
        match x:
            case (0 as z) | (1 as z) | (2 as z) if z == x % 2:
                y = 0
        self.assertEqual(x, 2)
        self.assertIs(y, None)
        self.assertEqual(z, 2)

    def test_patma_043(self):
        # [FUNCTION DOC] Subject 3 matches no alternative; y stays None.
        x = 3
        y = None
        match x:
            case (0 as z) | (1 as z) | (2 as z) if z == x % 2:
                y = 0
        self.assertEqual(x, 3)
        self.assertIs(y, None)

    def test_patma_044(self):
        # [FUNCTION DOC] A tuple () matches the list-style empty sequence pattern.
        x = ()
        match x:
            case []:
                y = 0
        self.assertEqual(x, ())
        self.assertEqual(y, 0)

    def test_patma_045(self):
        # [FUNCTION DOC] A tuple () matches the tuple-style empty sequence pattern.
        x = ()
        match x:
            case ():
                y = 0
        self.assertEqual(x, ())
        self.assertEqual(y, 0)

    def test_patma_046(self):
        # [FUNCTION DOC] A one-item tuple matches a one-item sequence pattern.
        x = (0,)
        match x:
            case [0]:
                y = 0
        self.assertEqual(x, (0,))
        self.assertEqual(y, 0)

    def test_patma_047(self):
        # [FUNCTION DOC] A tuple containing an empty tuple matches a nested empty sequence
        # pattern.
        x = ((),)
        match x:
            case [[]]:
                y = 0
        self.assertEqual(x, ((),))
        self.assertEqual(y, 0)

    def test_patma_048(self):
        # [FUNCTION DOC] Or of two sequence patterns; list [0, 1] matches the first.
        x = [0, 1]
        match x:
            case [0, 1] | [1, 0]:
                y = 0
        self.assertEqual(x, [0, 1])
        self.assertEqual(y, 0)

    def test_patma_049(self):
        # [FUNCTION DOC] List [1, 0] matches the second alternative.
        x = [1, 0]
        match x:
            case [0, 1] | [1, 0]:
                y = 0
        self.assertEqual(x, [1, 0])
        self.assertEqual(y, 0)

    def test_patma_050(self):
        # [FUNCTION DOC] List [0, 0] matches neither alternative; y stays None.
        x = [0, 0]
        y = None
        match x:
            case [0, 1] | [1, 0]:
                y = 0
        self.assertEqual(x, [0, 0])
        self.assertIs(y, None)

    def test_patma_051(self):
        # [FUNCTION DOC] The first case fails on length before binding w, so w stays None; the
        # second case's alternatives bind z, with the matching one setting z to 0; y is 1.
        w = None
        x = [1, 0]
        match x:
            case [(0 as w)]:
                y = 0
            case [z] | [1, (0 | 1 as z)] | [z]:
                y = 1
        self.assertIs(w, None)
        self.assertEqual(x, [1, 0])
        self.assertEqual(y, 1)
        self.assertEqual(z, 0)

    def test_patma_052(self):
        # [FUNCTION DOC] A guard rebinds the subject name to an empty list (falsy), so the
        # guard fails and the next case matches the original subject; x ends as [] and y is 2.
        x = [1, 0]
        match x:
            case [0]:
                y = 0
            case [1, 0] if (x := x[:0]):
                y = 1
            case [1, 0]:
                y = 2
        self.assertEqual(x, [])
        self.assertEqual(y, 2)

    def test_patma_053(self):
        # [FUNCTION DOC] A set is not a sequence, so a sequence pattern does not match; y
        # stays None.
        x = {0}
        y = None
        match x:
            case [0]:
                y = 0
        self.assertEqual(x, {0})
        self.assertIs(y, None)

    def test_patma_054(self):
        # [FUNCTION DOC] An empty set does not match the empty sequence pattern.
        x = set()
        y = None
        match x:
            case []:
                y = 0
        self.assertEqual(x, set())
        self.assertIs(y, None)

    def test_patma_055(self):
        # [FUNCTION DOC] An iterator is not matched by a sequence pattern and is not consumed
        # (it still yields 1, 2, 3 afterwards).
        x = iter([1, 2, 3])
        y = None
        match x:
            case []:
                y = 0
        self.assertEqual([*x], [1, 2, 3])
        self.assertIs(y, None)

    def test_patma_056(self):
        # [FUNCTION DOC] An empty dict does not match the empty sequence pattern.
        x = {}
        y = None
        match x:
            case []:
                y = 0
        self.assertEqual(x, {})
        self.assertIs(y, None)

    def test_patma_057(self):
        # [FUNCTION DOC] A dict does not match a sequence pattern even with keys 0 and 1.
        x = {0: False, 1: True}
        y = None
        match x:
            case [0, 1]:
                y = 0
        self.assertEqual(x, {0: False, 1: True})
        self.assertIs(y, None)

    def test_patma_058(self):
        # [FUNCTION DOC] Literal 0 matches subject 0.
        x = 0
        match x:
            case 0:
                y = 0
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_059(self):
        # [FUNCTION DOC] The False pattern compares by identity, so integer 0 does not match;
        # y stays None.
        x = 0
        y = None
        match x:
            case False:
                y = 0
        self.assertEqual(x, 0)
        self.assertEqual(y, None)

    def test_patma_060(self):
        # [FUNCTION DOC] Literal 1 does not match subject 0.
        x = 0
        y = None
        match x:
            case 1:
                y = 0
        self.assertEqual(x, 0)
        self.assertIs(y, None)

    def test_patma_061(self):
        # [FUNCTION DOC] The None pattern does not match subject 0.
        x = 0
        y = None
        match x:
            case None:
                y = 0
        self.assertEqual(x, 0)
        self.assertIs(y, None)

    def test_patma_062(self):
        # [FUNCTION DOC] Two identical literal cases; the first wins.
        x = 0
        match x:
            case 0:
                y = 0
            case 0:
                y = 1
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_063(self):
        # [FUNCTION DOC] Two non-matching identical cases; nothing runs.
        x = 0
        y = None
        match x:
            case 1:
                y = 0
            case 1:
                y = 1
        self.assertEqual(x, 0)
        self.assertIs(y, None)

    def test_patma_064(self):
        # [FUNCTION DOC] String literal cases; "x" selects the first.
        x = "x"
        match x:
            case "x":
                y = 0
            case "y":
                y = 1
        self.assertEqual(x, "x")
        self.assertEqual(y, 0)

    def test_patma_065(self):
        # [FUNCTION DOC] String literal cases in other order; the second matches.
        x = "x"
        match x:
            case "y":
                y = 0
            case "x":
                y = 1
        self.assertEqual(x, "x")
        self.assertEqual(y, 1)

    def test_patma_066(self):
        # [FUNCTION DOC] Empty-string case does not match "x"; the "x" case does.
        x = "x"
        match x:
            case "":
                y = 0
            case "x":
                y = 1
        self.assertEqual(x, "x")
        self.assertEqual(y, 1)

    def test_patma_067(self):
        # [FUNCTION DOC] Bytes literal cases select the one equal to b"x".
        x = b"x"
        match x:
            case b"y":
                y = 0
            case b"x":
                y = 1
        self.assertEqual(x, b"x")
        self.assertEqual(y, 1)

    def test_patma_068(self):
        # [FUNCTION DOC] A False-guarded case is skipped; the next case with the same literal
        # runs.
        x = 0
        match x:
            case 0 if False:
                y = 0
            case 0:
                y = 1
        self.assertEqual(x, 0)
        self.assertEqual(y, 1)

    def test_patma_069(self):
        # [FUNCTION DOC] Both cases have falsy numeric guards (0), so none run.
        x = 0
        y = None
        match x:
            case 0 if 0:
                y = 0
            case 0 if 0:
                y = 1
        self.assertEqual(x, 0)
        self.assertIs(y, None)

    def test_patma_070(self):
        # [FUNCTION DOC] Two True-guarded cases: the first runs.
        x = 0
        match x:
            case 0 if True:
                y = 0
            case 0 if True:
                y = 1
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_071(self):
        # [FUNCTION DOC] Two cases with truthy numeric guard 1: the first runs.
        x = 0
        match x:
            case 0 if 1:
                y = 0
            case 0 if 1:
                y = 1
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_072(self):
        # [FUNCTION DOC] After a matched case, code following the match statement overwrites y
        # with 2.
        x = 0
        match x:
            case 0 if True:
                y = 0
            case 0 if True:
                y = 1
        y = 2
        self.assertEqual(x, 0)
        self.assertEqual(y, 2)

    def test_patma_073(self):
        # [FUNCTION DOC] The first guard (0) fails and the second (1) passes; y is later
        # overwritten by 2.
        x = 0
        match x:
            case 0 if 0:
                y = 0
            case 0 if 1:
                y = 1
        y = 2
        self.assertEqual(x, 0)
        self.assertEqual(y, 2)

    def test_patma_074(self):
        # [FUNCTION DOC] A guard assigns x via assignment expression and is falsy; the next
        # literal case still compares against the original subject 0 and fails, so y stays
        # None while x is 1.
        x = 0
        y = None
        match x:
            case 0 if not (x := 1):
                y = 0
            case 1:
                y = 1
        self.assertEqual(x, 1)
        self.assertIs(y, None)

    def test_patma_075(self):
        # [FUNCTION DOC] A string is not matched by a one-item sequence pattern but matches
        # the string literal case.
        x = "x"
        match x:
            case ["x"]:
                y = 0
            case "x":
                y = 1
        self.assertEqual(x, "x")
        self.assertEqual(y, 1)

    def test_patma_076(self):
        # [FUNCTION DOC] bytes are not matched by sequence patterns (neither of bytes, string,
        # nor integer items); only the bytes literal case matches, y is 4.
        x = b"x"
        match x:
            case [b"x"]:
                y = 0
            case ["x"]:
                y = 1
            case [120]:
                y = 2
            case b"x":
                y = 4
        self.assertEqual(x, b"x")
        self.assertEqual(y, 4)

    def test_patma_077(self):
        # [FUNCTION DOC] A bytearray is not matched by the sequence pattern nor by the integer
        # literal 120; y stays None.
        x = bytearray(b"x")
        y = None
        match x:
            case [120]:
                y = 0
            case 120:
                y = 1
        self.assertEqual(x, b"x")
        self.assertIs(y, None)

    def test_patma_078(self):
        # [FUNCTION DOC] An empty string does not match sequence patterns; the literal "" case
        # runs.
        x = ""
        match x:
            case []:
                y = 0
            case [""]:
                y = 1
            case "":
                y = 2
        self.assertEqual(x, "")
        self.assertEqual(y, 2)

    def test_patma_079(self):
        # [FUNCTION DOC] String "xxx" is not split into items; only the literal case matches.
        x = "xxx"
        match x:
            case ["x", "x", "x"]:
                y = 0
            case ["xxx"]:
                y = 1
            case "xxx":
                y = 2
        self.assertEqual(x, "xxx")
        self.assertEqual(y, 2)

    def test_patma_080(self):
        # [FUNCTION DOC] Bytes b"xxx" does not match sequence patterns; the bytes literal case
        # matches.
        x = b"xxx"
        match x:
            case [120, 120, 120]:
                y = 0
            case [b"xxx"]:
                y = 1
            case b"xxx":
                y = 2
        self.assertEqual(x, b"xxx")
        self.assertEqual(y, 2)

    def test_patma_081(self):
        # [FUNCTION DOC] A guard rebinds x and fails; the next case with an as-capture matches
        # the original subject so x is 1, y is 1, z is 0.
        x = 0
        match x:
            case 0 if not (x := 1):
                y = 0
            case (0 as z):
                y = 1
        self.assertEqual(x, 1)
        self.assertEqual(y, 1)
        self.assertEqual(z, 0)

    def test_patma_082(self):
        # [FUNCTION DOC] The pattern (1 as z) fails to match before its guard is evaluated, so
        # x is never rebound; the second case runs.
        x = 0
        match x:
            case (1 as z) if not (x := 1):
                y = 0
            case 0:
                y = 1
        self.assertEqual(x, 0)
        self.assertEqual(y, 1)

    def test_patma_083(self):
        # [FUNCTION DOC] A literal pattern with an as-capture binds z to 0.
        x = 0
        match x:
            case (0 as z):
                y = 0
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)
        self.assertEqual(z, 0)

    def test_patma_084(self):
        # [FUNCTION DOC] A failing literal with as-capture does not match; y stays None.
        x = 0
        y = None
        match x:
            case (1 as z):
                y = 0
        self.assertEqual(x, 0)
        self.assertIs(y, None)

    def test_patma_085(self):
        # [FUNCTION DOC] The pattern binds z, then the guard assigns w and is falsy; y stays
        # None but z and w stay bound to 0.
        x = 0
        y = None
        match x:
            case (0 as z) if (w := 0):
                y = 0
        self.assertEqual(w, 0)
        self.assertEqual(x, 0)
        self.assertIs(y, None)
        self.assertEqual(z, 0)

    def test_patma_086(self):
        # [FUNCTION DOC] Nested as-captures bind both w and z to the subject.
        x = 0
        match x:
            case ((0 as w) as z):
                y = 0
        self.assertEqual(w, 0)
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)
        self.assertEqual(z, 0)

    def test_patma_087(self):
        # [FUNCTION DOC] Parenthesized or-pattern grouping at the left; matches 0.
        x = 0
        match x:
            case (0 | 1) | 2:
                y = 0
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_088(self):
        # [FUNCTION DOC] Same grouping matches 1.
        x = 1
        match x:
            case (0 | 1) | 2:
                y = 0
        self.assertEqual(x, 1)
        self.assertEqual(y, 0)

    def test_patma_089(self):
        # [FUNCTION DOC] Same grouping matches 2.
        x = 2
        match x:
            case (0 | 1) | 2:
                y = 0
        self.assertEqual(x, 2)
        self.assertEqual(y, 0)

    def test_patma_090(self):
        # [FUNCTION DOC] Same grouping does not match 3.
        x = 3
        y = None
        match x:
            case (0 | 1) | 2:
                y = 0
        self.assertEqual(x, 3)
        self.assertIs(y, None)

    def test_patma_091(self):
        # [FUNCTION DOC] Parenthesized or-pattern grouping at the right; matches 0.
        x = 0
        match x:
            case 0 | (1 | 2):
                y = 0
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_092(self):
        # [FUNCTION DOC] Right grouping matches 1.
        x = 1
        match x:
            case 0 | (1 | 2):
                y = 0
        self.assertEqual(x, 1)
        self.assertEqual(y, 0)

    def test_patma_093(self):
        # [FUNCTION DOC] Right grouping matches 2.
        x = 2
        match x:
            case 0 | (1 | 2):
                y = 0
        self.assertEqual(x, 2)
        self.assertEqual(y, 0)

    def test_patma_094(self):
        # [FUNCTION DOC] Right grouping does not match 3.
        x = 3
        y = None
        match x:
            case 0 | (1 | 2):
                y = 0
        self.assertEqual(x, 3)
        self.assertIs(y, None)

    def test_patma_095(self):
        # [FUNCTION DOC] Literal -0 matches subject 0.
        x = 0
        match x:
            case -0:
                y = 0
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_096(self):
        # [FUNCTION DOC] Literal -0.0 matches subject 0.
        x = 0
        match x:
            case -0.0:
                y = 0
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_097(self):
        # [FUNCTION DOC] Literal -0j matches subject 0.
        x = 0
        match x:
            case -0j:
                y = 0
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_098(self):
        # [FUNCTION DOC] Literal -0.0j matches subject 0.
        x = 0
        match x:
            case -0.0j:
                y = 0
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_099(self):
        # [FUNCTION DOC] Negative integer literal matches -1.
        x = -1
        match x:
            case -1:
                y = 0
        self.assertEqual(x, -1)
        self.assertEqual(y, 0)

    def test_patma_100(self):
        # [FUNCTION DOC] Negative float literal matches -1.5.
        x = -1.5
        match x:
            case -1.5:
                y = 0
        self.assertEqual(x, -1.5)
        self.assertEqual(y, 0)

    def test_patma_101(self):
        # [FUNCTION DOC] Negative imaginary literal matches -1j.
        x = -1j
        match x:
            case -1j:
                y = 0
        self.assertEqual(x, -1j)
        self.assertEqual(y, 0)

    def test_patma_102(self):
        # [FUNCTION DOC] Negative imaginary float literal matches -1.5j.
        x = -1.5j
        match x:
            case -1.5j:
                y = 0
        self.assertEqual(x, -1.5j)
        self.assertEqual(y, 0)

    def test_patma_103(self):
        # [FUNCTION DOC] Complex literal 0 + 0j matches subject 0.
        x = 0
        match x:
            case 0 + 0j:
                y = 0
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_104(self):
        # [FUNCTION DOC] Complex literal 0 - 0j matches subject 0.
        x = 0
        match x:
            case 0 - 0j:
                y = 0
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_105(self):
        # [FUNCTION DOC] Complex literal -0 + 0j matches subject 0.
        x = 0
        match x:
            case -0 + 0j:
                y = 0
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_106(self):
        # [FUNCTION DOC] Complex literal -0 - 0j matches subject 0.
        x = 0
        match x:
            case -0 - 0j:
                y = 0
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_107(self):
        # [FUNCTION DOC] Complex literal with positive real and imaginary parts matches an
        # equal value.
        x = 0.25 + 1.75j
        match x:
            case 0.25 + 1.75j:
                y = 0
        self.assertEqual(x, 0.25 + 1.75j)
        self.assertEqual(y, 0)

    def test_patma_108(self):
        # [FUNCTION DOC] Complex literal with positive real and negative imaginary part
        # matches an equal value.
        x = 0.25 - 1.75j
        match x:
            case 0.25 - 1.75j:
                y = 0
        self.assertEqual(x, 0.25 - 1.75j)
        self.assertEqual(y, 0)

    def test_patma_109(self):
        # [FUNCTION DOC] Complex literal with negative real and positive imaginary part
        # matches an equal value.
        x = -0.25 + 1.75j
        match x:
            case -0.25 + 1.75j:
                y = 0
        self.assertEqual(x, -0.25 + 1.75j)
        self.assertEqual(y, 0)

    def test_patma_110(self):
        # [FUNCTION DOC] Complex literal with both parts negative matches an equal value.
        x = -0.25 - 1.75j
        match x:
            case -0.25 - 1.75j:
                y = 0
        self.assertEqual(x, -0.25 - 1.75j)
        self.assertEqual(y, 0)

    def test_patma_111(self):
        # [FUNCTION DOC] Dotted value pattern A.B matches 0.
        class A:
            # [CLASS DOC] HELPER: local class holding class attribute B equal to 0.
            B = 0
        x = 0
        match x:
            case A.B:
                y = 0
        self.assertEqual(A.B, 0)
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_112(self):
        # [FUNCTION DOC] Two-level dotted value pattern A.B.C matches 0.
        class A:
            # [CLASS DOC] HELPER: local namespace class containing nested class B.
            class B:
                # [CLASS DOC] HELPER: nested namespace class holding C equal to 0.
                C = 0
        x = 0
        match x:
            case A.B.C:
                y = 0
        self.assertEqual(A.B.C, 0)
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_113(self):
        # [FUNCTION DOC] Two dotted cases; the second, A.B.D (value 1), matches subject 1.
        class A:
            # [CLASS DOC] HELPER: local namespace class containing nested class B.
            class B:
                # [CLASS DOC] HELPER: nested namespace class holding C equal to 0 and D equal
                # to 1.
                C = 0
                D = 1
        x = 1
        match x:
            case A.B.C:
                y = 0
            case A.B.D:
                y = 1
        self.assertEqual(A.B.C, 0)
        self.assertEqual(A.B.D, 1)
        self.assertEqual(x, 1)
        self.assertEqual(y, 1)

    def test_patma_114(self):
        # [FUNCTION DOC] Three-level dotted value pattern matches 0.
        class A:
            # [CLASS DOC] HELPER: local namespace class containing nested class B.
            class B:
                # [CLASS DOC] HELPER: nested namespace class containing nested class C.
                class C:
                    # [CLASS DOC] HELPER: innermost class holding D equal to 0.
                    D = 0
        x = 0
        match x:
            case A.B.C.D:
                y = 0
        self.assertEqual(A.B.C.D, 0)
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_115(self):
        # [FUNCTION DOC] Two three-level dotted cases; the second matches subject 1.
        class A:
            # [CLASS DOC] HELPER: local namespace class containing nested class B.
            class B:
                # [CLASS DOC] HELPER: nested namespace class containing nested class C.
                class C:
                    # [CLASS DOC] HELPER: innermost class holding D equal to 0 and E equal to
                    # 1.
                    D = 0
                    E = 1
        x = 1
        match x:
            case A.B.C.D:
                y = 0
            case A.B.C.E:
                y = 1
        self.assertEqual(A.B.C.D, 0)
        self.assertEqual(A.B.C.E, 1)
        self.assertEqual(x, 1)
        self.assertEqual(y, 1)

    def test_patma_116(self):
        # [FUNCTION DOC] The soft keywords match and case are used as ordinary variable names,
        # both as subject and as capture.
        match = case = 0
        match match:
            case case:
                x = 0
        self.assertEqual(match, 0)
        self.assertEqual(case, 0)
        self.assertEqual(x, 0)

    def test_patma_117(self):
        # [FUNCTION DOC] Same with subject and capture roles swapped (case as subject, match
        # as capture).
        match = case = 0
        match case:
            case match:
                x = 0
        self.assertEqual(match, 0)
        self.assertEqual(case, 0)
        self.assertEqual(x, 0)

    def test_patma_118(self):
        # [FUNCTION DOC] A sequence pattern needing at least one item does not match an empty
        # list; the empty pattern does.
        x = []
        match x:
            case [*_, _]:
                y = 0
            case []:
                y = 1
        self.assertEqual(x, [])
        self.assertEqual(y, 1)

    def test_patma_119(self):
        # [FUNCTION DOC] Mapping pattern on an empty defaultdict does not trigger the default
        # factory; falls through to the empty pattern and the dict stays empty.
        x = collections.defaultdict(int)
        match x:
            case {0: 0}:
                y = 0
            case {}:
                y = 1
        self.assertEqual(x, {})
        self.assertEqual(y, 1)

    def test_patma_120(self):
        # [FUNCTION DOC] Same with a rest capture; the capture is an empty dict and the
        # defaultdict is not populated.
        x = collections.defaultdict(int)
        match x:
            case {0: 0}:
                y = 0
            case {**z}:
                y = 1
        self.assertEqual(x, {})
        self.assertEqual(y, 1)
        self.assertEqual(z, {})

    def test_patma_121(self):
        # [FUNCTION DOC] Parenthesized empty tuple pattern matches an empty tuple.
        match ():
            case ():
                x = 0
        self.assertEqual(x, 0)

    def test_patma_122(self):
        # [FUNCTION DOC] Parenthesized star-only pattern captures all items as a list.
        match (0, 1, 2):
            case (*x,):
                y = 0
        self.assertEqual(x, [0, 1, 2])
        self.assertEqual(y, 0)

    def test_patma_123(self):
        # [FUNCTION DOC] Unparenthesized literal then star captures the rest.
        match (0, 1, 2):
            case 0, *x:
                y = 0
        self.assertEqual(x, [1, 2])
        self.assertEqual(y, 0)

    def test_patma_124(self):
        # [FUNCTION DOC] Parenthesized two literals then star with trailing comma captures the
        # last item.
        match (0, 1, 2):
            case (0, 1, *x,):
                y = 0
        self.assertEqual(x, [2])
        self.assertEqual(y, 0)

    def test_patma_125(self):
        # [FUNCTION DOC] Unparenthesized all literals then star captures an empty list.
        match (0, 1, 2):
            case 0, 1, 2, *x:
                y = 0
        self.assertEqual(x, [])
        self.assertEqual(y, 0)

    def test_patma_126(self):
        # [FUNCTION DOC] Unparenthesized star then literal with trailing comma captures the
        # first two items.
        match (0, 1, 2):
            case *x, 2,:
                y = 0
        self.assertEqual(x, [0, 1])
        self.assertEqual(y, 0)

    def test_patma_127(self):
        # [FUNCTION DOC] Parenthesized star then two literals captures the first item.
        match (0, 1, 2):
            case (*x, 1, 2):
                y = 0
        self.assertEqual(x, [0])
        self.assertEqual(y, 0)

    def test_patma_128(self):
        # [FUNCTION DOC] Unparenthesized star then three literals captures an empty list.
        match (0, 1, 2):
            case *x, 0, 1, 2,:
                y = 0
        self.assertEqual(x, [])
        self.assertEqual(y, 0)

    def test_patma_129(self):
        # [FUNCTION DOC] Parenthesized literal, star, literal captures the middle item.
        match (0, 1, 2):
            case (0, *x, 2):
                y = 0
        self.assertEqual(x, [1])
        self.assertEqual(y, 0)

    def test_patma_130(self):
        # [FUNCTION DOC] Unparenthesized literals around a star capture an empty list.
        match (0, 1, 2):
            case 0, 1, *x, 2,:
                y = 0
        self.assertEqual(x, [])
        self.assertEqual(y, 0)

    def test_patma_131(self):
        # [FUNCTION DOC] Parenthesized literal, star, two literals captures an empty list.
        match (0, 1, 2):
            case (0, *x, 1, 2):
                y = 0
        self.assertEqual(x, [])
        self.assertEqual(y, 0)

    def test_patma_132(self):
        # [FUNCTION DOC] Unparenthesized star with trailing comma captures the whole tuple.
        match (0, 1, 2):
            case *x,:
                y = 0
        self.assertEqual(x, [0, 1, 2])
        self.assertEqual(y, 0)

    def test_patma_133(self):
        # [FUNCTION DOC] On a defaultdict with key 0 mapped to 1: key 1 missing and value
        # mismatch fail, the empty pattern matches; the dict is not extended.
        x = collections.defaultdict(int, {0: 1})
        match x:
            case {1: 0}:
                y = 0
            case {0: 0}:
                y = 1
            case {}:
                y = 2
        self.assertEqual(x, {0: 1})
        self.assertEqual(y, 2)

    def test_patma_134(self):
        # [FUNCTION DOC] Same defaultdict, falling to a rest-capture case; the capture is a
        # dict equal to the full mapping.
        x = collections.defaultdict(int, {0: 1})
        match x:
            case {1: 0}:
                y = 0
            case {0: 0}:
                y = 1
            case {**z}:
                y = 2
        self.assertEqual(x, {0: 1})
        self.assertEqual(y, 2)
        self.assertEqual(z, {0: 1})

    def test_patma_135(self):
        # [FUNCTION DOC] Same defaultdict with key 0 matched by wildcard plus rest capture;
        # the rest is empty.
        x = collections.defaultdict(int, {0: 1})
        match x:
            case {1: 0}:
                y = 0
            case {0: 0}:
                y = 1
            case {0: _, **z}:
                y = 2
        self.assertEqual(x, {0: 1})
        self.assertEqual(y, 2)
        self.assertEqual(z, {})

    def test_patma_136(self):
        # [FUNCTION DOC] Plain dict version: the empty mapping pattern is the first match.
        x = {0: 1}
        match x:
            case {1: 0}:
                y = 0
            case {0: 0}:
                y = 0
            case {}:
                y = 1
        self.assertEqual(x, {0: 1})
        self.assertEqual(y, 1)

    def test_patma_137(self):
        # [FUNCTION DOC] Plain dict version with a rest capture equal to the whole dict.
        x = {0: 1}
        match x:
            case {1: 0}:
                y = 0
            case {0: 0}:
                y = 0
            case {**z}:
                y = 1
        self.assertEqual(x, {0: 1})
        self.assertEqual(y, 1)
        self.assertEqual(z, {0: 1})

    def test_patma_138(self):
        # [FUNCTION DOC] Plain dict version with key 0 and a rest capture that ends up empty.
        x = {0: 1}
        match x:
            case {1: 0}:
                y = 0
            case {0: 0}:
                y = 0
            case {0: _, **z}:
                y = 1
        self.assertEqual(x, {0: 1})
        self.assertEqual(y, 1)
        self.assertEqual(z, {})

    def test_patma_139(self):
        # [FUNCTION DOC] Class pattern bool(z) with a single positional subpattern binds z to
        # the subject itself (False).
        x = False
        match x:
            case bool(z):
                y = 0
        self.assertIs(x, False)
        self.assertEqual(y, 0)
        self.assertIs(z, x)

    def test_patma_140(self):
        # [FUNCTION DOC] Same class pattern with True.
        x = True
        match x:
            case bool(z):
                y = 0
        self.assertIs(x, True)
        self.assertEqual(y, 0)
        self.assertIs(z, x)

    def test_patma_141(self):
        # [FUNCTION DOC] Same self-binding behavior for bytearray.
        x = bytearray()
        match x:
            case bytearray(z):
                y = 0
        self.assertEqual(x, bytearray())
        self.assertEqual(y, 0)
        self.assertIs(z, x)

    def test_patma_142(self):
        # [FUNCTION DOC] Same self-binding behavior for bytes.
        x = b""
        match x:
            case bytes(z):
                y = 0
        self.assertEqual(x, b"")
        self.assertEqual(y, 0)
        self.assertIs(z, x)

    def test_patma_143(self):
        # [FUNCTION DOC] Same self-binding behavior for dict.
        x = {}
        match x:
            case dict(z):
                y = 0
        self.assertEqual(x, {})
        self.assertEqual(y, 0)
        self.assertIs(z, x)

    def test_patma_144(self):
        # [FUNCTION DOC] Same self-binding behavior for float.
        x = 0.0
        match x:
            case float(z):
                y = 0
        self.assertEqual(x, 0.0)
        self.assertEqual(y, 0)
        self.assertIs(z, x)

    def test_patma_145(self):
        # [FUNCTION DOC] Same self-binding behavior for frozenset.
        x = frozenset()
        match x:
            case frozenset(z):
                y = 0
        self.assertEqual(x, frozenset())
        self.assertEqual(y, 0)
        self.assertIs(z, x)

    def test_patma_146(self):
        # [FUNCTION DOC] Same self-binding behavior for int.
        x = 0
        match x:
            case int(z):
                y = 0
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)
        self.assertIs(z, x)

    def test_patma_147(self):
        # [FUNCTION DOC] Same self-binding behavior for list.
        x = []
        match x:
            case list(z):
                y = 0
        self.assertEqual(x, [])
        self.assertEqual(y, 0)
        self.assertIs(z, x)

    def test_patma_148(self):
        # [FUNCTION DOC] Same self-binding behavior for set.
        x = set()
        match x:
            case set(z):
                y = 0
        self.assertEqual(x, set())
        self.assertEqual(y, 0)
        self.assertIs(z, x)

    def test_patma_149(self):
        # [FUNCTION DOC] Same self-binding behavior for str.
        x = ""
        match x:
            case str(z):
                y = 0
        self.assertEqual(x, "")
        self.assertEqual(y, 0)
        self.assertIs(z, x)

    def test_patma_150(self):
        # [FUNCTION DOC] Same self-binding behavior for tuple.
        x = ()
        match x:
            case tuple(z):
                y = 0
        self.assertEqual(x, ())
        self.assertEqual(y, 0)
        self.assertIs(z, x)

    def test_patma_151(self):
        # [FUNCTION DOC] One-item tuple subject built with a trailing comma matched by a one-
        # item pattern; y is the original item.
        x = 0
        match x,:
            case y,:
                z = 0
        self.assertEqual(x, 0)
        self.assertIs(y, x)
        self.assertIs(z, 0)

    def test_patma_152(self):
        # [FUNCTION DOC] Tuple subject of two names matched by a two-capture pattern.
        w = 0
        x = 0
        match w, x:
            case y, z:
                v = 0
        self.assertEqual(w, 0)
        self.assertEqual(x, 0)
        self.assertIs(y, w)
        self.assertIs(z, x)
        self.assertEqual(v, 0)

    def test_patma_153(self):
        # [FUNCTION DOC] The subject is a tuple whose item is an assignment expression binding
        # w; the pattern uses as to bind v; checks the identities of w, y and v.
        x = 0
        match w := x,:
            case y as v,:
                z = 0
        self.assertEqual(x, 0)
        self.assertIs(y, x)
        self.assertEqual(z, 0)
        self.assertIs(w, x)
        self.assertIs(v, y)

    def test_patma_154(self):
        # [FUNCTION DOC] A literal case with a falsy guard does not run.
        x = 0
        y = None
        match x:
            case 0 if x:
                y = 0
        self.assertEqual(x, 0)
        self.assertIs(y, None)

    def test_patma_155(self):
        # [FUNCTION DOC] A literal with value infinity (from an overflowing float) does not
        # match 0.
        x = 0
        y = None
        match x:
            case 1e1000:
                y = 0
        self.assertEqual(x, 0)
        self.assertIs(y, None)

    def test_patma_156(self):
        # [FUNCTION DOC] A bare capture pattern always matches and binds z to the subject.
        x = 0
        match x:
            case z:
                y = 0
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)
        self.assertIs(z, x)

    def test_patma_157(self):
        # [FUNCTION DOC] A wildcard with a falsy guard does not run.
        x = 0
        y = None
        match x:
            case _ if x:
                y = 0
        self.assertEqual(x, 0)
        self.assertIs(y, None)

    def test_patma_158(self):
        # [FUNCTION DOC] Negative infinity literal does not match; the next case, 0, does.
        x = 0
        match x:
            case -1e1000:
                y = 0
            case 0:
                y = 1
        self.assertEqual(x, 0)
        self.assertEqual(y, 1)

    def test_patma_159(self):
        # [FUNCTION DOC] Literal case with a truthy guard (not x) runs first.
        x = 0
        match x:
            case 0 if not x:
                y = 0
            case 1:
                y = 1
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_160(self):
        # [FUNCTION DOC] After the first case matches, the second case's capture is never
        # bound; z stays None.
        x = 0
        z = None
        match x:
            case 0:
                y = 0
            case z if x:
                y = 1
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)
        self.assertIs(z, None)

    def test_patma_161(self):
        # [FUNCTION DOC] The wildcard case after a matching literal case is not run.
        x = 0
        match x:
            case 0:
                y = 0
            case _:
                y = 1
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_162(self):
        # [FUNCTION DOC] The first case's pattern fails; the second literal runs.
        x = 0
        match x:
            case 1 if x:
                y = 0
            case 0:
                y = 1
        self.assertEqual(x, 0)
        self.assertEqual(y, 1)

    def test_patma_163(self):
        # [FUNCTION DOC] Neither literal case matches; y stays None.
        x = 0
        y = None
        match x:
            case 1:
                y = 0
            case 1 if not x:
                y = 1
        self.assertEqual(x, 0)
        self.assertIs(y, None)

    def test_patma_164(self):
        # [FUNCTION DOC] After a failed literal, a capture case runs and binds z.
        x = 0
        match x:
            case 1:
                y = 0
            case z:
                y = 1
        self.assertEqual(x, 0)
        self.assertEqual(y, 1)
        self.assertIs(z, x)

    def test_patma_165(self):
        # [FUNCTION DOC] A guarded literal fails then the wildcard runs.
        x = 0
        match x:
            case 1 if x:
                y = 0
            case _:
                y = 1
        self.assertEqual(x, 0)
        self.assertEqual(y, 1)

    def test_patma_166(self):
        # [FUNCTION DOC] A guarded capture with a truthy guard runs first and binds z.
        x = 0
        match x:
            case z if not z:
                y = 0
            case 0 if x:
                y = 1
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)
        self.assertIs(z, x)

    def test_patma_167(self):
        # [FUNCTION DOC] A guarded capture runs before a non-matching literal.
        x = 0
        match x:
            case z if not z:
                y = 0
            case 1:
                y = 1
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)
        self.assertIs(z, x)

    def test_patma_168(self):
        # [FUNCTION DOC] Two capture cases; the first with a truthy guard wins.
        x = 0
        match x:
            case z if not x:
                y = 0
            case z:
                y = 1
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)
        self.assertIs(z, x)

    def test_patma_169(self):
        # [FUNCTION DOC] A capture with truthy guard wins over a later guarded wildcard.
        x = 0
        match x:
            case z if not z:
                y = 0
            case _ if x:
                y = 1
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)
        self.assertIs(z, x)

    def test_patma_170(self):
        # [FUNCTION DOC] A guarded wildcard wins over a later literal.
        x = 0
        match x:
            case _ if not x:
                y = 0
            case 0:
                y = 1
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_171(self):
        # [FUNCTION DOC] A falsy-guarded wildcard and a non-matching literal; y stays None.
        x = 0
        y = None
        match x:
            case _ if x:
                y = 0
            case 1:
                y = 1
        self.assertEqual(x, 0)
        self.assertIs(y, None)

    def test_patma_172(self):
        # [FUNCTION DOC] First guarded wildcard runs, so the later capture is not bound; z
        # stays None.
        x = 0
        z = None
        match x:
            case _ if not x:
                y = 0
            case z if not x:
                y = 1
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)
        self.assertIs(z, None)

    def test_patma_173(self):
        # [FUNCTION DOC] A guarded wildcard wins over an unguarded wildcard.
        x = 0
        match x:
            case _ if not x:
                y = 0
            case _:
                y = 1
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_174(self):
        # [FUNCTION DOC] Defines a status-code function with literal cases and a default
        # wildcard (the example from the Python tutorial) and checks hits, an unlisted code, a
        # string and a bitwise-or expression value.
        def http_error(status):
            # [FUNCTION DOC] HELPER: maps an HTTP status to a message with literal cases 400,
            # 401, 403, 404 and 418, and a wildcard default; returns the message.
            match status:
                case 400:
                    return "Bad request"
                case 401:
                    return "Unauthorized"
                case 403:
                    return "Forbidden"
                case 404:
                    return "Not found"
                case 418:
                    return "I'm a teapot"
                case _:
                    return "Something else"
        self.assertEqual(http_error(400), "Bad request")
        self.assertEqual(http_error(401), "Unauthorized")
        self.assertEqual(http_error(403), "Forbidden")
        self.assertEqual(http_error(404), "Not found")
        self.assertEqual(http_error(418), "I'm a teapot")
        self.assertEqual(http_error(123), "Something else")
        self.assertEqual(http_error("400"), "Something else")
        self.assertEqual(http_error(401 | 403 | 404), "Something else")  # 407

    def test_patma_175(self):
        # [FUNCTION DOC] Status-code function using an or-pattern and no default; unmatched
        # inputs return None.
        def http_error(status):
            # [FUNCTION DOC] HELPER: maps a status to a message; 401, 403 and 404 share one
            # or-pattern case; returns None when nothing matches.
            match status:
                case 400:
                    return "Bad request"
                case 401 | 403 | 404:
                    return "Not allowed"
                case 418:
                    return "I'm a teapot"
        self.assertEqual(http_error(400), "Bad request")
        self.assertEqual(http_error(401), "Not allowed")
        self.assertEqual(http_error(403), "Not allowed")
        self.assertEqual(http_error(404), "Not allowed")
        self.assertEqual(http_error(418), "I'm a teapot")
        self.assertIs(http_error(123), None)
        self.assertIs(http_error("400"), None)
        self.assertIs(http_error(401 | 403 | 404), None)  # 407

    def test_patma_176(self):
        # [FUNCTION DOC] Point-location function over plain tuples with literal and capture
        # items; non-tuples give "Not a point".
        def whereis(point):
            # [FUNCTION DOC] HELPER: classifies a tuple point as Origin, on the Y axis, on the
            # X axis, general, or not a point; returns a description string.
            match point:
                case (0, 0):
                    return "Origin"
                case (0, y):
                    return f"Y={y}"
                case (x, 0):
                    return f"X={x}"
                case (x, y):
                    return f"X={x}, Y={y}"
                case _:
                    return "Not a point"
        self.assertEqual(whereis((0, 0)), "Origin")
        self.assertEqual(whereis((0, -1.0)), "Y=-1.0")
        self.assertEqual(whereis(("X", 0)), "X=X")
        self.assertEqual(whereis((None, 1j)), "X=None, Y=1j")
        self.assertEqual(whereis(42), "Not a point")

    def test_patma_177(self):
        # [FUNCTION DOC] Point-location function using positional class patterns on the Point
        # dataclass; checks False equals 0, floats, strings, the class itself and non-
        # instances.
        def whereis(point):
            # [FUNCTION DOC] HELPER: classifies a Point via positional class patterns as
            # Origin, Y or X axis, somewhere else, or not a point; returns a description
            # string.
            match point:
                case Point(0, 0):
                    return "Origin"
                case Point(0, y):
                    return f"Y={y}"
                case Point(x, 0):
                    return f"X={x}"
                case Point():
                    return "Somewhere else"
                case _:
                    return "Not a point"
        self.assertEqual(whereis(Point(1, 0)), "X=1")
        self.assertEqual(whereis(Point(0, 0)), "Origin")
        self.assertEqual(whereis(10), "Not a point")
        self.assertEqual(whereis(Point(False, False)), "Origin")
        self.assertEqual(whereis(Point(0, -1.0)), "Y=-1.0")
        self.assertEqual(whereis(Point("X", 0)), "X=X")
        self.assertEqual(whereis(Point(None, 1j)), "Somewhere else")
        self.assertEqual(whereis(Point), "Not a point")
        self.assertEqual(whereis(42), "Not a point")

    def test_patma_178(self):
        # [FUNCTION DOC] Class pattern with one literal and one capture; returns the second
        # field when the first is 1, else None.
        def whereis(point):
            # [FUNCTION DOC] HELPER: returns the y field of a Point whose x equals 1 using
            # positional class patterns, else None.
            match point:
                case Point(1, var):
                    return var
        self.assertEqual(whereis(Point(1, 0)), 0)
        self.assertIs(whereis(Point(0, 0)), None)

    def test_patma_179(self):
        # [FUNCTION DOC] Class pattern mixing a positional literal and a keyword capture.
        def whereis(point):
            # [FUNCTION DOC] HELPER: returns the y field of a Point whose x equals 1 via a
            # keyword for y, else None.
            match point:
                case Point(1, y=var):
                    return var
        self.assertEqual(whereis(Point(1, 0)), 0)
        self.assertIs(whereis(Point(0, 0)), None)

    def test_patma_180(self):
        # [FUNCTION DOC] Class pattern with keyword-only subpatterns.
        def whereis(point):
            # [FUNCTION DOC] HELPER: returns y of a Point with x equal to 1 using keywords
            # only, else None.
            match point:
                case Point(x=1, y=var):
                    return var
        self.assertEqual(whereis(Point(1, 0)), 0)
        self.assertIs(whereis(Point(0, 0)), None)

    def test_patma_181(self):
        # [FUNCTION DOC] Keyword subpatterns given in swapped order still match by name.
        def whereis(point):
            # [FUNCTION DOC] HELPER: same as before with keywords written in y, x order; else
            # None.
            match point:
                case Point(y=var, x=1):
                    return var
        self.assertEqual(whereis(Point(1, 0)), 0)
        self.assertIs(whereis(Point(0, 0)), None)

    def test_patma_182(self):
        # [FUNCTION DOC] Function matching lists of Point objects with several sequence
        # patterns (empty, one point at origin, any single point, two points on the Y axis)
        # and a default.
        def whereis(points):
            # [FUNCTION DOC] HELPER: describes a list of points: empty, one at origin, a
            # single point, two on the Y axis, or something else; returns a string.
            match points:
                case []:
                    return "No points"
                case [Point(0, 0)]:
                    return "The origin"
                case [Point(x, y)]:
                    return f"Single point {x}, {y}"
                case [Point(0, y1), Point(0, y2)]:
                    return f"Two on the Y axis at {y1}, {y2}"
                case _:
                    return "Something else"
        self.assertEqual(whereis([]), "No points")
        self.assertEqual(whereis([Point(0, 0)]), "The origin")
        self.assertEqual(whereis([Point(0, 1)]), "Single point 0, 1")
        self.assertEqual(whereis([Point(0, 0), Point(0, 0)]), "Two on the Y axis at 0, 0")
        self.assertEqual(whereis([Point(0, 1), Point(0, 1)]), "Two on the Y axis at 1, 1")
        self.assertEqual(whereis([Point(0, 0), Point(1, 0)]), "Something else")
        self.assertEqual(whereis([Point(0, 0), Point(0, 0), Point(0, 0)]), "Something else")
        self.assertEqual(whereis([Point(0, 1), Point(0, 1), Point(0, 1)]), "Something else")

    def test_patma_183(self):
        # [FUNCTION DOC] Class pattern with guard x equal to y; compares captured fields
        # including False vs 0, complex and strings.
        def whereis(point):
            # [FUNCTION DOC] HELPER: returns a diagonal message using a guard that x equals y,
            # or a not-on-diagonal message; None for non-points.
            match point:
                case Point(x, y) if x == y:
                    return f"Y=X at {x}"
                case Point(x, y):
                    return "Not on the diagonal"
        self.assertEqual(whereis(Point(0, 0)), "Y=X at 0")
        self.assertEqual(whereis(Point(0, False)), "Y=X at 0")
        self.assertEqual(whereis(Point(False, 0)), "Y=X at False")
        self.assertEqual(whereis(Point(-1 - 1j, -1 - 1j)), "Y=X at (-1-1j)")
        self.assertEqual(whereis(Point("X", "X")), "Y=X at X")
        self.assertEqual(whereis(Point("X", "x")), "Not on the diagonal")

    def test_patma_184(self):
        # [FUNCTION DOC] A Sequence subclass with a None getitem and length 0 matches an empty
        # sequence pattern without indexing.
        class Seq(collections.abc.Sequence):
            # [CLASS DOC] HELPER: Sequence subclass with a non-callable getitem and zero
            # length.
            __getitem__ = None
            def __len__(self):
                # [FUNCTION DOC] HELPER: returns the length 0.
                return 0
        match Seq():
            case []:
                y = 0
        self.assertEqual(y, 0)

    def test_patma_185(self):
        # [FUNCTION DOC] A Sequence subclass with length 42 and a None getitem still matches a
        # star-only pattern, since nothing is indexed.
        class Seq(collections.abc.Sequence):
            # [CLASS DOC] HELPER: Sequence subclass with a non-callable getitem and length 42.
            __getitem__ = None
            def __len__(self):
                # [FUNCTION DOC] HELPER: returns the length 42.
                return 42
        match Seq():
            case [*_]:
                y = 0
        self.assertEqual(y, 0)

    def test_patma_186(self):
        # [FUNCTION DOC] A Sequence subclass whose getitem returns the index and length is 42;
        # first and last item patterns bind 0 and 41.
        class Seq(collections.abc.Sequence):
            # [CLASS DOC] HELPER: Sequence subclass whose getitem returns the index itself and
            # whose length is 42.
            def __getitem__(self, i):
                # [FUNCTION DOC] HELPER: returns the index argument unchanged.
                return i
            def __len__(self):
                # [FUNCTION DOC] HELPER: returns the length 42.
                return 42
        match Seq():
            case [x, *_, y]:
                z = 0
        self.assertEqual(x, 0)
        self.assertEqual(y, 41)
        self.assertEqual(z, 0)

    def test_patma_187(self):
        # [FUNCTION DOC] A range of 10 items matches a sequence pattern with two captures and
        # a star; the star capture is a list of 2..9.
        w = range(10)
        match w:
            case [x, y, *rest]:
                z = 0
        self.assertEqual(w, range(10))
        self.assertEqual(x, 0)
        self.assertEqual(y, 1)
        self.assertEqual(z, 0)
        self.assertEqual(rest, list(range(2, 10)))

    def test_patma_188(self):
        # [FUNCTION DOC] Same with a range of 100 and a parenthesized pattern.
        w = range(100)
        match w:
            case (x, y, *rest):
                z = 0
        self.assertEqual(w, range(100))
        self.assertEqual(x, 0)
        self.assertEqual(y, 1)
        self.assertEqual(z, 0)
        self.assertEqual(rest, list(range(2, 100)))

    def test_patma_189(self):
        # [FUNCTION DOC] Same with a range of 1000 and an unparenthesized pattern.
        w = range(1000)
        match w:
            case x, y, *rest:
                z = 0
        self.assertEqual(w, range(1000))
        self.assertEqual(x, 0)
        self.assertEqual(y, 1)
        self.assertEqual(z, 0)
        self.assertEqual(rest, list(range(2, 1000)))

    def test_patma_190(self):
        # [FUNCTION DOC] A range of 1024 with a discarding star only binds the first two
        # items.
        w = range(1 << 10)
        match w:
            case [x, y, *_]:
                z = 0
        self.assertEqual(w, range(1 << 10))
        self.assertEqual(x, 0)
        self.assertEqual(y, 1)
        self.assertEqual(z, 0)

    def test_patma_191(self):
        # [FUNCTION DOC] A range of 2 to the power 20 with a discarding star; the star must
        # not materialize items.
        w = range(1 << 20)
        match w:
            case (x, y, *_):
                z = 0
        self.assertEqual(w, range(1 << 20))
        self.assertEqual(x, 0)
        self.assertEqual(y, 1)
        self.assertEqual(z, 0)

    def test_patma_192(self):
        # [FUNCTION DOC] A range of 2 to the power 30 with a discarding star; must stay cheap
        # and bind the first two items.
        w = range(1 << 30)
        match w:
            case x, y, *_:
                z = 0
        self.assertEqual(w, range(1 << 30))
        self.assertEqual(x, 0)
        self.assertEqual(y, 1)
        self.assertEqual(z, 0)

    def test_patma_193(self):
        # [FUNCTION DOC] Mapping pattern with two string keys binds the two values by
        # identity.
        x = {"bandwidth": 0, "latency": 1}
        match x:
            case {"bandwidth": b, "latency": l}:
                y = 0
        self.assertEqual(x, {"bandwidth": 0, "latency": 1})
        self.assertIs(b, x["bandwidth"])
        self.assertIs(l, x["latency"])
        self.assertEqual(y, 0)

    def test_patma_194(self):
        # [FUNCTION DOC] Key order in the pattern does not matter and extra keys are ignored.
        x = {"bandwidth": 0, "latency": 1, "key": "value"}
        match x:
            case {"latency": l, "bandwidth": b}:
                y = 0
        self.assertEqual(x, {"bandwidth": 0, "latency": 1, "key": "value"})
        self.assertIs(l, x["latency"])
        self.assertIs(b, x["bandwidth"])
        self.assertEqual(y, 0)

    def test_patma_195(self):
        # [FUNCTION DOC] Rest capture collects the one unmatched key into a dict.
        x = {"bandwidth": 0, "latency": 1, "key": "value"}
        match x:
            case {"bandwidth": b, "latency": l, **rest}:
                y = 0
        self.assertEqual(x, {"bandwidth": 0, "latency": 1, "key": "value"})
        self.assertIs(b, x["bandwidth"])
        self.assertIs(l, x["latency"])
        self.assertEqual(rest, {"key": "value"})
        self.assertEqual(y, 0)

    def test_patma_196(self):
        # [FUNCTION DOC] Rest capture is an empty dict when all keys are matched.
        x = {"bandwidth": 0, "latency": 1}
        match x:
            case {"latency": l, "bandwidth": b, **rest}:
                y = 0
        self.assertEqual(x, {"bandwidth": 0, "latency": 1})
        self.assertIs(l, x["latency"])
        self.assertIs(b, x["bandwidth"])
        self.assertEqual(rest, {})
        self.assertEqual(y, 0)

    def test_patma_197(self):
        # [FUNCTION DOC] Sequence of two Point class patterns with as-capture on the second;
        # checks field and object identities.
        w = [Point(-1, 0), Point(1, 2)]
        match w:
            case (Point(x1, y1), Point(x2, y2) as p2):
                z = 0
        self.assertEqual(w, [Point(-1, 0), Point(1, 2)])
        self.assertIs(x1, w[0].x)
        self.assertIs(y1, w[0].y)
        self.assertIs(p2, w[1])
        self.assertIs(x2, w[1].x)
        self.assertIs(y2, w[1].y)
        self.assertIs(z, 0)

    def test_patma_198(self):
        # [FUNCTION DOC] Value patterns on plain Enum members: members match themselves, while
        # the class, ints, bool, complex and floats all fail and return None.
        class Color(enum.Enum):
            # [CLASS DOC] HELPER: plain Enum with members RED, GREEN, BLUE (values 0, 1, 2).
            RED = 0
            GREEN = 1
            BLUE = 2
        def f(color):
            # [FUNCTION DOC] HELPER: returns a message for each Color member through dotted
            # value patterns, else None.
            match color:
                case Color.RED:
                    return "I see red!"
                case Color.GREEN:
                    return "Grass is green"
                case Color.BLUE:
                    return "I'm feeling the blues :("
        self.assertEqual(f(Color.RED), "I see red!")
        self.assertEqual(f(Color.GREEN), "Grass is green")
        self.assertEqual(f(Color.BLUE), "I'm feeling the blues :(")
        self.assertIs(f(Color), None)
        self.assertIs(f(0), None)
        self.assertIs(f(1), None)
        self.assertIs(f(2), None)
        self.assertIs(f(3), None)
        self.assertIs(f(False), None)
        self.assertIs(f(True), None)
        self.assertIs(f(2+0j), None)
        self.assertIs(f(3.0), None)

    def test_patma_199(self):
        # [FUNCTION DOC] Value patterns on a mixin of int and Enum: members and equal ints
        # (including bool and a complex with zero imaginary part) match; 3 and 3.0 do not.
        class Color(int, enum.Enum):
            # [CLASS DOC] HELPER: Enum that is also an int, members RED, GREEN, BLUE (values
            # 0, 1, 2).
            RED = 0
            GREEN = 1
            BLUE = 2
        def f(color):
            # [FUNCTION DOC] HELPER: returns a message for each Color member through dotted
            # value patterns, else None.
            match color:
                case Color.RED:
                    return "I see red!"
                case Color.GREEN:
                    return "Grass is green"
                case Color.BLUE:
                    return "I'm feeling the blues :("
        self.assertEqual(f(Color.RED), "I see red!")
        self.assertEqual(f(Color.GREEN), "Grass is green")
        self.assertEqual(f(Color.BLUE), "I'm feeling the blues :(")
        self.assertIs(f(Color), None)
        self.assertEqual(f(0), "I see red!")
        self.assertEqual(f(1), "Grass is green")
        self.assertEqual(f(2), "I'm feeling the blues :(")
        self.assertIs(f(3), None)
        self.assertEqual(f(False), "I see red!")
        self.assertEqual(f(True), "Grass is green")
        self.assertEqual(f(2+0j), "I'm feeling the blues :(")
        self.assertIs(f(3.0), None)

    def test_patma_200(self):
        # [FUNCTION DOC] Class pattern with positional subpatterns uses the class's
        # __match_args__ names (a, b).
        class Class:
            # [CLASS DOC] HELPER: class whose __match_args__ names attributes a and b.
            __match_args__ = ("a", "b")
        c = Class()
        c.a = 0
        c.b = 1
        match c:
            case Class(x, y):
                z = 0
        self.assertIs(x, c.a)
        self.assertIs(y, c.b)
        self.assertEqual(z, 0)

    def test_patma_201(self):
        # [FUNCTION DOC] Mix of a positional subpattern and a keyword subpattern with
        # __match_args__.
        class Class:
            # [CLASS DOC] HELPER: class whose __match_args__ names attributes a and b.
            __match_args__ = ("a", "b")
        c = Class()
        c.a = 0
        c.b = 1
        match c:
            case Class(x, b=y):
                z = 0
        self.assertIs(x, c.a)
        self.assertIs(y, c.b)
        self.assertEqual(z, 0)

    def test_patma_202(self):
        # [FUNCTION DOC] When the pattern names the parent class, the parent's __match_args__
        # is used even if the instance's class has its own.
        class Parent:
            # [CLASS DOC] HELPER: parent class with __match_args__ of a and b.
            __match_args__ = "a", "b"
        class Child(Parent):
            # [CLASS DOC] HELPER: child class overriding __match_args__ with c and d.
            __match_args__ = ("c", "d")
        c = Child()
        c.a = 0
        c.b = 1
        match c:
            case Parent(x, y):
                z = 0
        self.assertIs(x, c.a)
        self.assertIs(y, c.b)
        self.assertEqual(z, 0)

    def test_patma_203(self):
        # [FUNCTION DOC] Same with a keyword subpattern: the positional one comes from the
        # parent's __match_args__.
        class Parent:
            # [CLASS DOC] HELPER: parent class with __match_args__ of a and b.
            __match_args__ = ("a", "b")
        class Child(Parent):
            # [CLASS DOC] HELPER: child class overriding __match_args__ with c and d.
            __match_args__ = "c", "d"
        c = Child()
        c.a = 0
        c.b = 1
        match c:
            case Parent(x, b=y):
                z = 0
        self.assertIs(x, c.a)
        self.assertIs(y, c.b)
        self.assertEqual(z, 0)

    def test_patma_204(self):
        # [FUNCTION DOC] Function returns the local variables (minus the argument) after a
        # literal-42 case; shows no names are bound, and that 42.0 matches but 0 and "42" do
        # not.
        def f(w):
            # [FUNCTION DOC] HELPER: on literal 42 returns the local variables minus the
            # argument; otherwise None.
            match w:
                case 42:
                    out = locals()
                    del out["w"]
                    return out
        self.assertEqual(f(42), {})
        self.assertIs(f(0), None)
        self.assertEqual(f(42.0), {})
        self.assertIs(f("42"), None)

    def test_patma_205(self):
        # [FUNCTION DOC] Same check with literal 42.0; 42 and 42.0 match, 0 and 0.0 do not.
        def f(w):
            # [FUNCTION DOC] HELPER: on literal 42.0 returns the locals minus the argument;
            # otherwise None.
            match w:
                case 42.0:
                    out = locals()
                    del out["w"]
                    return out
        self.assertEqual(f(42.0), {})
        self.assertEqual(f(42), {})
        self.assertIs(f(0.0), None)
        self.assertIs(f(0), None)

    def test_patma_206(self):
        # [FUNCTION DOC] Or-pattern of literals binds no names; 3.0 matches but 0, 4 and "1"
        # do not.
        def f(w):
            # [FUNCTION DOC] HELPER: on an or-pattern of 1, 2, 3 returns the locals minus the
            # argument; otherwise None.
            match w:
                case 1 | 2 | 3:
                    out = locals()
                    del out["w"]
                    return out
        self.assertEqual(f(1), {})
        self.assertEqual(f(2), {})
        self.assertEqual(f(3), {})
        self.assertEqual(f(3.0), {})
        self.assertIs(f(0), None)
        self.assertIs(f(4), None)
        self.assertIs(f("1"), None)

    def test_patma_207(self):
        # [FUNCTION DOC] Or of sequence literals binds no names; list [1, 2.0] matches, wrong
        # length or non-sequence does not.
        def f(w):
            # [FUNCTION DOC] HELPER: on a two-item sequence alternative returns the locals
            # minus the argument; otherwise None.
            match w:
                case [1, 2] | [3, 4]:
                    out = locals()
                    del out["w"]
                    return out
        self.assertEqual(f([1, 2]), {})
        self.assertEqual(f([3, 4]), {})
        self.assertIs(f(42), None)
        self.assertIs(f([2, 3]), None)
        self.assertIs(f([1, 2, 3]), None)
        self.assertEqual(f([1, 2.0]), {})

    def test_patma_208(self):
        # [FUNCTION DOC] A capture binds only that name for any subject.
        def f(w):
            # [FUNCTION DOC] HELPER: on a capture of x returns the locals minus the argument.
            match w:
                case x:
                    out = locals()
                    del out["w"]
                    return out
        self.assertEqual(f(42), {"x": 42})
        self.assertEqual(f((1, 2)), {"x": (1, 2)})
        self.assertEqual(f(None), {"x": None})

    def test_patma_209(self):
        # [FUNCTION DOC] A wildcard binds no names for any subject.
        def f(w):
            # [FUNCTION DOC] HELPER: on a wildcard returns the locals minus the argument.
            match w:
                case _:
                    out = locals()
                    del out["w"]
                    return out
        self.assertEqual(f(42), {})
        self.assertEqual(f(None), {})
        self.assertEqual(f((1, 2)), {})

    def test_patma_210(self):
        # [FUNCTION DOC] Three-name sequence pattern binds x, y, z for 3-item tuples,
        # array.array and memoryview, but not for wrong lengths, ints, str, bytes or
        # bytearray.
        def f(w):
            # [FUNCTION DOC] HELPER: on a three-item sequence capture returns the locals minus
            # the argument; otherwise None.
            match w:
                case (x, y, z):
                    out = locals()
                    del out["w"]
                    return out
        self.assertEqual(f((1, 2, 3)), {"x": 1, "y": 2, "z": 3})
        self.assertIs(f((1, 2)), None)
        self.assertIs(f((1, 2, 3, 4)), None)
        self.assertIs(f(123), None)
        self.assertIs(f("abc"), None)
        self.assertIs(f(b"abc"), None)
        self.assertEqual(f(array.array("b", b"abc")), {'x': 97, 'y': 98, 'z': 99})
        self.assertEqual(f(memoryview(b"abc")), {"x": 97, "y": 98, "z": 99})
        self.assertIs(f(bytearray(b"abc")), None)

    def test_patma_211(self):
        # [FUNCTION DOC] Mapping pattern with a literal value match binds only x and z;
        # mismatched literal or missing key gives None.
        def f(w):
            # [FUNCTION DOC] HELPER: on a mapping with captures and a literal value returns
            # the locals minus the argument; otherwise None.
            match w:
                case {"x": x, "y": "y", "z": z}:
                    out = locals()
                    del out["w"]
                    return out
        self.assertEqual(f({"x": "x", "y": "y", "z": "z"}), {"x": "x", "z": "z"})
        self.assertEqual(f({"x": "x", "y": "y", "z": "z", "a": "a"}), {"x": "x", "z": "z"})
        self.assertIs(f(({"x": "x", "y": "yy", "z": "z", "a": "a"})), None)
        self.assertIs(f(({"x": "x", "y": "y"})), None)

    def test_patma_212(self):
        # [FUNCTION DOC] Class pattern with nested int class pattern and keyword literal binds
        # only xx.
        def f(w):
            # [FUNCTION DOC] HELPER: on a Point class pattern with a nested int pattern
            # returns the locals minus the argument; otherwise None.
            match w:
                case Point(int(xx), y="hello"):
                    out = locals()
                    del out["w"]
                    return out
        self.assertEqual(f(Point(42, "hello")), {"xx": 42})

    def test_patma_213(self):
        # [FUNCTION DOC] Sequence pattern with as binds p, q and the whole subject x; works
        # for tuples and lists only.
        def f(w):
            # [FUNCTION DOC] HELPER: on a pair pattern with as returns the locals minus the
            # argument; otherwise None.
            match w:
                case (p, q) as x:
                    out = locals()
                    del out["w"]
                    return out
        self.assertEqual(f((1, 2)), {"p": 1, "q": 2, "x": (1, 2)})
        self.assertEqual(f([1, 2]), {"p": 1, "q": 2, "x": [1, 2]})
        self.assertIs(f(12), None)
        self.assertIs(f((1, 2, 3)), None)

    def test_patma_214(self):
        # [FUNCTION DOC] The set of local names after a literal case is empty.
        def f():
            # [FUNCTION DOC] HELPER: matches literal 42 and returns the current locals
            # mapping.
            match 42:
                case 42:
                    return locals()
        self.assertEqual(set(f()), set())

    def test_patma_215(self):
        # [FUNCTION DOC] An or-pattern literal case also leaves no new locals.
        def f():
            # [FUNCTION DOC] HELPER: matches 1 against an or-pattern and returns the locals
            # mapping.
            match 1:
                case 1 | 2 | 3:
                    return locals()
        self.assertEqual(set(f()), set())

    def test_patma_216(self):
        # [FUNCTION DOC] A wildcard case leaves no new locals.
        def f():
            # [FUNCTION DOC] HELPER: matches Ellipsis against a wildcard and returns the
            # locals mapping.
            match ...:
                case _:
                    return locals()
        self.assertEqual(set(f()), set())

    def test_patma_217(self):
        # [FUNCTION DOC] A capture binds only abc.
        def f():
            # [FUNCTION DOC] HELPER: matches Ellipsis against a capture named abc and returns
            # the locals mapping.
            match ...:
                case abc:
                    return locals()
        self.assertEqual(set(f()), {"abc"})

    def test_patma_218(self):
        # [FUNCTION DOC] A two-capture sequence binds exactly a and b.
        def f():
            # [FUNCTION DOC] HELPER: matches a pair of Ellipsis against two captures and
            # returns the locals mapping.
            match ..., ...:
                case a, b:
                    return locals()
        self.assertEqual(set(f()), {"a", "b"})

    def test_patma_219(self):
        # [FUNCTION DOC] A mapping pattern with two captures binds exactly a and b.
        def f():
            # [FUNCTION DOC] HELPER: matches a two-key mapping of Ellipsis values with two
            # captures and returns the locals mapping.
            match {"k": ..., "l": ...}:
                case {"k": a, "l": b}:
                    return locals()
        self.assertEqual(set(f()), {"a", "b"})

    def test_patma_220(self):
        # [FUNCTION DOC] A class pattern with positional and keyword captures binds x and y.
        def f():
            # [FUNCTION DOC] HELPER: matches a Point built from Ellipsis fields with a mixed
            # class pattern and returns the locals mapping.
            match Point(..., ...):
                case Point(x, y=y):
                    return locals()
        self.assertEqual(set(f()), {"x", "y"})

    def test_patma_221(self):
        # [FUNCTION DOC] A capture with as binds both names.
        def f():
            # [FUNCTION DOC] HELPER: matches Ellipsis against a capture with as and returns
            # the locals mapping.
            match ...:
                case b as a:
                    return locals()
        self.assertEqual(set(f()), {"a", "b"})

    def test_patma_222(self):
        # [FUNCTION DOC] A function whose only case is a wildcard always returns 0.
        def f(x):
            # [FUNCTION DOC] HELPER: always returns 0 through a wildcard case.
            match x:
                case _:
                    return 0
        self.assertEqual(f(0), 0)
        self.assertEqual(f(1), 0)
        self.assertEqual(f(2), 0)
        self.assertEqual(f(3), 0)

    def test_patma_223(self):
        # [FUNCTION DOC] A single literal case: returns 0 for 0, None otherwise.
        def f(x):
            # [FUNCTION DOC] HELPER: returns 0 on literal 0, else None.
            match x:
                case 0:
                    return 0
        self.assertEqual(f(0), 0)
        self.assertIs(f(1), None)
        self.assertIs(f(2), None)
        self.assertIs(f(3), None)

    def test_patma_224(self):
        # [FUNCTION DOC] Literal then wildcard default.
        def f(x):
            # [FUNCTION DOC] HELPER: returns 0 on literal 0, else 1.
            match x:
                case 0:
                    return 0
                case _:
                    return 1
        self.assertEqual(f(0), 0)
        self.assertEqual(f(1), 1)
        self.assertEqual(f(2), 1)
        self.assertEqual(f(3), 1)

    def test_patma_225(self):
        # [FUNCTION DOC] Two literal cases with no default; others return None.
        def f(x):
            # [FUNCTION DOC] HELPER: returns 0 or 1 on those literals, else None.
            match x:
                case 0:
                    return 0
                case 1:
                    return 1
        self.assertEqual(f(0), 0)
        self.assertEqual(f(1), 1)
        self.assertIs(f(2), None)
        self.assertIs(f(3), None)

    def test_patma_226(self):
        # [FUNCTION DOC] Two literals and a wildcard default.
        def f(x):
            # [FUNCTION DOC] HELPER: returns 0, 1 on those literals, else 2.
            match x:
                case 0:
                    return 0
                case 1:
                    return 1
                case _:
                    return 2
        self.assertEqual(f(0), 0)
        self.assertEqual(f(1), 1)
        self.assertEqual(f(2), 2)
        self.assertEqual(f(3), 2)

    def test_patma_227(self):
        # [FUNCTION DOC] Three literal cases and no default.
        def f(x):
            # [FUNCTION DOC] HELPER: returns 0, 1, 2 for those literals, else None.
            match x:
                case 0:
                    return 0
                case 1:
                    return 1
                case 2:
                    return 2
        self.assertEqual(f(0), 0)
        self.assertEqual(f(1), 1)
        self.assertEqual(f(2), 2)
        self.assertIs(f(3), None)

    def test_patma_228(self):
        # [FUNCTION DOC] The match and case keywords directly followed by empty parentheses
        # (no space) parse and match an empty tuple.
        match():
            case():
                x = 0
        self.assertEqual(x, 0)

    def test_patma_229(self):
        # [FUNCTION DOC] Same without spaces with a parenthesized capture.
        x = 0
        match(x):
            case(x):
                y = 0
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_230(self):
        # [FUNCTION DOC] A False pattern does not match 0, so the later 0 case runs.
        x = 0
        match x:
            case False:
                y = 0
            case 0:
                y = 1
        self.assertEqual(x, 0)
        self.assertEqual(y, 1)

    def test_patma_231(self):
        # [FUNCTION DOC] A True pattern does not match 1, so the later 1 case runs.
        x = 1
        match x:
            case True:
                y = 0
            case 1:
                y = 1
        self.assertEqual(x, 1)
        self.assertEqual(y, 1)

    def test_patma_232(self):
        # [FUNCTION DOC] A class whose equality always returns True: None, True and False
        # patterns compare by identity, so none match it.
        class Eq:
            # [CLASS DOC] HELPER: class whose equality method always returns True.
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: always reports equality.
                return True
        x = eq = Eq()
        # None
        y = None
        match x:
            case None:
                y = 0
        self.assertIs(x, eq)
        self.assertEqual(y, None)
        # True
        y = None
        match x:
            case True:
                y = 0
        self.assertIs(x, eq)
        self.assertEqual(y, None)
        # False
        y = None
        match x:
            case False:
                y = 0
        self.assertIs(x, eq)
        self.assertEqual(y, None)

    def test_patma_233(self):
        # [FUNCTION DOC] The False pattern matches False.
        x = False
        match x:
            case False:
                y = 0
        self.assertIs(x, False)
        self.assertEqual(y, 0)

    def test_patma_234(self):
        # [FUNCTION DOC] The True pattern matches True.
        x = True
        match x:
            case True:
                y = 0
        self.assertIs(x, True)
        self.assertEqual(y, 0)

    def test_patma_235(self):
        # [FUNCTION DOC] The None pattern matches None.
        x = None
        match x:
            case None:
                y = 0
        self.assertIs(x, None)
        self.assertEqual(y, 0)

    def test_patma_236(self):
        # [FUNCTION DOC] Chained as-captures both bind the subject.
        # KNOWN ISSUE: body is identical to test_patma_237, so one of them is redundant.
        x = 0
        match x:
            case (0 as w) as z:
                y = 0
        self.assertEqual(w, 0)
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)
        self.assertEqual(z, 0)

    def test_patma_237(self):
        # [FUNCTION DOC] Chained as-captures both bind the subject.
        # KNOWN ISSUE: body is identical to test_patma_236, so one of them is redundant.
        x = 0
        match x:
            case (0 as w) as z:
                y = 0
        self.assertEqual(w, 0)
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)
        self.assertEqual(z, 0)

    def test_patma_238(self):
        # [FUNCTION DOC] Nested as-captures inside a pair of two-item tuples bind every sub-
        # part and whole sub-tuples.
        x = ((0, 1), (2, 3))
        match x:
            case ((a as b, c as d) as e) as w, ((f as g, h) as i) as z:
                y = 0
        self.assertEqual(a, 0)
        self.assertEqual(b, 0)
        self.assertEqual(c, 1)
        self.assertEqual(d, 1)
        self.assertEqual(e, (0, 1))
        self.assertEqual(f, 2)
        self.assertEqual(g, 2)
        self.assertEqual(h, 3)
        self.assertEqual(i, (2, 3))
        self.assertEqual(w, (0, 1))
        self.assertEqual(x, ((0, 1), (2, 3)))
        self.assertEqual(y, 0)
        self.assertEqual(z, (2, 3))

    def test_patma_239(self):
        # [FUNCTION DOC] A UserDict (a Mapping but not a dict) matches a mapping pattern.
        x = collections.UserDict({0: 1, 2: 3})
        match x:
            case {2: 3}:
                y = 0
        self.assertEqual(x, {0: 1, 2: 3})
        self.assertEqual(y, 0)

    def test_patma_240(self):
        # [FUNCTION DOC] Rest capture over a UserDict yields a dict with the unmatched
        # entries.
        x = collections.UserDict({0: 1, 2: 3})
        match x:
            case {2: 3, **z}:
                y = 0
        self.assertEqual(x, {0: 1, 2: 3})
        self.assertEqual(y, 0)
        self.assertEqual(z, {0: 1})

    def test_patma_241(self):
        # [FUNCTION DOC] Deeply nested pattern combining a list class pattern, sequence, a
        # mapping with a complex key equal to 0 and an int class pattern with keyword fields;
        # binds z to 0.
        x = [[{0: 0}]]
        match x:
            case list([({-0-0j: int(real=0+0j, imag=0-0j) | (1) as z},)]):
                y = 0
        self.assertEqual(x, [[{0: 0}]])
        self.assertEqual(y, 0)
        self.assertEqual(z, 0)

    def test_patma_242(self):
        # [FUNCTION DOC] Range subject with first, star-discard and last captures binds the
        # end points.
        x = range(3)
        match x:
            case [y, *_, z]:
                w = 0
        self.assertEqual(w, 0)
        self.assertEqual(x, range(3))
        self.assertEqual(y, 0)
        self.assertEqual(z, 2)

    def test_patma_243(self):
        # [FUNCTION DOC] Wildcard first, star discard, last capture binds the last item.
        x = range(3)
        match x:
            case [_, *_, y]:
                z = 0
        self.assertEqual(x, range(3))
        self.assertEqual(y, 2)
        self.assertEqual(z, 0)

    def test_patma_244(self):
        # [FUNCTION DOC] Star discard followed by a last-item capture binds the last item.
        x = range(3)
        match x:
            case [*_, y]:
                z = 0
        self.assertEqual(x, range(3))
        self.assertEqual(y, 2)
        self.assertEqual(z, 0)

    def test_patma_245(self):
        # [FUNCTION DOC] Mapping value pattern is an or of two as-captures to the same name y;
        # binds y to 1.
        x = {"y": 1}
        match x:
            case {"y": (0 as y) | (1 as y)}:
                z = 0
        self.assertEqual(x, {"y": 1})
        self.assertEqual(y, 1)
        self.assertEqual(z, 0)

    def test_patma_246(self):
        # [FUNCTION DOC] Function with an or-pattern of four 10-item sequences binding the
        # same nine names in different positions; returns the locals minus x. Checks four
        # ranges, one per alternative, and a non-matching range.
        def f(x):
            # [FUNCTION DOC] HELPER: matches a 10-item sequence against four alternatives
            # binding the same names and returns the locals minus the argument; empty mapping
            # when nothing matches.
            match x:
                case ((a, b, c, d, e, f, g, h, i, 9) |
                      (h, g, i, a, b, d, e, c, f, 10) |
                      (g, b, a, c, d, -5, e, h, i, f) |
                      (-1, d, f, b, g, e, i, a, h, c)):
                    w = 0
            out = locals()
            del out["x"]
            return out
        alts = [
            dict(a=0, b=1, c=2, d=3, e=4, f=5, g=6, h=7, i=8, w=0),
            dict(h=1, g=2, i=3, a=4, b=5, d=6, e=7, c=8, f=9, w=0),
            dict(g=0, b=-1, a=-2, c=-3, d=-4, e=-6, h=-7, i=-8, f=-9, w=0),
            dict(d=-2, f=-3, b=-4, g=-5, e=-6, i=-7, a=-8, h=-9, c=-10, w=0),
            dict(),
        ]
        self.assertEqual(f(range(10)), alts[0])
        self.assertEqual(f(range(1, 11)), alts[1])
        self.assertEqual(f(range(0, -10, -1)), alts[2])
        self.assertEqual(f(range(-1, -11, -1)), alts[3])
        self.assertEqual(f(range(10, 20)), alts[4])

    def test_patma_247(self):
        # [FUNCTION DOC] Same four alternatives nested in the middle of a three-item pattern
        # with two extra captures y and z.
        def f(x):
            # [FUNCTION DOC] HELPER: same four alternatives nested inside a three-item
            # sequence; returns the locals minus the argument, or an empty mapping on no
            # match.
            match x:
                case [y, (a, b, c, d, e, f, g, h, i, 9) |
                         (h, g, i, a, b, d, e, c, f, 10) |
                         (g, b, a, c, d, -5, e, h, i, f) |
                         (-1, d, f, b, g, e, i, a, h, c), z]:
                    w = 0
            out = locals()
            del out["x"]
            return out
        alts = [
            dict(a=0, b=1, c=2, d=3, e=4, f=5, g=6, h=7, i=8, w=0, y=False, z=True),
            dict(h=1, g=2, i=3, a=4, b=5, d=6, e=7, c=8, f=9, w=0, y=False, z=True),
            dict(g=0, b=-1, a=-2, c=-3, d=-4, e=-6, h=-7, i=-8, f=-9, w=0, y=False, z=True),
            dict(d=-2, f=-3, b=-4, g=-5, e=-6, i=-7, a=-8, h=-9, c=-10, w=0, y=False, z=True),
            dict(),
        ]
        self.assertEqual(f((False, range(10), True)), alts[0])
        self.assertEqual(f((False, range(1, 11), True)), alts[1])
        self.assertEqual(f((False, range(0, -10, -1), True)), alts[2])
        self.assertEqual(f((False, range(-1, -11, -1), True)), alts[3])
        self.assertEqual(f((False, range(10, 20), True)), alts[4])

    def test_patma_248(self):
        # [FUNCTION DOC] A dict subclass defines a static get returning "bar"; mapping pattern
        # on key foo binds bar.
        # KNOWN ISSUE: the real value is also "bar", so the test cannot tell whether the
        # overridden get is used.
        class C(dict):
            # [CLASS DOC] HELPER: dict subclass overriding get.
            @staticmethod
            def get(key, default=None):
                # [FUNCTION DOC] HELPER: static method that ignores its arguments and returns
                # "bar".
                return 'bar'

        x = C({'foo': 'bar'})
        match x:
            case {'foo': bar}:
                y = bar

        self.assertEqual(y, 'bar')

    def test_patma_249(self):
        # [FUNCTION DOC] Checks name mangling is not applied to keyword names in class
        # patterns: inside a method of another class, the pattern attribute named __attr reads
        # the literal "__attr" attribute set via setattr, not a mangled one.
        class C:
            # [CLASS DOC] HELPER: class with a double-underscore class attribute (stored under
            # the mangled name) and a literal underscore-Outer-attribute class attribute.
            __attr = "eggs"  # mangled to _C__attr
            _Outer__attr = "bacon"
        class Outer:
            # [CLASS DOC] HELPER: class whose method contains a class pattern with a double-
            # underscore keyword.
            def f(self, x):
                # [FUNCTION DOC] HELPER: returns the value captured by a class pattern on C
                # using the attribute named double-underscore attr, else None.
                match x:
                    # looks up __attr, not _C__attr or _Outer__attr
                    case C(__attr=y):
                        return y
        c = C()
        setattr(c, "__attr", "spam")  # setattr is needed because we're in a class scope
        self.assertEqual(Outer().f(c), "spam")

    def test_patma_250(self):
        # [FUNCTION DOC] Mapping patterns with guards on the captured value choose True or
        # False by sign.
        def f(x):
            # [FUNCTION DOC] HELPER: returns True when the captured foo value is non-negative,
            # False when negative, else None.
            match x:
                case {"foo": y} if y >= 0:
                    return True
                case {"foo": y} if y < 0:
                    return False

        self.assertIs(f({"foo": 1}), True)
        self.assertIs(f({"foo": -1}), False)

    def test_patma_251(self):
        # [FUNCTION DOC] Value patterns on an attribute of a parameter with guards: returns
        # True, False or None depending on the pattern and guards.
        def f(v, x):
            # [FUNCTION DOC] HELPER: matches the value against an attribute of x with guards
            # for sign; returns True, False, or None by default.
            match v:
                case x.attr if x.attr >= 0:
                    return True
                case x.attr if x.attr < 0:
                    return False
                case _:
                    return None

        class X:
            # [CLASS DOC] HELPER: class holding one attribute.
            def __init__(self, attr):
                # [FUNCTION DOC] HELPER: stores attr on the instance.
                self.attr = attr

        self.assertIs(f(1, X(1)), True)
        self.assertIs(f(-1, X(-1)), False)
        self.assertIs(f(1, X(-1)), None)

    def test_patma_252(self):
        # [FUNCTION DOC] Guards can have side effects: a recording helper is called for each
        # guard until one passes; asserts the result and that all three calls were recorded in
        # order.
        # Side effects must be possible in guards:
        effects = []
        def lt(x, y):
            # [FUNCTION DOC] HELPER: records the pair (x, y) in the effects list and returns
            # whether x is less than y.
            effects.append((x, y))
            return x < y

        res = None
        match {"foo": 1}:
            case {"foo": x} if lt(x, 0):
                res = 0
            case {"foo": x} if lt(x, 1):
                res = 1
            case {"foo": x} if lt(x, 2):
                res = 2

        self.assertEqual(res, 2)
        self.assertEqual(effects, [(1, 0), (1, 1), (1, 2)])

    def test_patma_253(self):
        # [FUNCTION DOC] Or-pattern with a sequence capture and a plain capture of the same
        # name; binds x either way.
        def f(v):
            # [FUNCTION DOC] HELPER: returns x bound by an or-pattern of a one-item sequence
            # and a plain capture.
            match v:
                case [x] | x:
                    return x

        self.assertEqual(f(1), 1)
        self.assertEqual(f([1]), 1)

    def test_patma_254(self):
        # [FUNCTION DOC] Or-pattern with a mapping capture and a plain capture of the same
        # name.
        def f(v):
            # [FUNCTION DOC] HELPER: returns x bound by an or-pattern of a one-key mapping and
            # a plain capture.
            match v:
                case {"x": x} | x:
                    return x

        self.assertEqual(f(1), 1)
        self.assertEqual(f({"x": 1}), 1)

    def test_patma_255(self):
        # [FUNCTION DOC] A guard mutates the subject list (appending None) and is falsy; the
        # next case sees the mutated subject and matches; x ends as the mutated list.
        x = []
        match x:
            case [] as z if z.append(None):
                y = 0
            case [None]:
                y = 1
        self.assertEqual(x, [None])
        self.assertEqual(y, 1)
        self.assertIs(z, x)

    def test_patma_runtime_checkable_protocol(self):
        # [FUNCTION DOC] For classes A and a subclass B (under subTest), checks a runtime-
        # checkable Protocol with attributes x and y: a bare class pattern matches by
        # attribute presence and a keyword class pattern extracts the values.
        # Runtime-checkable protocol
        from typing import Protocol, runtime_checkable

        @runtime_checkable
        class P(Protocol):
            # [CLASS DOC] HELPER: runtime-checkable Protocol with integer attributes x and y.
            x: int
            y: int

        class A:
            # [CLASS DOC] HELPER: concrete class that stores x and y from its constructor.
            def __init__(self, x: int, y: int):
                # [FUNCTION DOC] HELPER: stores the x and y arguments as attributes.
                self.x = x
                self.y = y

        class B(A): ...
            # [CLASS DOC] HELPER: subclass of A with no changes.

        for cls in (A, B):
            with self.subTest(cls=cls.__name__):
                inst = cls(1, 2)
                w = 0
                match inst:
                    case P() as p:
                        self.assertIsInstance(p, cls)
                        self.assertEqual(p.x, 1)
                        self.assertEqual(p.y, 2)
                        w = 1
                self.assertEqual(w, 1)

                q = 0
                match inst:
                    case P(x=x, y=y):
                        self.assertEqual(x, 1)
                        self.assertEqual(y, 2)
                        q = 1
                self.assertEqual(q, 1)


    def test_patma_generic_protocol(self):
        # [FUNCTION DOC] For a class with attributes x, y and a generic class, checks that a
        # runtime-checkable generic protocol requiring a and b does not match, leaving the
        # flag at 0.
        # Runtime-checkable generic protocol
        from typing import Generic, TypeVar, Protocol, runtime_checkable

        T = TypeVar('T')  # not using PEP695 to be able to backport changes

        @runtime_checkable
        class P(Protocol[T]):
            # [CLASS DOC] HELPER: runtime-checkable generic Protocol with attributes a and b.
            a: T
            b: T

        class A:
            # [CLASS DOC] HELPER: class with attributes x and y, not a and b.
            def __init__(self, x: int, y: int):
                # [FUNCTION DOC] HELPER: stores the x and y arguments as attributes.
                self.x = x
                self.y = y

        class G(Generic[T]):
            # [CLASS DOC] HELPER: generic class storing x and y.
            def __init__(self, x: T, y: T):
                # [FUNCTION DOC] HELPER: stores the x and y arguments as attributes.
                self.x = x
                self.y = y

        for cls in (A, G):
            with self.subTest(cls=cls.__name__):
                inst = cls(1, 2)
                w = 0
                match inst:
                    case P():
                        w = 1
                self.assertEqual(w, 0)

    def test_patma_protocol_with_match_args(self):
        # [FUNCTION DOC] Same as the first protocol test but the protocol defines
        # __match_args__; also checks keyword literal and positional patterns (regression for
        # a past CPython issue).
        # Runtime-checkable protocol with `__match_args__`
        from typing import Protocol, runtime_checkable

        # Used to fail before
        # https://github.com/python/cpython/issues/110682
        @runtime_checkable
        class P(Protocol):
            # [CLASS DOC] HELPER: runtime-checkable Protocol with __match_args__ of x and y
            # and attributes x, y.
            __match_args__ = ('x', 'y')
            x: int
            y: int

        class A:
            # [CLASS DOC] HELPER: concrete class that stores x and y from its constructor.
            def __init__(self, x: int, y: int):
                # [FUNCTION DOC] HELPER: stores the x and y arguments as attributes.
                self.x = x
                self.y = y

        class B(A): ...
            # [CLASS DOC] HELPER: subclass of A with no changes.

        for cls in (A, B):
            with self.subTest(cls=cls.__name__):
                inst = cls(1, 2)
                w = 0
                match inst:
                    case P() as p:
                        self.assertIsInstance(p, cls)
                        self.assertEqual(p.x, 1)
                        self.assertEqual(p.y, 2)
                        w = 1
                self.assertEqual(w, 1)

                q = 0
                match inst:
                    case P(x=x, y=y):
                        self.assertEqual(x, 1)
                        self.assertEqual(y, 2)
                        q = 1
                self.assertEqual(q, 1)

                j = 0
                match inst:
                    case P(x=1, y=2):
                        j = 1
                self.assertEqual(j, 1)

                g = 0
                match inst:
                    case P(x, y):
                        self.assertEqual(x, 1)
                        self.assertEqual(y, 2)
                        g = 1
                self.assertEqual(g, 1)

                h = 0
                match inst:
                    case P(1, 2):
                        h = 1
                self.assertEqual(h, 1)


class TestSyntaxErrors(CPythonTestCase):
    # [CLASS DOC] Suite checking that invalid pattern syntax or semantics fail compilation
    # with a SyntaxError.

    def assert_syntax_error(self, code: str):
        # [FUNCTION DOC] HELPER: Dedents the given source snippet and compiles it in exec
        # mode; asserts that a SyntaxError is raised. Nothing is executed.
        with self.assertRaises(SyntaxError):
            compile(inspect.cleandoc(code), "<test>", "exec")

    def test_alternative_patterns_bind_different_names_0(self):
        # [FUNCTION DOC] A literal alternative with a capture alternative must not compile
        # because the alternatives bind different names.
        self.assert_syntax_error("""
        match ...:
            case "a" | a:
                pass
        """)

    def test_alternative_patterns_bind_different_names_1(self):
        # [FUNCTION DOC] Alternatives binding different names (b, c, d) inside a sequence must
        # not compile.
        self.assert_syntax_error("""
        match ...:
            case [a, [b] | [c] | [d]]:
                pass
        """)


    def test_attribute_name_repeated_in_class_pattern(self):
        # [FUNCTION DOC] A class pattern repeating the keyword a must not compile.
        self.assert_syntax_error("""
        match ...:
            case Class(a=_, a=_):
                pass
        """)

    def test_imaginary_number_required_in_complex_literal_0(self):
        # [FUNCTION DOC] Literal 0+0 (no imaginary part) must not compile as a pattern.
        self.assert_syntax_error("""
        match ...:
            case 0+0:
                pass
        """)

    def test_imaginary_number_required_in_complex_literal_1(self):
        # [FUNCTION DOC] The same literal as a mapping key must not compile.
        self.assert_syntax_error("""
        match ...:
            case {0+0: _}:
                pass
        """)

    def test_invalid_syntax_0(self):
        # [FUNCTION DOC] A mapping with a rest capture followed by another key must not
        # compile.
        self.assert_syntax_error("""
        match ...:
            case {**rest, "key": value}:
                pass
        """)

    def test_invalid_syntax_1(self):
        # [FUNCTION DOC] A rest capture in the middle of the keys must not compile.
        self.assert_syntax_error("""
        match ...:
            case {"first": first, **rest, "last": last}:
                pass
        """)

    def test_invalid_syntax_2(self):
        # [FUNCTION DOC] A rest capture alone with a wildcard name must not compile.
        self.assert_syntax_error("""
        match ...:
            case {**_}:
                pass
        """)

    def test_invalid_syntax_3(self):
        # [FUNCTION DOC] The as-wildcard pattern must not compile.
        self.assert_syntax_error("""
        match ...:
            case 42 as _:
                pass
        """)

    def test_len1_tuple_sequence_pattern_comma(self):
        # [FUNCTION DOC] A parenthesized star without a trailing comma must not compile.
        # correct syntax would be `case(*x,):`
        self.assert_syntax_error("""
        match ...:
            case (*x):
                pass
        """)

    def test_mapping_pattern_keys_may_only_match_literals_and_attribute_lookups(self):
        # [FUNCTION DOC] An f-string as a mapping key must not compile.
        self.assert_syntax_error("""
        match ...:
            case {f"": _}:
                pass
        """)

    def test_multiple_assignments_to_name_in_pattern_0(self):
        # [FUNCTION DOC] A sequence pattern binding the same name twice must not compile.
        self.assert_syntax_error("""
        match ...:
            case a, a:
                pass
        """)

    def test_multiple_assignments_to_name_in_pattern_1(self):
        # [FUNCTION DOC] A mapping pattern binding the same name twice must not compile.
        self.assert_syntax_error("""
        match ...:
            case {"k": a, "l": a}:
                pass
        """)

    def test_multiple_assignments_to_name_in_pattern_2(self):
        # [FUNCTION DOC] Positional class subpatterns binding the same name must not compile.
        self.assert_syntax_error("""
        match ...:
            case MyClass(x, x):
                pass
        """)

    def test_multiple_assignments_to_name_in_pattern_3(self):
        # [FUNCTION DOC] Keyword class subpatterns binding the same name must not compile.
        self.assert_syntax_error("""
        match ...:
            case MyClass(x=x, y=x):
                pass
        """)

    def test_multiple_assignments_to_name_in_pattern_4(self):
        # [FUNCTION DOC] Positional plus keyword subpatterns binding the same name must not
        # compile.
        self.assert_syntax_error("""
        match ...:
            case MyClass(x, y=x):
                pass
        """)

    def test_multiple_assignments_to_name_in_pattern_5(self):
        # [FUNCTION DOC] An as-capture to the name already captured must not compile.
        self.assert_syntax_error("""
        match ...:
            case a as a:
                pass
        """)

    def test_multiple_starred_names_in_sequence_pattern_0(self):
        # [FUNCTION DOC] A sequence pattern with three stars must not compile.
        self.assert_syntax_error("""
        match ...:
            case *a, b, *c, d, *e:
                pass
        """)

    def test_multiple_starred_names_in_sequence_pattern_1(self):
        # [FUNCTION DOC] A sequence pattern with two stars must not compile.
        self.assert_syntax_error("""
        match ...:
            case a, *b, c, *d, e:
                pass
        """)

    def test_name_capture_makes_remaining_patterns_unreachable_0(self):
        # [FUNCTION DOC] A capture followed by an alternative in an or-pattern must not
        # compile because the capture is irrefutable.
        self.assert_syntax_error("""
        match ...:
            case a | "a":
                pass
        """)

    def test_name_capture_makes_remaining_patterns_unreachable_1(self):
        # [FUNCTION DOC] A capture case followed by another case must not compile.
        self.assert_syntax_error("""
        match 42:
            case x:
                pass
            case y:
                pass
        """)

    def test_name_capture_makes_remaining_patterns_unreachable_2(self):
        # [FUNCTION DOC] An irrefutable first alternative with guard and later alternatives
        # must not compile.
        self.assert_syntax_error("""
        match ...:
            case x | [_ as x] if x:
                pass
        """)

    def test_name_capture_makes_remaining_patterns_unreachable_3(self):
        # [FUNCTION DOC] A capture case followed by a later guarded case must not compile.
        self.assert_syntax_error("""
        match ...:
            case x:
                pass
            case [x] if x:
                pass
        """)

    def test_name_capture_makes_remaining_patterns_unreachable_4(self):
        # [FUNCTION DOC] A capture case followed by a wildcard case must not compile.
        self.assert_syntax_error("""
        match ...:
            case x:
                pass
            case _:
                pass
        """)

    def test_patterns_may_only_match_literals_and_attribute_lookups_0(self):
        # [FUNCTION DOC] An empty f-string as a pattern must not compile.
        self.assert_syntax_error("""
        match ...:
            case f"":
                pass
        """)

    def test_patterns_may_only_match_literals_and_attribute_lookups_1(self):
        # [FUNCTION DOC] An f-string with a field as a pattern must not compile.
        self.assert_syntax_error("""
        match ...:
            case f"{x}":
                pass
        """)

    def test_real_number_required_in_complex_literal_0(self):
        # [FUNCTION DOC] Imaginary plus real (0j+0) must not compile as a pattern.
        self.assert_syntax_error("""
        match ...:
            case 0j+0:
                pass
        """)

    def test_real_number_required_in_complex_literal_1(self):
        # [FUNCTION DOC] Imaginary plus imaginary must not compile.
        self.assert_syntax_error("""
        match ...:
            case 0j+0j:
                pass
        """)

    def test_real_number_required_in_complex_literal_2(self):
        # [FUNCTION DOC] Imaginary plus real as a mapping key must not compile.
        self.assert_syntax_error("""
        match ...:
            case {0j+0: _}:
                pass
        """)

    def test_real_number_required_in_complex_literal_3(self):
        # [FUNCTION DOC] Imaginary plus imaginary as a mapping key must not compile.
        self.assert_syntax_error("""
        match ...:
            case {0j+0j: _}:
                pass
        """)

    def test_real_number_multiple_ops(self):
        # [FUNCTION DOC] A chain of several additions in a literal pattern must not compile.
        self.assert_syntax_error("""
        match ...:
            case 0 + 0j + 0:
                pass
        """)

    def test_real_number_wrong_ops(self):
        # [FUNCTION DOC] For each of the operators multiply, divide, matmul, power, modulo and
        # floor-divide, checks via subTest that four literal patterns mixing real and
        # imaginary operands with that operator all fail to compile.
        for op in ["*", "/", "@", "**", "%", "//"]:
            with self.subTest(op=op):
                self.assert_syntax_error(f"""
                match ...:
                    case 0 {op} 0j:
                        pass
                """)
                self.assert_syntax_error(f"""
                match ...:
                    case 0j {op} 0:
                        pass
                """)
                self.assert_syntax_error(f"""
                match ...:
                    case -0j {op} 0:
                        pass
                """)
                self.assert_syntax_error(f"""
                match ...:
                    case 0j {op} -0:
                        pass
                """)

    def test_wildcard_makes_remaining_patterns_unreachable_0(self):
        # [FUNCTION DOC] Two wildcard alternatives in an or-pattern must not compile.
        self.assert_syntax_error("""
        match ...:
            case _ | _:
                pass
        """)

    def test_wildcard_makes_remaining_patterns_unreachable_1(self):
        # [FUNCTION DOC] A wildcard alternative with capture before a later alternative must
        # not compile.
        self.assert_syntax_error("""
        match ...:
            case (_ as x) | [x]:
                pass
        """)

    def test_wildcard_makes_remaining_patterns_unreachable_2(self):
        # [FUNCTION DOC] Two wildcards with a guard in an or-pattern must not compile.
        self.assert_syntax_error("""
        match ...:
            case _ | _ if condition():
                pass
        """)

    def test_wildcard_makes_remaining_patterns_unreachable_3(self):
        # [FUNCTION DOC] A wildcard case followed by another case must not compile.
        self.assert_syntax_error("""
        match ...:
            case _:
                pass
            case None:
                pass
        """)

    def test_wildcard_makes_remaining_patterns_unreachable_4(self):
        # [FUNCTION DOC] A nested irrefutable or-pattern before a wildcard must not compile.
        self.assert_syntax_error("""
        match ...:
            case (None | _) | _:
                pass
        """)

    def test_wildcard_makes_remaining_patterns_unreachable_5(self):
        # [FUNCTION DOC] A wildcard alternative before a True or False group must not compile.
        self.assert_syntax_error("""
        match ...:
            case _ | (True | False):
                pass
        """)

    def test_mapping_pattern_duplicate_key(self):
        # [FUNCTION DOC] A mapping pattern with the same literal key twice must not compile.
        self.assert_syntax_error("""
        match ...:
            case {"a": _, "a": _}:
                pass
        """)

    def test_mapping_pattern_duplicate_key_edge_case0(self):
        # [FUNCTION DOC] Keys 0 and False are equal, so the mapping must not compile.
        self.assert_syntax_error("""
        match ...:
            case {0: _, False: _}:
                pass
        """)

    def test_mapping_pattern_duplicate_key_edge_case1(self):
        # [FUNCTION DOC] Keys 0 and 0.0 are equal, so the mapping must not compile.
        self.assert_syntax_error("""
        match ...:
            case {0: _, 0.0: _}:
                pass
        """)

    def test_mapping_pattern_duplicate_key_edge_case2(self):
        # [FUNCTION DOC] Keys 0 and negative zero are equal, so the mapping must not compile.
        self.assert_syntax_error("""
        match ...:
            case {0: _, -0: _}:
                pass
        """)

    def test_mapping_pattern_duplicate_key_edge_case3(self):
        # [FUNCTION DOC] Keys 0 and 0j are equal, so the mapping must not compile.
        self.assert_syntax_error("""
        match ...:
            case {0: _, 0j: _}:
                pass
        """)

class TestTypeErrors(CPythonTestCase):
    # [CLASS DOC] Suite checking that malformed class patterns raise TypeError at match time
    # and leave names unbound.

    def test_accepts_positional_subpatterns_0(self):
        # [FUNCTION DOC] Class with an empty __match_args__ used with a positional subpattern
        # raises TypeError; y and z stay None.
        class Class:
            # [CLASS DOC] HELPER: class with an empty __match_args__, so it accepts no
            # positional subpatterns.
            __match_args__ = ()
        x = Class()
        y = z = None
        with self.assertRaises(TypeError):
            match x:
                case Class(y):
                    z = 0
        self.assertIs(y, None)
        self.assertIs(z, None)

    def test_accepts_positional_subpatterns_1(self):
        # [FUNCTION DOC] range(10) as a class pattern with a positional subpattern raises
        # TypeError because range accepts none; subject unchanged.
        x = range(10)
        y = None
        with self.assertRaises(TypeError):
            match x:
                case range(10):
                    y = 0
        self.assertEqual(x, range(10))
        self.assertIs(y, None)

    def test_got_multiple_subpatterns_for_attribute_0(self):
        # [FUNCTION DOC] __match_args__ listing the same name twice with two positional
        # subpatterns raises TypeError.
        class Class:
            # [CLASS DOC] HELPER: class whose __match_args__ repeats the attribute name a
            # twice.
            __match_args__ = ("a", "a")
            a = None
        x = Class()
        w = y = z = None
        with self.assertRaises(TypeError):
            match x:
                case Class(y, z):
                    w = 0
        self.assertIs(w, None)
        self.assertIs(y, None)
        self.assertIs(z, None)

    def test_got_multiple_subpatterns_for_attribute_1(self):
        # [FUNCTION DOC] A positional subpattern and a keyword subpattern for the same
        # attribute raise TypeError.
        class Class:
            # [CLASS DOC] HELPER: class whose __match_args__ names a, with attribute a set to
            # None.
            __match_args__ = ("a",)
            a = None
        x = Class()
        w = y = z = None
        with self.assertRaises(TypeError):
            match x:
                case Class(y, a=z):
                    w = 0
        self.assertIs(w, None)
        self.assertIs(y, None)
        self.assertIs(z, None)

    def test_match_args_elements_must_be_strings(self):
        # [FUNCTION DOC] A non-string element in __match_args__ raises TypeError.
        class Class:
            # [CLASS DOC] HELPER: class whose __match_args__ contains a non-string (None).
            __match_args__ = (None,)
        x = Class()
        y = z = None
        with self.assertRaises(TypeError):
            match x:
                case Class(y):
                    z = 0
        self.assertIs(y, None)
        self.assertIs(z, None)

    def test_match_args_must_be_a_tuple_0(self):
        # [FUNCTION DOC] A None __match_args__ with a positional subpattern raises TypeError.
        class Class:
            # [CLASS DOC] HELPER: class whose __match_args__ is None instead of a tuple.
            __match_args__ = None
        x = Class()
        y = z = None
        with self.assertRaises(TypeError):
            match x:
                case Class(y):
                    z = 0
        self.assertIs(y, None)
        self.assertIs(z, None)

    def test_match_args_must_be_a_tuple_1(self):
        # [FUNCTION DOC] A string __match_args__ raises TypeError.
        class Class:
            # [CLASS DOC] HELPER: class whose __match_args__ is a string instead of a tuple.
            __match_args__ = "XYZ"
        x = Class()
        y = z = None
        with self.assertRaises(TypeError):
            match x:
                case Class(y):
                    z = 0
        self.assertIs(y, None)
        self.assertIs(z, None)

    def test_match_args_must_be_a_tuple_2(self):
        # [FUNCTION DOC] A list __match_args__ raises TypeError even though the attributes
        # exist.
        class Class:
            # [CLASS DOC] HELPER: class whose __match_args__ is a list and which has
            # attributes spam and eggs.
            __match_args__ = ["spam", "eggs"]
            spam = 0
            eggs = 1
        x = Class()
        w = y = z = None
        with self.assertRaises(TypeError):
            match x:
                case Class(y, z):
                    w = 0
        self.assertIs(w, None)
        self.assertIs(y, None)
        self.assertIs(z, None)

    def test_class_pattern_not_type(self):
        # [FUNCTION DOC] Using a builtin function (not a type) as a class pattern raises
        # TypeError; w stays None.
        w = None
        with self.assertRaises(TypeError):
            match 1:
                case max(0, 1):
                    w = 0
        self.assertIsNone(w)

    def test_regular_protocol(self):
        # [FUNCTION DOC] Matching a class pattern on a non-runtime-checkable Protocol raises
        # TypeError with a message about runtime_checkable protocols.
        from typing import Protocol
        class P(Protocol): ...
            # [CLASS DOC] HELPER: Protocol class that is not runtime-checkable, empty body.
        msg = (
            'Instance and class checks can only be used '
            'with @runtime_checkable protocols'
        )
        w = None
        with self.assertRaisesRegex(TypeError, msg):
            match 1:
                case P():
                    w = 0
        self.assertIsNone(w)

    def test_positional_patterns_with_regular_protocol(self):
        # [FUNCTION DOC] A protocol without __match_args__ used with positional subpatterns
        # raises TypeError.
        from typing import Protocol
        class P(Protocol):
            # [CLASS DOC] HELPER: Protocol with attributes x and y and no __match_args__.
            x: int  # no `__match_args__`
            y: int
        class A:
            # [CLASS DOC] HELPER: class with attributes x and y.
            x = 1
            y = 2
        w = None
        with self.assertRaises(TypeError):
            match A():
                case P(x, y):
                    w = 0
        self.assertIsNone(w)


class TestValueErrors(CPythonTestCase):
    # [CLASS DOC] Suite for matches that raise ValueError at match time.

    def test_mapping_pattern_checks_duplicate_key_1(self):
        # [FUNCTION DOC] A mapping pattern with a dotted-name key and a literal key evaluating
        # to the same value raises ValueError at match time; no names get bound.
        class Keys:
            # [CLASS DOC] HELPER: namespace class with the string constant KEY equal to "a".
            KEY = "a"
        x = {"a": 0, "b": 1}
        w = y = z = None
        with self.assertRaises(ValueError):
            match x:
                case {Keys.KEY: y, "a": z}:
                    w = 0
        self.assertIs(w, None)
        self.assertIs(y, None)
        self.assertIs(z, None)

class TestSourceLocations(CPythonTestCase):
    # [CLASS DOC] Suite checking line information in compiled match code.
    def test_jump_threading(self):
        # [FUNCTION DOC] Compiles a function with a match whose cases share a guard,
        # disassembles it, and asserts that every jump instruction has a line number
        # (regression for jump threading).
        # See gh-123048
        def f():
            # [FUNCTION DOC] HELPER: function containing two cases each with a guarded
            # assignment; only inspected via disassembly, never called.
            x = 0
            v = 1
            match v:
                case 1:
                    if x < 0:
                        x = 1
                case 2:
                    if x < 0:
                        x = 1
            x += 1

        for inst in dis.get_instructions(f):
            if inst.opcode in dis.hasjump:
                self.assertIsNotNone(inst.positions.lineno, "jump without location")

class TestTracing(CPythonTestCase):
    # [CLASS DOC] Suite verifying the sequence of traced lines when match statements execute.

    @staticmethod
    def _trace(func, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Static helper. Runs func with the given arguments while a
        # line-event trace function is installed, records each executed line relative to the
        # function's first line, restores the previous trace function even on exceptions, and
        # returns the list of line offsets.
        actual_linenos = []

        def trace(frame, event, arg):
            # [FUNCTION DOC] HELPER: trace callback that appends line offsets for line events
            # in the traced function and returns itself.
            if event == "line" and frame.f_code.co_name == func.__name__:
                assert arg is None
                relative_lineno = frame.f_lineno - func.__code__.co_firstlineno
                actual_linenos.append(relative_lineno)
            return trace

        old_trace = sys.gettrace()
        sys.settrace(trace)
        try:
            func(*args, **kwargs)
        finally:
            sys.settrace(old_trace)
        return actual_linenos

    def test_default_wildcard(self):
        # [FUNCTION DOC] Traces a command parser with guarded and plain list cases and a
        # wildcard default; asserts the exact executed line offsets for go n, go x and spam.
        def f(command):                                         # 0
            # [FUNCTION DOC] HELPER: command parser used for tracing; guarded go case, bare go
            # case and wildcard default; returns a string.
            match command.split():                              # 1
                case ["go", direction] if direction in "nesw":  # 2
                    return f"go {direction}"                    # 3
                case ["go", _]:                                 # 4
                    return "no go"                              # 5
                case _:                                         # 6
                    return "default"                            # 7

        self.assertListEqual(self._trace(f, "go n"), [1, 2, 3])
        self.assertListEqual(self._trace(f, "go x"), [1, 2, 4, 5])
        self.assertListEqual(self._trace(f, "spam"), [1, 2, 4, 6, 7])

    def test_default_capture(self):
        # [FUNCTION DOC] Same trace expectations with a capture case as the default.
        def f(command):                                         # 0
            # [FUNCTION DOC] HELPER: command parser with the default being a capture that
            # returns the command pieces.
            match command.split():                              # 1
                case ["go", direction] if direction in "nesw":  # 2
                    return f"go {direction}"                    # 3
                case ["go", _]:                                 # 4
                    return "no go"                              # 5
                case x:                                         # 6
                    return x                                    # 7

        self.assertListEqual(self._trace(f, "go n"), [1, 2, 3])
        self.assertListEqual(self._trace(f, "go x"), [1, 2, 4, 5])
        self.assertListEqual(self._trace(f, "spam"), [1, 2, 4, 6, 7])

    def test_no_default(self):
        # [FUNCTION DOC] Same cases without a default; the unmatched input stops at line 4.
        def f(command):                                         # 0
            # [FUNCTION DOC] HELPER: command parser with two go cases and no default.
            match command.split():                              # 1
                case ["go", direction] if direction in "nesw":  # 2
                    return f"go {direction}"                    # 3
                case ["go", _]:                                 # 4
                    return "no go"                              # 5

        self.assertListEqual(self._trace(f, "go n"), [1, 2, 3])
        self.assertListEqual(self._trace(f, "go x"), [1, 2, 4, 5])
        self.assertListEqual(self._trace(f, "spam"), [1, 2, 4])

    def test_only_default_wildcard(self):
        # [FUNCTION DOC] A function with only a wildcard case executes the same lines for any
        # input.
        def f(command):               # 0
            # [FUNCTION DOC] HELPER: matches the split command against a lone wildcard and
            # returns "default".
            match command.split():    # 1
                case _:               # 2
                    return "default"  # 3

        self.assertListEqual(self._trace(f, "go n"), [1, 2, 3])
        self.assertListEqual(self._trace(f, "go x"), [1, 2, 3])
        self.assertListEqual(self._trace(f, "spam"), [1, 2, 3])

    def test_only_default_capture(self):
        # [FUNCTION DOC] Same with a single capture case.
        def f(command):             # 0
            # [FUNCTION DOC] HELPER: matches the split command against a lone capture and
            # returns it.
            match command.split():  # 1
                case x:             # 2
                    return x        # 3

        self.assertListEqual(self._trace(f, "go n"), [1, 2, 3])
        self.assertListEqual(self._trace(f, "go x"), [1, 2, 3])
        self.assertListEqual(self._trace(f, "spam"), [1, 2, 3])

    def test_unreachable_code(self):
        # [FUNCTION DOC] Traces a match with two cases containing unreachable if False bodies;
        # asserts the lines executed for inputs 1 and 0.
        def f(command):               # 0
            # [FUNCTION DOC] HELPER: match on a number with unreachable-body ifs; returns 1 or
            # 0 only in dead branches.
            match command:            # 1
                case 1:               # 2
                    if False:         # 3
                        return 1      # 4
                case _:               # 5
                    if False:         # 6
                        return 0      # 7

        self.assertListEqual(self._trace(f, 1), [1, 2, 3])
        self.assertListEqual(self._trace(f, 0), [1, 2, 5, 6])

    @torch._dynamo.error_on_graph_break(False)
    def test_parser_deeply_nested_patterns(self):
        # [FUNCTION DOC] Builds three patterns nested 100 levels deep (class parens, mappings,
        # lists) and compiles a match using each; guards against exponential backtracking in
        # the parser (see gh-93671). Wrapped so graph breaks do not raise.
        # Deeply nested patterns can cause exponential backtracking when parsing.
        # See gh-93671 for more information.

        levels = 100

        patterns = [
            "A" + "(" * levels + ")" * levels,
            "{1:" * levels + "1" + "}" * levels,
            "[" * levels + "1" + "]" * levels,
        ]

        for pattern in patterns:
            with self.subTest(pattern):
                code = inspect.cleandoc("""
                    match None:
                        case {}:
                            pass
                """.format(pattern))
                compile(code, "<string>", "exec")


if __name__ == "__main__":
    run_tests()
