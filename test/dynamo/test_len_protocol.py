# Owner(s): ["module: dynamo"]

# [MODULE DOC] Dynamo tests for the len builtin and the dunder length protocol. Defines mixin and
# TestCase classes covering list, tuple, dict-like, set, range, str, Tensor, nn.Module containers,
# dict views, deque, mappingproxy, user classes, builtin subclasses with overridden length,
# descriptors, metaclasses and dataclasses, plus TypeError cases. Setting enable_trace_unittest is
# done per test in setUp. Running the file calls the dynamo run_tests entry point.
# KNOWN ISSUE: the docstring says unbound-method calls are marked expectedFailure, but no test
# uses that marker; the unbound calls are asserted normally.
"""
Comprehensive tests for len() builtin and __len__() method protocol in PyTorch Dynamo.

Tests cover:
- len(obj) builtin calls
- obj.__len__() method calls
- type(obj).__len__(obj) unbound method calls (marked as expectedFailure - Dynamo limitation)
- Various container types: list, tuple, dict, set, frozenset, range, str, Tensor, nn.Module
- Dict views: keys(), values(), items()
- User-defined classes with __len__
"""

import collections
import dataclasses
import types

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import make_dynamo_test


class _BaseSequenceLen:
    # [CLASS DOC] HELPER: Mixin holding sequence len tests shared by list and tuple. The subclass
    # sets thetype to the concrete type; when it is still None every test is skipped. Not
    # collected on its own because it is not a TestCase.
    """Base class for testing len() on sequence types (list, tuple)"""

    thetype = None  # Override in subclass

    def setUp(self):
        # [FUNCTION DOC] HELPER: Fixture setup. Skips the test if thetype is None (the mixin used
        # directly), otherwise saves enable_trace_unittest in an attribute, enables it and calls
        # the parent setUp.
        if self.thetype is None:
            self.skipTest("Base class - not meant to be run directly")
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Fixture teardown. Restores the saved enable_trace_unittest config
        # value and returns the result of the parent tearDown.
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_len_basic(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Builds a three element sequence and checks that both the len builtin
        # and the dunder length method return three. The container type comes from the class
        # attribute thetype.
        seq = self.thetype([1, 2, 3])
        self.assertEqual(len(seq), 3)
        self.assertEqual(seq.__len__(), 3)

    @make_dynamo_test
    def test_len_empty(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Builds an empty sequence and checks that both ways of asking for the
        # length return zero. The container type comes from the class attribute thetype.
        seq = self.thetype([])
        self.assertEqual(len(seq), 0)
        self.assertEqual(seq.__len__(), 0)

    @make_dynamo_test
    def test_len_single_element(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Builds a one element sequence and checks both length forms return
        # one. The container type comes from the class attribute thetype.
        seq = self.thetype([42])
        self.assertEqual(len(seq), 1)
        self.assertEqual(seq.__len__(), 1)

    @make_dynamo_test
    def test_len_nested(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Builds a sequence containing three references to one inner sequence
        # and checks the outer length is three, not the flattened count. The container type comes
        # from the class attribute thetype.
        inner = self.thetype([1, 2])
        seq = self.thetype([inner, inner, inner])
        self.assertEqual(len(seq), 3)
        self.assertEqual(seq.__len__(), 3)

    @make_dynamo_test
    def test_len_with_tensors(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Builds a sequence of three scalar tensors and checks both length
        # forms return three. The container type comes from the class attribute thetype.
        seq = self.thetype([torch.tensor(1), torch.tensor(2), torch.tensor(3)])
        self.assertEqual(len(seq), 3)
        self.assertEqual(seq.__len__(), 3)

    @make_dynamo_test
    def test_len_with_mixed_types(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Builds a sequence mixing an int, a string, a float and a tensor and
        # checks both length forms return four. The container type comes from the class attribute
        # thetype.
        seq = self.thetype([1, "hello", 3.14, torch.tensor(4)])
        self.assertEqual(len(seq), 4)
        self.assertEqual(seq.__len__(), 4)

    @make_dynamo_test
    def test_len_large(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Builds a sequence from a range of one hundred numbers and checks
        # both length forms return one hundred. The container type comes from the class attribute
        # thetype.
        seq = self.thetype(range(100))
        self.assertEqual(len(seq), 100)
        self.assertEqual(seq.__len__(), 100)


class TestListLen(_BaseSequenceLen, torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Runs the shared sequence len tests with thetype set to list.
    """Tests for len() on list objects"""

    thetype = list


class TestTupleLen(_BaseSequenceLen, torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Runs the shared sequence len tests with thetype set to tuple.
    """Tests for len() on tuple objects"""

    thetype = tuple


class _BaseMappingLen:
    # [CLASS DOC] HELPER: Mixin holding mapping len tests shared by dict, OrderedDict and
    # defaultdict. Subclasses override get_mapping; the default builds a plain dict.
    """Base class for testing len() on mapping types (dict, OrderedDict)"""

    def get_mapping(self, items):
        # [FUNCTION DOC] HELPER: Factory hook. Takes an iterable of key/value pairs and returns a
        # plain dict built from it; subclasses override it to return another mapping type.
        """Override in subclass to return appropriate mapping type"""
        return dict(items)

    def setUp(self):
        # [FUNCTION DOC] HELPER: Fixture setup. Remembers the dynamo enable_trace_unittest config
        # value in an instance attribute, turns the flag on so unittest assertions can be traced
        # inside dynamo, then calls the parent setUp.
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Fixture teardown. Restores the saved enable_trace_unittest config
        # value and returns the result of the parent tearDown.
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_len_basic(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Builds a mapping of three integer keys and checks both length forms
        # return three. The mapping is built through get_mapping from an items view.
        d = self.get_mapping({1: "a", 2: "b", 3: "c"}.items())
        self.assertEqual(len(d), 3)
        self.assertEqual(d.__len__(), 3)

    @make_dynamo_test
    def test_len_empty(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Builds an empty mapping and checks both length forms return zero.
        # The mapping is built through get_mapping from an items view.
        d = self.get_mapping({}.items())
        self.assertEqual(len(d), 0)
        self.assertEqual(d.__len__(), 0)

    @make_dynamo_test
    def test_len_single_entry(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Builds a one entry mapping and checks both length forms return one.
        # The mapping is built through get_mapping from an items view.
        d = self.get_mapping({"key": "value"}.items())
        self.assertEqual(len(d), 1)
        self.assertEqual(d.__len__(), 1)

    @make_dynamo_test
    def test_len_string_keys(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Builds a mapping with three string keys and checks both length forms
        # return three. The mapping is built through get_mapping from an items view.
        d = self.get_mapping({"one": 1, "two": 2, "three": 3}.items())
        self.assertEqual(len(d), 3)
        self.assertEqual(d.__len__(), 3)

    @make_dynamo_test
    def test_len_int_keys(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Builds a mapping with four integer keys and checks both length forms
        # return four. The mapping is built through get_mapping from an items view.
        d = self.get_mapping({1: "one", 2: "two", 3: "three", 4: "four"}.items())
        self.assertEqual(len(d), 4)
        self.assertEqual(d.__len__(), 4)

    @make_dynamo_test
    def test_len_with_tensor_values(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Builds a mapping whose values are scalar tensors and checks both
        # length forms return three. The mapping is built through get_mapping from an items view.
        d = self.get_mapping(
            {"a": torch.tensor(1), "b": torch.tensor(2), "c": torch.tensor(3)}.items()
        )
        self.assertEqual(len(d), 3)
        self.assertEqual(d.__len__(), 3)

    @make_dynamo_test
    def test_len_large(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Builds a mapping of twenty entries from a comprehension and checks
        # both length forms return twenty. The mapping is built through get_mapping from an items
        # view.
        d = self.get_mapping({i: i * 2 for i in range(20)}.items())
        self.assertEqual(len(d), 20)
        self.assertEqual(d.__len__(), 20)


class TestDictLen(_BaseMappingLen, torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Runs the mapping len tests on plain dict; uses the default get_mapping.
    """Tests for len() on dict objects"""


class TestOrderedDictLen(_BaseMappingLen, torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Runs the mapping len tests on collections.OrderedDict.
    """Tests for len() on OrderedDict objects"""

    def get_mapping(self, items):
        # [FUNCTION DOC] HELPER: Returns an OrderedDict built from the given key/value pairs.
        return collections.OrderedDict(items)


class TestDefaultDictLen(_BaseMappingLen, torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Runs the mapping len tests on collections.defaultdict with an int default
    # factory.
    """Tests for len() on defaultdict objects"""

    def get_mapping(self, items):
        # [FUNCTION DOC] HELPER: Creates a defaultdict with an int default factory, copies every
        # key/value pair of the argument into it in a loop and returns it.
        d = collections.defaultdict(int)
        for k, v in items:
            d[k] = v
        return d


class _BaseSetLen:
    # [CLASS DOC] HELPER: Mixin holding set len tests shared by set and frozenset. Subclasses
    # override get_set. The class flag that disables pytest collection is set because the mixin is
    # not a TestCase.
    """Base class for testing len() on set types"""

    __test__ = False  # Prevent pytest from collecting this as a test class

    def get_set(self, items):
        # [FUNCTION DOC] HELPER: Factory hook. Takes an iterable and returns a plain set of its
        # elements; subclasses override it.
        """Override in subclass to return appropriate set type"""
        return set(items)

    def setUp(self):
        # [FUNCTION DOC] HELPER: Fixture setup. Remembers the dynamo enable_trace_unittest config
        # value in an instance attribute, turns the flag on so unittest assertions can be traced
        # inside dynamo, then calls the parent setUp.
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Fixture teardown. Restores the saved enable_trace_unittest config
        # value and returns the result of the parent tearDown.
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_len_basic(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Builds a set from three integers and checks both length forms return
        # three. The set is built through get_set.
        s = self.get_set([1, 2, 3])
        self.assertEqual(len(s), 3)
        self.assertEqual(s.__len__(), 3)

    @make_dynamo_test
    def test_len_empty(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Builds an empty set and checks both length forms return zero. The
        # set is built through get_set.
        s = self.get_set([])
        self.assertEqual(len(s), 0)
        self.assertEqual(s.__len__(), 0)

    @make_dynamo_test
    def test_len_single_element(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Builds a one element set and checks both length forms return one.
        # The set is built through get_set.
        s = self.get_set([42])
        self.assertEqual(len(s), 1)
        self.assertEqual(s.__len__(), 1)

    @make_dynamo_test
    def test_len_with_strings(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Builds a set of four strings and checks both length forms return
        # four. The set is built through get_set.
        s = self.get_set(["a", "b", "c", "d"])
        self.assertEqual(len(s), 4)
        self.assertEqual(s.__len__(), 4)

    @make_dynamo_test
    def test_len_with_duplicates(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Builds a set from a list with repeated values and checks that
        # duplicates are collapsed so both length forms return three. The set is built through
        # get_set.
        # Set constructor deduplicates
        s = self.get_set([1, 2, 2, 3, 3, 3])
        self.assertEqual(len(s), 3)
        self.assertEqual(s.__len__(), 3)

    @make_dynamo_test
    def test_len_large(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Builds a set from a range of fifty numbers and checks both length
        # forms return fifty. The set is built through get_set.
        s = self.get_set(range(50))
        self.assertEqual(len(s), 50)
        self.assertEqual(s.__len__(), 50)


class TestSetLen(_BaseSetLen, torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Runs the set len tests on set.
    """Tests for len() on set objects"""

    def get_set(self, items):
        # [FUNCTION DOC] HELPER: Returns a set of the given items.
        return set(items)


class TestFrozenSetLen(_BaseSetLen, torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Runs the set len tests on frozenset.
    """Tests for len() on frozenset objects"""

    def get_set(self, items):
        # [FUNCTION DOC] HELPER: Returns a frozenset of the given items.
        return frozenset(items)


class TestRangeLen(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Len tests for range objects. Each test checks three call forms: the builtin, the
    # instance dunder method and the unbound method taken from the range type.
    """Tests for len() on range objects"""

    def setUp(self):
        # [FUNCTION DOC] HELPER: Fixture setup. Remembers the dynamo enable_trace_unittest config
        # value in an instance attribute, turns the flag on so unittest assertions can be traced
        # inside dynamo, then calls the parent setUp.
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Fixture teardown. Restores the saved enable_trace_unittest config
        # value and returns the result of the parent tearDown.
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_len_basic(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Uses a range with only a stop value of five and checks all three
        # length forms return five.
        r = range(5)
        self.assertEqual(len(r), 5)
        self.assertEqual(r.__len__(), 5)
        self.assertEqual(range.__len__(r), 5)

    @make_dynamo_test
    def test_len_with_start_stop(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Uses a range from five to fifteen and checks all three length forms
        # return ten.
        r = range(5, 15)
        self.assertEqual(len(r), 10)
        self.assertEqual(r.__len__(), 10)
        self.assertEqual(range.__len__(r), 10)

    @make_dynamo_test
    def test_len_with_step(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Uses a range from zero to ten with step two and checks all three
        # length forms return five.
        r = range(0, 10, 2)
        self.assertEqual(len(r), 5)
        self.assertEqual(r.__len__(), 5)
        self.assertEqual(range.__len__(r), 5)

    @make_dynamo_test
    def test_len_negative_step(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Uses a range counting down from ten to one with step minus one and
        # checks all three length forms return ten.
        r = range(10, 0, -1)
        self.assertEqual(len(r), 10)
        self.assertEqual(r.__len__(), 10)
        self.assertEqual(range.__len__(r), 10)

    @make_dynamo_test
    def test_len_empty(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Uses a range whose start equals its stop and checks all three length
        # forms return zero.
        r = range(5, 5)
        self.assertEqual(len(r), 0)
        self.assertEqual(r.__len__(), 0)
        self.assertEqual(range.__len__(r), 0)


class TestStringLen(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Len tests for str objects. Each test checks the builtin, the instance dunder
    # method and the unbound method taken from the str type.
    """Tests for len() on string objects"""

    def setUp(self):
        # [FUNCTION DOC] HELPER: Fixture setup. Remembers the dynamo enable_trace_unittest config
        # value in an instance attribute, turns the flag on so unittest assertions can be traced
        # inside dynamo, then calls the parent setUp.
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Fixture teardown. Restores the saved enable_trace_unittest config
        # value and returns the result of the parent tearDown.
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_len_empty(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Checks the empty string has length zero in all three call forms.
        s = ""
        self.assertEqual(len(s), 0)
        self.assertEqual(s.__len__(), 0)
        self.assertEqual(str.__len__(s), 0)

    @make_dynamo_test
    def test_len_single_char(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Checks a one character string has length one in all three call
        # forms.
        s = "a"
        self.assertEqual(len(s), 1)
        self.assertEqual(s.__len__(), 1)
        self.assertEqual(str.__len__(s), 1)

    @make_dynamo_test
    def test_len_multiple_chars(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Checks the string hello has length five in all three call forms.
        s = "hello"
        self.assertEqual(len(s), 5)
        self.assertEqual(s.__len__(), 5)
        self.assertEqual(str.__len__(s), 5)

    @make_dynamo_test
    def test_len_with_spaces(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Checks a string with an embedded space has length eleven in all
        # three call forms.
        s = "hello world"
        self.assertEqual(len(s), 11)
        self.assertEqual(s.__len__(), 11)
        self.assertEqual(str.__len__(s), 11)


class TestTensorLen(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Len tests for torch.Tensor; the length is the size of the first dimension. Each
    # test checks the builtin, the instance method and the unbound method taken from the Tensor
    # class.
    """Tests for len() on torch.Tensor objects"""

    def setUp(self):
        # [FUNCTION DOC] HELPER: Fixture setup. Remembers the dynamo enable_trace_unittest config
        # value in an instance attribute, turns the flag on so unittest assertions can be traced
        # inside dynamo, then calls the parent setUp.
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Fixture teardown. Restores the saved enable_trace_unittest config
        # value and returns the result of the parent tearDown.
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_len_1d(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. One dimensional tensor of five integers has length five in all three
        # call forms.
        t = torch.tensor([1, 2, 3, 4, 5])
        self.assertEqual(len(t), 5)
        self.assertEqual(t.__len__(), 5)
        self.assertEqual(torch.Tensor.__len__(t), 5)

    @make_dynamo_test
    def test_len_2d(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Tensor with shape three by two has length three, the first
        # dimension, in all three forms.
        t = torch.tensor([[1, 2], [3, 4], [5, 6]])
        self.assertEqual(len(t), 3)
        self.assertEqual(t.__len__(), 3)
        self.assertEqual(torch.Tensor.__len__(t), 3)

    @make_dynamo_test
    def test_len_3d(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Random tensor with shape four by five by six has length four in all
        # three forms.
        t = torch.randn(4, 5, 6)
        self.assertEqual(len(t), 4)
        self.assertEqual(t.__len__(), 4)
        self.assertEqual(torch.Tensor.__len__(t), 4)

    @make_dynamo_test
    def test_len_empty(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. An empty one dimensional tensor has length zero in all three forms.
        t = torch.tensor([])
        self.assertEqual(len(t), 0)
        self.assertEqual(t.__len__(), 0)
        self.assertEqual(torch.Tensor.__len__(t), 0)

    @make_dynamo_test
    def test_len_large_batch(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Random tensor with shape one hundred by five by five has length one
        # hundred in all three forms.
        t = torch.randn(100, 5, 5)
        self.assertEqual(len(t), 100)
        self.assertEqual(t.__len__(), 100)
        self.assertEqual(torch.Tensor.__len__(t), 100)

    @make_dynamo_test
    def test_len_different_dtypes(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. A float32 tensor of three elements has length three in all three
        # forms.
        t = torch.tensor([1, 2, 3], dtype=torch.float32)
        self.assertEqual(len(t), 3)
        self.assertEqual(t.__len__(), 3)
        self.assertEqual(torch.Tensor.__len__(t), 3)


class TestNNModuleLen(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Len tests for nn.Module containers (Sequential, ModuleList, ModuleDict). The
    # instances are built once in setUp so they are not constructed inside the traced test bodies.
    """Tests for len() on torch.nn module containers"""

    def setUp(self):
        # [FUNCTION DOC] HELPER: Fixture setup. Saves and enables enable_trace_unittest, calls the
        # parent setUp, then stores on self a three layer Sequential, an empty Sequential, a two
        # item ModuleList, an empty ModuleList, a two item ModuleDict, an empty ModuleDict and a
        # five layer Sequential.
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

        # Pre-construct nn.Module instances outside compiled regions
        self.seq_3layers = torch.nn.Sequential(
            torch.nn.Linear(10, 20), torch.nn.ReLU(), torch.nn.Linear(20, 5)
        )
        self.empty_seq = torch.nn.Sequential()
        self.ml_2modules = torch.nn.ModuleList(
            [torch.nn.Linear(10, 20), torch.nn.Linear(20, 30)]
        )
        self.empty_ml = torch.nn.ModuleList()
        self.md_2modules = torch.nn.ModuleDict(
            {"layer1": torch.nn.Linear(10, 20), "layer2": torch.nn.Linear(20, 5)}
        )
        self.empty_md = torch.nn.ModuleDict()
        self.seq_5layers = torch.nn.Sequential(
            *[torch.nn.Linear(10, 10) for _ in range(5)]
        )

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Fixture teardown. Restores the saved enable_trace_unittest config
        # value and returns the result of the parent tearDown.
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_len_sequential(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. The three layer Sequential has length three in both call forms.
        seq = self.seq_3layers
        self.assertEqual(len(seq), 3)
        self.assertEqual(seq.__len__(), 3)

    @make_dynamo_test
    def test_len_empty_sequential(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. The empty Sequential has length zero in both call forms.
        seq = self.empty_seq
        self.assertEqual(len(seq), 0)
        self.assertEqual(seq.__len__(), 0)

    @make_dynamo_test
    def test_len_module_list(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. The two item ModuleList has length two in both call forms.
        ml = self.ml_2modules
        self.assertEqual(len(ml), 2)
        self.assertEqual(ml.__len__(), 2)

    @make_dynamo_test
    def test_len_empty_module_list(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. The empty ModuleList has length zero in both call forms.
        ml = self.empty_ml
        self.assertEqual(len(ml), 0)
        self.assertEqual(ml.__len__(), 0)

    @make_dynamo_test
    def test_len_module_dict(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. The two item ModuleDict has length two in both call forms.
        md = self.md_2modules
        self.assertEqual(len(md), 2)
        self.assertEqual(md.__len__(), 2)

    @make_dynamo_test
    def test_len_empty_module_dict(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. The empty ModuleDict has length zero in both call forms.
        md = self.empty_md
        self.assertEqual(len(md), 0)
        self.assertEqual(md.__len__(), 0)

    @make_dynamo_test
    def test_len_sequential_with_multiple_layers(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. The five layer Sequential has length five in both call forms.
        seq = self.seq_5layers
        self.assertEqual(len(seq), 5)
        self.assertEqual(seq.__len__(), 5)


class TestDictViewLen(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Len tests for dict views returned by the keys, values and items methods; each
    # test checks the builtin and the instance dunder method.
    """Tests for len() on dict view objects (keys, values, items)"""

    def setUp(self):
        # [FUNCTION DOC] HELPER: Fixture setup. Remembers the dynamo enable_trace_unittest config
        # value in an instance attribute, turns the flag on so unittest assertions can be traced
        # inside dynamo, then calls the parent setUp.
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Fixture teardown. Restores the saved enable_trace_unittest config
        # value and returns the result of the parent tearDown.
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_len_dict_keys_view(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Keys view of a three entry dict has length three.
        d = {"a": 1, "b": 2, "c": 3}
        keys = d.keys()
        self.assertEqual(len(keys), 3)
        self.assertEqual(keys.__len__(), 3)

    @make_dynamo_test
    def test_len_dict_values_view(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Values view of a three entry dict has length three.
        d = {"a": 1, "b": 2, "c": 3}
        values = d.values()
        self.assertEqual(len(values), 3)
        self.assertEqual(values.__len__(), 3)

    @make_dynamo_test
    def test_len_dict_items_view(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Items view of a three entry dict has length three.
        d = {"x": 10, "y": 20, "z": 30}
        items = d.items()
        self.assertEqual(len(items), 3)
        self.assertEqual(items.__len__(), 3)

    @make_dynamo_test
    def test_len_dict_keys_empty(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Keys view of an empty dict has length zero.
        d = {}
        keys = d.keys()
        self.assertEqual(len(keys), 0)
        self.assertEqual(keys.__len__(), 0)

    @make_dynamo_test
    def test_len_dict_values_empty(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Values view of an empty dict has length zero.
        d = {}
        values = d.values()
        self.assertEqual(len(values), 0)
        self.assertEqual(values.__len__(), 0)

    @make_dynamo_test
    def test_len_dict_items_empty(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Items view of an empty dict has length zero.
        d = {}
        items = d.items()
        self.assertEqual(len(items), 0)
        self.assertEqual(items.__len__(), 0)

    @make_dynamo_test
    def test_len_dict_keys_single_entry(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Keys view of a one entry dict has length one.
        """Test len() on dict.keys() with single entry"""
        d = {"key": "value"}
        keys = d.keys()
        self.assertEqual(len(keys), 1)
        self.assertEqual(keys.__len__(), 1)

    @make_dynamo_test
    def test_len_dict_values_single_entry(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Values view of a one entry dict has length one.
        """Test len() on dict.values() with single entry"""
        d = {"key": "value"}
        values = d.values()
        self.assertEqual(len(values), 1)
        self.assertEqual(values.__len__(), 1)

    @make_dynamo_test
    def test_len_dict_items_single_entry(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Items view of a one entry dict has length one.
        """Test len() on dict.items() with single entry"""
        d = {"key": "value"}
        items = d.items()
        self.assertEqual(len(items), 1)
        self.assertEqual(items.__len__(), 1)

    @make_dynamo_test
    def test_len_dict_keys_int_keys(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Keys view of a dict with four integer keys has length four.
        """Test len() on dict.keys() with integer keys"""
        d = {1: "one", 2: "two", 3: "three", 4: "four"}
        keys = d.keys()
        self.assertEqual(len(keys), 4)
        self.assertEqual(keys.__len__(), 4)

    @make_dynamo_test
    def test_len_dict_values_tensor_values(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Values view of a dict whose values are tensors has length three.
        """Test len() on dict.values() with tensor values"""
        d = {"a": torch.tensor(1), "b": torch.tensor(2), "c": torch.tensor(3)}
        values = d.values()
        self.assertEqual(len(values), 3)
        self.assertEqual(values.__len__(), 3)

    @make_dynamo_test
    def test_len_dict_items_mixed_types(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Items view of a dict with string, integer and tuple keys and mixed
        # value types has length three.
        """Test len() on dict.items() with mixed key/value types"""
        d = {"str": "value", 42: torch.tensor(1), (1, 2): "tuple_key"}
        items = d.items()
        self.assertEqual(len(items), 3)
        self.assertEqual(items.__len__(), 3)

    @make_dynamo_test
    def test_len_dict_keys_large(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Keys view of a fifty entry dict has length fifty.
        """Test len() on dict.keys() with large number of entries"""
        d = {i: i * 2 for i in range(50)}
        keys = d.keys()
        self.assertEqual(len(keys), 50)
        self.assertEqual(keys.__len__(), 50)

    @make_dynamo_test
    def test_len_dict_values_large(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Values view of a fifty entry dict has length fifty.
        """Test len() on dict.values() with large number of entries"""
        d = {i: i * 2 for i in range(50)}
        values = d.values()
        self.assertEqual(len(values), 50)
        self.assertEqual(values.__len__(), 50)

    @make_dynamo_test
    def test_len_dict_items_large(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Items view of a fifty entry dict has length fifty.
        """Test len() on dict.items() with large number of entries"""
        d = {i: i * 2 for i in range(50)}
        items = d.items()
        self.assertEqual(len(items), 50)
        self.assertEqual(items.__len__(), 50)


# User-defined classes for TestUserDefinedLen
class CustomList:
    # [CLASS DOC] HELPER: Plain user class that stores a list in the items attribute and reports
    # its length through the dunder length method.
    def __init__(self, items):
        # [FUNCTION DOC] HELPER: Stores the given items in the items attribute.
        self.items = items

    def __len__(self):
        # [FUNCTION DOC] HELPER: Returns the length of the stored items.
        return len(self.items)


class CustomContainer:
    # [CLASS DOC] HELPER: User class with no state whose dunder length method always reports zero.
    def __len__(self):
        # [FUNCTION DOC] HELPER: Always returns zero.
        return 0


class Container:
    # [CLASS DOC] HELPER: User class holding an integer size attribute that is reported as its
    # length.
    def __init__(self, size):
        # [FUNCTION DOC] HELPER: Stores the given size in the size attribute.
        self.size = size

    def __len__(self):
        # [FUNCTION DOC] HELPER: Returns the stored size.
        return self.size


class FixedSize:
    # [CLASS DOC] HELPER: Stateless user class whose length is always ten.
    def __len__(self):
        # [FUNCTION DOC] HELPER: Always returns ten.
        return 10


class ListWrapper:
    # [CLASS DOC] HELPER: User class owning a fixed five element list in the data attribute; its
    # length is that of the list.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Creates the data attribute holding the list of integers one to
        # five.
        self.data = [1, 2, 3, 4, 5]

    def __len__(self):
        # [FUNCTION DOC] HELPER: Returns the length of the data list.
        return len(self.data)


class ListSubclassCustomLen(list):
    # [CLASS DOC] HELPER: List subclass whose dunder length method reports twice the real number
    # of elements.
    def __len__(self):
        # [FUNCTION DOC] HELPER: Returns twice the length reported by the list base class.
        return super().__len__() * 2


class CustomMapping:
    # [CLASS DOC] HELPER: Dict-like user class wrapping a dictionary in the private _data
    # attribute and forwarding length, item access, keys, values and items to it. Not a real dict
    # subclass.
    """A user-defined mapping (dict-like) class"""

    def __init__(self, data):
        # [FUNCTION DOC] HELPER: Stores the argument as _data, using it directly when it is
        # already a dict and otherwise converting it to a dict.
        self._data = dict(data) if not isinstance(data, dict) else data

    def __len__(self):
        # [FUNCTION DOC] HELPER: Returns the number of entries in the wrapped dict.
        return len(self._data)

    def __getitem__(self, key):
        # [FUNCTION DOC] HELPER: Returns the value stored under the given key in the wrapped dict;
        # a missing key raises KeyError from the dict.
        return self._data[key]

    def keys(self):
        # [FUNCTION DOC] HELPER: Returns the keys view of the wrapped dict.
        return self._data.keys()

    def values(self):
        # [FUNCTION DOC] HELPER: Returns the values view of the wrapped dict.
        return self._data.values()

    def items(self):
        # [FUNCTION DOC] HELPER: Returns the items view of the wrapped dict.
        return self._data.items()


class CustomMappingSubclass(CustomMapping):
    # [CLASS DOC] HELPER: Subclass of CustomMapping whose length is one more than the wrapped dict
    # size.
    """A subclass of CustomMapping"""

    def __len__(self):
        # [FUNCTION DOC] HELPER: Returns the parent length plus one.
        # Custom len implementation (e.g., filtered length)
        return super().__len__() + 1


class SimpleDictLike:
    # [CLASS DOC] HELPER: Minimal user class that stores a size and reports it as its length;
    # despite the name it has no mapping methods.
    """Minimal dict-like class with just __len__"""

    def __init__(self, size):
        # [FUNCTION DOC] HELPER: Stores the given size in the size attribute.
        self.size = size

    def __len__(self):
        # [FUNCTION DOC] HELPER: Returns the stored size.
        return self.size


class TupleSubclassCustomLen(tuple):
    # [CLASS DOC] HELPER: Tuple subclass with empty slots whose reported length is one more than
    # the real element count.
    __slots__ = ()

    def __len__(self):
        # [FUNCTION DOC] HELPER: Returns the tuple base class length plus one.
        return super().__len__() + 1


class DictSubclassCustomLen(dict):
    # [CLASS DOC] HELPER: Dict subclass whose reported length is one less than the real entry
    # count.
    def __len__(self):
        # [FUNCTION DOC] HELPER: Returns the dict base class length minus one.
        return super().__len__() - 1


class SetSubclassCustomLen(set):
    # [CLASS DOC] HELPER: Set subclass whose reported length is always zero.
    def __len__(self):
        # [FUNCTION DOC] HELPER: Always returns zero regardless of contents.
        return 0


class TestUserDefinedLen(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Len tests for user defined classes that implement the dunder length method; each
    # test checks the builtin, the instance method and the unbound method taken from the class.
    """Tests for len() on user-defined classes with __len__"""

    def setUp(self):
        # [FUNCTION DOC] HELPER: Fixture setup. Remembers the dynamo enable_trace_unittest config
        # value in an instance attribute, turns the flag on so unittest assertions can be traced
        # inside dynamo, then calls the parent setUp.
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Fixture teardown. Restores the saved enable_trace_unittest config
        # value and returns the result of the parent tearDown.
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_len_simple_custom_class(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. CustomList holding three items reports length three in all three
        # call forms.
        obj = CustomList([1, 2, 3])
        self.assertEqual(len(obj), 3)
        self.assertEqual(obj.__len__(), 3)
        self.assertEqual(CustomList.__len__(obj), 3)

    @make_dynamo_test
    def test_len_custom_class_empty(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. CustomContainer reports length zero in all three call forms.
        obj = CustomContainer()
        self.assertEqual(len(obj), 0)
        self.assertEqual(obj.__len__(), 0)
        self.assertEqual(CustomContainer.__len__(obj), 0)

    @make_dynamo_test
    def test_len_custom_class_with_properties(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Container created with size forty two reports forty two in all three
        # call forms.
        obj = Container(42)
        self.assertEqual(len(obj), 42)
        self.assertEqual(obj.__len__(), 42)
        self.assertEqual(Container.__len__(obj), 42)

    @make_dynamo_test
    def test_len_custom_class_constant_return(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. FixedSize reports ten in all three call forms.
        obj = FixedSize()
        self.assertEqual(len(obj), 10)
        self.assertEqual(obj.__len__(), 10)
        self.assertEqual(FixedSize.__len__(obj), 10)

    @make_dynamo_test
    def test_len_custom_class_with_list_attr(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. ListWrapper reports five, the length of its internal list, in all
        # three call forms.
        obj = ListWrapper()
        self.assertEqual(len(obj), 5)
        self.assertEqual(obj.__len__(), 5)
        self.assertEqual(ListWrapper.__len__(obj), 5)


class TestSubclassOverloadedLen(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Len tests for subclasses of builtin containers that override the dunder length
    # method; the overridden value must win over the builtin one.
    """Tests for custom classes that inherit from builtins and overload __len__"""

    def setUp(self):
        # [FUNCTION DOC] HELPER: Fixture setup. Remembers the dynamo enable_trace_unittest config
        # value in an instance attribute, turns the flag on so unittest assertions can be traced
        # inside dynamo, then calls the parent setUp.
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Fixture teardown. Restores the saved enable_trace_unittest config
        # value and returns the result of the parent tearDown.
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_list_subclass_custom_len(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. List subclass with three items reports six (doubled) in all three
        # call forms.
        obj = ListSubclassCustomLen([1, 2, 3])
        self.assertEqual(len(obj), 6)
        self.assertEqual(obj.__len__(), 6)
        self.assertEqual(ListSubclassCustomLen.__len__(obj), 6)

    @make_dynamo_test
    def test_tuple_subclass_custom_len(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Tuple subclass with three items reports four (one extra) in all
        # three call forms.
        obj = TupleSubclassCustomLen([1, 2, 3])
        self.assertEqual(len(obj), 4)
        self.assertEqual(obj.__len__(), 4)
        self.assertEqual(TupleSubclassCustomLen.__len__(obj), 4)

    @make_dynamo_test
    def test_dict_subclass_custom_len(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Dict subclass with three entries reports two (one fewer) in all
        # three call forms.
        obj = DictSubclassCustomLen({"a": 1, "b": 2, "c": 3})
        self.assertEqual(len(obj), 2)
        self.assertEqual(obj.__len__(), 2)
        self.assertEqual(DictSubclassCustomLen.__len__(obj), 2)

    @make_dynamo_test
    def test_set_subclass_custom_len(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Set subclass with three items reports zero in all three call forms.
        obj = SetSubclassCustomLen([1, 2, 3])
        self.assertEqual(len(obj), 0)
        self.assertEqual(obj.__len__(), 0)
        self.assertEqual(SetSubclassCustomLen.__len__(obj), 0)


class DescriptorLenClass:
    # [CLASS DOC] HELPER: Fixture class whose length is a regular instance method that returns
    # forty.
    """Test class with __len__ as a regular instance method"""

    def __len__(self):
        # [FUNCTION DOC] HELPER: Returns forty.
        """Regular instance method __len__ - should return 40"""
        return 40


class PartialLenClass:
    # [CLASS DOC] HELPER: Fixture class with a private five element list; its length is that of
    # the list. The docstring talks about a callable object but the code is an ordinary method.
    """Class where __len__ is a lambda/callable object"""

    def __init__(self):
        # [FUNCTION DOC] HELPER: Creates the _items attribute holding the integers one to five.
        self._items = [1, 2, 3, 4, 5]

    def __len__(self):
        # [FUNCTION DOC] HELPER: Returns the length of the _items list.
        return len(self._items)


class StaticMethodLenClass:
    # [CLASS DOC] HELPER: Fixture class whose dunder length method is a staticmethod taking no
    # arguments.
    """Test class where __len__ is a staticmethod (unusual, likely to fail)"""

    @staticmethod
    def __len__():
        # [FUNCTION DOC] HELPER: Takes no arguments and returns ten.
        """Staticmethod __len__ - unusual pattern"""
        return 10


class ClassMethodLenClass:
    # [CLASS DOC] HELPER: Fixture class whose dunder length method is a classmethod.
    """Test class where __len__ is a classmethod (unusual, likely to fail)"""

    @classmethod
    def __len__(cls):
        # [FUNCTION DOC] HELPER: Receives the class and returns twenty.
        """Classmethod __len__ - unusual pattern"""
        return 20


class CustomDescriptorLenClass:
    # [CLASS DOC] HELPER: Fixture class whose dunder length attribute is an instance of a nested
    # custom descriptor class, so the length callable is resolved through the descriptor protocol.
    """Test class where __len__ is a custom descriptor"""

    class CustomDescriptorLen:
        # [CLASS DOC] HELPER: Descriptor type with only a get method. Accessed on an instance it
        # yields a zero argument callable returning fifty.
        """Custom descriptor that implements __get__ method"""

        def __get__(self, obj, objtype=None):
            # [FUNCTION DOC] HELPER: Descriptor get hook. Returns the descriptor itself when
            # accessed on the class (no instance), otherwise a lambda taking no arguments that
            # returns fifty.
            """Descriptor protocol: return a callable that returns len"""
            if obj is None:
                return self
            return lambda: 50

    __len__ = CustomDescriptorLen()


class TestUserDefinedMappingLen(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Len tests for user defined dict-like classes; each test checks the builtin and
    # the instance dunder method.
    """Tests for len() on user-defined mapping (dict-like) classes"""

    def setUp(self):
        # [FUNCTION DOC] HELPER: Fixture setup. Remembers the dynamo enable_trace_unittest config
        # value in an instance attribute, turns the flag on so unittest assertions can be traced
        # inside dynamo, then calls the parent setUp.
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Fixture teardown. Restores the saved enable_trace_unittest config
        # value and returns the result of the parent tearDown.
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_len_custom_mapping_basic(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. CustomMapping with three entries has length three.
        """Test len() on a basic custom mapping class"""
        m = CustomMapping({"a": 1, "b": 2, "c": 3})
        self.assertEqual(len(m), 3)
        self.assertEqual(m.__len__(), 3)

    @make_dynamo_test
    def test_len_custom_mapping_empty(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. CustomMapping with no entries has length zero.
        """Test len() on an empty custom mapping"""
        m = CustomMapping({})
        self.assertEqual(len(m), 0)
        self.assertEqual(m.__len__(), 0)

    @make_dynamo_test
    def test_len_custom_mapping_single_entry(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. CustomMapping with one entry has length one.
        """Test len() on a custom mapping with single entry"""
        m = CustomMapping({"key": "value"})
        self.assertEqual(len(m), 1)
        self.assertEqual(m.__len__(), 1)

    @make_dynamo_test
    def test_len_custom_mapping_string_keys(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. CustomMapping with three string keys has length three.
        """Test len() on a custom mapping with string keys"""
        m = CustomMapping({"one": 1, "two": 2, "three": 3})
        self.assertEqual(len(m), 3)
        self.assertEqual(m.__len__(), 3)

    @make_dynamo_test
    def test_len_custom_mapping_int_keys(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. CustomMapping with four integer keys has length four.
        """Test len() on a custom mapping with int keys"""
        m = CustomMapping({1: "one", 2: "two", 3: "three", 4: "four"})
        self.assertEqual(len(m), 4)
        self.assertEqual(m.__len__(), 4)

    @make_dynamo_test
    def test_len_custom_mapping_with_tensor_values(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. CustomMapping whose values are tensors has length three.
        """Test len() on a custom mapping with tensor values"""
        m = CustomMapping(
            {"a": torch.tensor(1), "b": torch.tensor(2), "c": torch.tensor(3)}
        )
        self.assertEqual(len(m), 3)
        self.assertEqual(m.__len__(), 3)

    @make_dynamo_test
    def test_len_custom_mapping_large(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. CustomMapping with twenty entries has length twenty.
        """Test len() on a large custom mapping"""
        m = CustomMapping({i: i * 2 for i in range(20)})
        self.assertEqual(len(m), 20)
        self.assertEqual(m.__len__(), 20)

    @make_dynamo_test
    def test_len_custom_mapping_subclass(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. CustomMappingSubclass with two entries reports three because its
        # length adds one.
        """Test len() on a subclass of CustomMapping"""
        m = CustomMappingSubclass({"a": 1, "b": 2})
        # CustomMappingSubclass.__len__ returns len + 1
        self.assertEqual(len(m), 3)
        self.assertEqual(m.__len__(), 3)

    @make_dynamo_test
    def test_len_simple_dict_like(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. SimpleDictLike created with size forty two reports forty two.
        """Test len() on a minimal dict-like class"""
        m = SimpleDictLike(42)
        self.assertEqual(len(m), 42)
        self.assertEqual(m.__len__(), 42)


class TestDescriptorLenImpl(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests that len handles dunder length attributes that are descriptors (regular
    # method, staticmethod, classmethod, custom descriptor). These tests only check the builtin
    # form.
    """Test that len_impl handles descriptor-based __len__ correctly"""

    def setUp(self):
        # [FUNCTION DOC] HELPER: Fixture setup. Remembers the dynamo enable_trace_unittest config
        # value in an instance attribute, turns the flag on so unittest assertions can be traced
        # inside dynamo, then calls the parent setUp.
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Fixture teardown. Restores the saved enable_trace_unittest config
        # value and returns the result of the parent tearDown.
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_regular_instance_len(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. DescriptorLenClass reports forty through both the builtin and the
        # instance method.
        """Test regular instance method __len__"""
        obj = DescriptorLenClass()

        # Regular instance methods should work fine
        self.assertEqual(len(obj), 40)
        self.assertEqual(obj.__len__(), 40)

    @make_dynamo_test
    def test_partial_callable_len(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. PartialLenClass reports five through both the builtin and the
        # instance method.
        """Test lambda/callable as __len__"""
        obj = PartialLenClass()

        # Callable __len__ should work
        self.assertEqual(len(obj), 5)
        self.assertEqual(obj.__len__(), 5)

    @make_dynamo_test
    def test_staticmethod_len_works(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. StaticMethodLenClass reports ten through the builtin; the
        # staticmethod descriptor unwraps without binding the instance.
        """Test that staticmethod as __len__ actually works (unusual but supported)

        Staticmethods are descriptors that don't bind. CPython resolves the
        descriptor and calls the underlying function without passing self.
        """
        obj = StaticMethodLenClass()

        # Surprisingly, CPython's descriptor protocol makes this work
        # staticmethod.__get__ returns the unwrapped function
        self.assertEqual(len(obj), 10)

    @make_dynamo_test
    def test_classmethod_len_works(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. ClassMethodLenClass reports twenty through the builtin; the
        # classmethod descriptor binds the class.
        """Test that classmethod as __len__ actually works (unusual but supported)

        Classmethods are descriptors that bind to the class. CPython's
        descriptor protocol handles this and passes the class instead of instance.
        """
        obj = ClassMethodLenClass()

        # Surprisingly, CPython's descriptor protocol makes this work
        # classmethod.__get__ returns a bound method with the class
        self.assertEqual(len(obj), 20)

    @make_dynamo_test
    def test_custom_descriptor_len(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. CustomDescriptorLenClass reports fifty through the builtin; the
        # callable comes from the descriptor get hook.
        """Test custom descriptor with __get__ method as __len__

        Custom descriptors implement the descriptor protocol via __get__.
        The descriptor is resolved and the returned callable is used as __len__.
        """
        obj = CustomDescriptorLenClass()

        # Custom descriptor's __get__ returns a callable that returns 50
        self.assertEqual(len(obj), 50)


class TestRaisesTypeError(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Len tests for objects that have no length: len must raise TypeError, as in plain
    # Python.
    """Tests for types that don't support len() - should raise TypeError like Python"""

    def setUp(self):
        # [FUNCTION DOC] HELPER: Fixture setup. Remembers the dynamo enable_trace_unittest config
        # value in an instance attribute, turns the flag on so unittest assertions can be traced
        # inside dynamo, then calls the parent setUp.
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Fixture teardown. Restores the saved enable_trace_unittest config
        # value and returns the result of the parent tearDown.
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_len_slice_raises_type_error(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Calling len on a slice with start and stop must raise TypeError.
        """slice objects do not support len() - should raise TypeError"""
        s = slice(1, 5)
        with self.assertRaises(TypeError):
            len(s)

    @make_dynamo_test
    def test_len_slice_with_step_raises_type_error(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Calling len on a slice that also has a step must raise TypeError.
        """slice with step also raises TypeError"""
        s = slice(0, 10, 2)
        with self.assertRaises(TypeError):
            len(s)

    @make_dynamo_test
    def test_len_list_iterator_raises_type_error(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Calling len on an iterator over a three element list must raise
        # TypeError.
        """list iterator does not support len() - should raise TypeError"""
        it = iter([1, 2, 3])
        with self.assertRaises(TypeError):
            len(it)

    @make_dynamo_test
    def test_len_empty_list_iterator_raises_type_error(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Calling len on an iterator over an empty list must raise TypeError.
        """empty list iterator also raises TypeError"""
        it = iter([])
        with self.assertRaises(TypeError):
            len(it)

    @make_dynamo_test
    def test_len_tuple_iterator_raises_type_error(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Calling len on an iterator over a tuple must raise TypeError.
        """tuple iterator does not support len() - should raise TypeError"""
        it = iter((1, 2, 3))
        with self.assertRaises(TypeError):
            len(it)

    @make_dynamo_test
    def test_len_range_iterator_raises_type_error(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Calling len on an iterator over a range must raise TypeError.
        """range iterator does not support len() - should raise TypeError"""
        it = iter(range(5))
        with self.assertRaises(TypeError):
            len(it)

    @make_dynamo_test
    def test_len_dict_iterator_raises_type_error(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Calling len on an iterator over the keys of a dict must raise
        # TypeError.
        """dict iterator (keys) does not support len() - should raise TypeError"""
        d = {"a": 1, "b": 2}
        it = iter(d)
        with self.assertRaises(TypeError):
            len(it)


class TestDequeLen(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Len tests for collections.deque; each test checks the builtin and the instance
    # dunder method.
    """Tests for len() on collections.deque objects"""

    def setUp(self):
        # [FUNCTION DOC] HELPER: Fixture setup. Remembers the dynamo enable_trace_unittest config
        # value in an instance attribute, turns the flag on so unittest assertions can be traced
        # inside dynamo, then calls the parent setUp.
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Fixture teardown. Restores the saved enable_trace_unittest config
        # value and returns the result of the parent tearDown.
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_len_basic(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Deque of three integers has length three.
        d = collections.deque([1, 2, 3])
        self.assertEqual(len(d), 3)
        self.assertEqual(d.__len__(), 3)

    @make_dynamo_test
    def test_len_empty(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Empty deque has length zero.
        d = collections.deque([])
        self.assertEqual(len(d), 0)
        self.assertEqual(d.__len__(), 0)

    @make_dynamo_test
    def test_len_single_element(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Deque of one element has length one.
        d = collections.deque([42])
        self.assertEqual(len(d), 1)
        self.assertEqual(d.__len__(), 1)

    @make_dynamo_test
    def test_len_with_maxlen(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Deque built from five items with a maximum length of three keeps
        # only three, so the length is three.
        d = collections.deque([1, 2, 3, 4, 5], maxlen=3)
        self.assertEqual(len(d), 3)
        self.assertEqual(d.__len__(), 3)

    @make_dynamo_test
    def test_len_with_strings(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Deque of four strings has length four.
        d = collections.deque(["a", "b", "c", "d"])
        self.assertEqual(len(d), 4)
        self.assertEqual(d.__len__(), 4)

    @make_dynamo_test
    def test_len_large(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Deque built from fifty numbers has length fifty.
        d = collections.deque(range(50))
        self.assertEqual(len(d), 50)
        self.assertEqual(d.__len__(), 50)


class TestMappingProxyLen(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Len tests for types.MappingProxyType read-only mapping views; each test checks
    # the builtin and the instance dunder method.
    """Tests for len() on types.MappingProxyType objects"""

    def setUp(self):
        # [FUNCTION DOC] HELPER: Fixture setup. Remembers the dynamo enable_trace_unittest config
        # value in an instance attribute, turns the flag on so unittest assertions can be traced
        # inside dynamo, then calls the parent setUp.
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Fixture teardown. Restores the saved enable_trace_unittest config
        # value and returns the result of the parent tearDown.
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_len_basic(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Proxy over a three entry dict has length three.
        """Test len() on a basic MappingProxyType"""
        d = types.MappingProxyType({"a": 1, "b": 2, "c": 3})
        self.assertEqual(len(d), 3)
        self.assertEqual(d.__len__(), 3)

    @make_dynamo_test
    def test_len_empty(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Proxy over an empty dict has length zero.
        """Test len() on an empty MappingProxyType"""
        d = types.MappingProxyType({})
        self.assertEqual(len(d), 0)
        self.assertEqual(d.__len__(), 0)

    @make_dynamo_test
    def test_len_single_entry(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Proxy over a one entry dict has length one.
        """Test len() on a MappingProxyType with single entry"""
        d = types.MappingProxyType({"key": "value"})
        self.assertEqual(len(d), 1)
        self.assertEqual(d.__len__(), 1)

    @make_dynamo_test
    def test_len_string_keys(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Proxy over a dict with three string keys has length three.
        """Test len() on a MappingProxyType with string keys"""
        d = types.MappingProxyType({"one": 1, "two": 2, "three": 3})
        self.assertEqual(len(d), 3)
        self.assertEqual(d.__len__(), 3)

    @make_dynamo_test
    def test_len_int_keys(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Proxy over a dict with four integer keys has length four.
        """Test len() on a MappingProxyType with int keys"""
        d = types.MappingProxyType({1: "one", 2: "two", 3: "three", 4: "four"})
        self.assertEqual(len(d), 4)
        self.assertEqual(d.__len__(), 4)

    @make_dynamo_test
    def test_len_with_tensor_values(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Proxy over a dict with tensor values has length three.
        """Test len() on a MappingProxyType with tensor values"""
        d = types.MappingProxyType(
            {"a": torch.tensor(1), "b": torch.tensor(2), "c": torch.tensor(3)}
        )
        self.assertEqual(len(d), 3)
        self.assertEqual(d.__len__(), 3)

    @make_dynamo_test
    def test_len_large(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Proxy over a twenty entry dict has length twenty.
        """Test len() on a large MappingProxyType"""
        d = types.MappingProxyType({i: i * 2 for i in range(20)})
        self.assertEqual(len(d), 20)
        self.assertEqual(d.__len__(), 20)


class MetaclassWithLen(type):
    # [CLASS DOC] HELPER: Metaclass that defines the dunder length method, so classes created with
    # it have a length.
    """A metaclass that defines __len__ on the class itself"""

    def __len__(cls):
        # [FUNCTION DOC] HELPER: Metaclass length hook; receives the class and always returns
        # five.
        """Return the number of items defined in the metaclass"""
        return 5


class SimpleMetaclassClass(metaclass=MetaclassWithLen):
    # [CLASS DOC] HELPER: Empty class created with MetaclassWithLen as its metaclass, used to
    # check len on a class object.
    """A class using the MetaclassWithLen metaclass"""


class TestMetaclassLen(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Len tests where the length comes from a metaclass: len applied to a class
    # object. The class docstring also mentions classmethods, staticmethods and properties but
    # only the metaclass cases are tested.
    """Tests for len() on metaclasses, classmethods, staticmethods, and properties"""

    def setUp(self):
        # [FUNCTION DOC] HELPER: Fixture setup. Remembers the dynamo enable_trace_unittest config
        # value in an instance attribute, turns the flag on so unittest assertions can be traced
        # inside dynamo, then calls the parent setUp.
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Fixture teardown. Restores the saved enable_trace_unittest config
        # value and returns the result of the parent tearDown.
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_metaclass_len_basic(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. The builtin len on SimpleMetaclassClass returns five, supplied by
        # the metaclass.
        """Test len() on a class with __len__ defined in metaclass"""
        self.assertEqual(len(SimpleMetaclassClass), 5)

    @make_dynamo_test
    def test_metaclass_len_direct_call(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Calling the dunder length method directly on SimpleMetaclassClass
        # returns five.
        """Test direct call to __len__() on a class with metaclass-defined __len__"""
        self.assertEqual(SimpleMetaclassClass.__len__(), 5)


class CustomMutableMapping(collections.abc.MutableMapping):
    # [CLASS DOC] HELPER: Mutable mapping built on the collections.abc base, storing entries in
    # the _data dict and implementing the abstract methods by delegating to it.
    """Custom mutable mapping implementation with __len__."""

    def __init__(self, data=None):
        # [FUNCTION DOC] HELPER: Uses the given dict as _data, or a new empty dict when none
        # (None) is given.
        self._data = data if data is not None else {}

    def __len__(self):
        # [FUNCTION DOC] HELPER: Returns the number of entries in _data.
        return len(self._data)

    def __getitem__(self, key):
        # [FUNCTION DOC] HELPER: Returns the value for the key from _data; raises KeyError if
        # missing.
        return self._data[key]

    def __setitem__(self, key, value):
        # [FUNCTION DOC] HELPER: Stores the value under the key in _data.
        self._data[key] = value

    def __delitem__(self, key):
        # [FUNCTION DOC] HELPER: Removes the key from _data; raises KeyError if missing.
        del self._data[key]

    def __iter__(self):
        # [FUNCTION DOC] HELPER: Returns an iterator over the keys of _data.
        return iter(self._data)


@dataclasses.dataclass(frozen=True)
class FrozenPoint:
    # [CLASS DOC] HELPER: Frozen dataclass with three float fields x, y, z whose length is always
    # three.
    """Frozen dataclass with __len__ method."""

    x: float
    y: float
    z: float

    def __len__(self):
        # [FUNCTION DOC] HELPER: Always returns three.
        return 3


@dataclasses.dataclass(frozen=True)
class FrozenData:
    # [CLASS DOC] HELPER: Frozen dataclass with one tuple field items; its length is the length of
    # that tuple.
    """Frozen dataclass with __len__ based on items."""

    items: tuple

    def __len__(self):
        # [FUNCTION DOC] HELPER: Returns the number of elements in the items tuple.
        return len(self.items)


class TestMutableMappingLen(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Len tests for a user defined MutableMapping subclass.
    """Tests for len() on mutable mapping types."""

    def setUp(self):
        # [FUNCTION DOC] HELPER: Fixture setup. Remembers the dynamo enable_trace_unittest config
        # value in an instance attribute, turns the flag on so unittest assertions can be traced
        # inside dynamo, then calls the parent setUp.
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Fixture teardown. Restores the saved enable_trace_unittest config
        # value and returns the result of the parent tearDown.
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_len_custom_mutable_mapping(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. CustomMutableMapping with three entries has length three through
        # both call forms.
        """Test len() on custom mutable mapping."""
        m = CustomMutableMapping({"x": 10, "y": 20, "z": 30})
        self.assertEqual(len(m), 3)
        self.assertEqual(m.__len__(), 3)

    @make_dynamo_test
    def test_len_custom_mutable_mapping_empty(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. A default constructed CustomMutableMapping has length zero.
        """Test len() on empty custom mutable mapping."""
        m = CustomMutableMapping()
        self.assertEqual(len(m), 0)


class TestFrozenDataclassLen(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Len tests for frozen dataclasses that define the dunder length method.
    """Tests for len() on frozen dataclasses."""

    def setUp(self):
        # [FUNCTION DOC] HELPER: Fixture setup. Remembers the dynamo enable_trace_unittest config
        # value in an instance attribute, turns the flag on so unittest assertions can be traced
        # inside dynamo, then calls the parent setUp.
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Fixture teardown. Restores the saved enable_trace_unittest config
        # value and returns the result of the parent tearDown.
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_len_frozen_dataclass_with_len(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. FrozenPoint has length three through both call forms.
        """Test len() on frozen dataclass with custom __len__."""
        p = FrozenPoint(1.0, 2.0, 3.0)
        self.assertEqual(len(p), 3)
        self.assertEqual(p.__len__(), 3)

    @make_dynamo_test
    def test_len_frozen_dataclass_via_tuple(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. FrozenData holding a four element tuple has length four through both
        # call forms.
        """Test len() on frozen dataclass with __len__ based on contained data."""
        obj = FrozenData((1, 2, 3, 4))
        self.assertEqual(len(obj), 4)
        self.assertEqual(obj.__len__(), 4)

    @make_dynamo_test
    def test_len_frozen_dataclass_consistency(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. FrozenData holding three strings: calls len twice and checks both
        # results are three.
        """Test that len() on frozen dataclass is consistent across multiple calls."""
        obj = FrozenData(("a", "b", "c"))
        # Call len() twice to ensure consistency
        len1 = len(obj)
        len2 = len(obj)
        self.assertEqual(len1, 3)
        self.assertEqual(len2, 3)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
