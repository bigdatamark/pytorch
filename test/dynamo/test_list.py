# Owner(s): ["module: dynamo"]

# [MODULE DOC] Dynamo tests for tuple and list methods, operators and dunder protocol. Defines
# small helper classes for remove and sort, the TupleTests class which is parameterized by a
# sequence type, and ListTests that reuses it for list and adds list mutation tests, including
# compiled tests that mutate a module level list. Defines a module global list used by those
# tests. Running the file calls the dynamo run_tests entry point.
# TODO: move set tests from test_functions.py/test_misc.py to this file


import sys

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import make_dynamo_test


lst = []


class AlwaysEqualForListRemove:
    # [CLASS DOC] HELPER: Fixture whose equality comparison always reports true, used to check
    # list remove uses the stored item comparison.
    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Equality hook that ignores the other operand and returns true.
        return True


class NeverEqualForListRemove:
    # [CLASS DOC] HELPER: Fixture whose equality comparison always reports false.
    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Equality hook that ignores the other operand and returns false.
        return False


class CmpKeyForListSort:
    # [CLASS DOC] HELPER: Sort key wrapper holding a value v. Its less-than hook compares the
    # values modulo three, so sorting by it groups numbers by remainder.
    def __init__(self, v):
        # [FUNCTION DOC] HELPER: Stores the given value in the attribute v.
        self.v = v

    def __lt__(self, other):
        # [FUNCTION DOC] HELPER: Less-than hook. Returns true when the remainder of this value
        # modulo three is smaller than that of the other object's value.
        return self.v % 3 < other.v % 3


class TupleTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Dynamo tests for tuple methods, binary operators and dunder methods. The class
    # attribute thetype selects the sequence type (tuple here); ListTests subclasses it with list
    # so all of these tests also run for lists. Fixture turns on enable_trace_unittest.
    # Tuple methods
    # + count
    # + index
    # BinOps:
    # +, <, >, <=, >=, ==, !=
    # Dunder methods:
    # + __getitem__
    # + __contains__
    # + __delitem__

    thetype = tuple

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

    def assertEqual(self, a, b):
        # [FUNCTION DOC] Overrides the TestCase equality assertion with a simple version that
        # asserts the two operands compare equal using the equality operator; on failure the
        # message shows both values. Returns the result of the assertTrue call.
        return self.assertTrue(a == b, lambda msg: f"{msg}\n{a} != {b}")

    def assertNotEqual(self, x, y, msg=None, *, atol=None, rtol=None, **kwargs):
        # [FUNCTION DOC] Overrides the not-equal assertion with a simple version asserting the
        # operands compare unequal; the extra tolerance arguments are accepted but ignored. On
        # failure the message shows both values.
        return self.assertTrue(x != y, lambda msg: f"{msg}\n{x} == {y}")

    @make_dynamo_test
    def test_count(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Checks count on a sequence built from five characters: counts single
        # matches, returns zero for a two character item, and wrong argument counts raise
        # TypeError.
        p = self.thetype("abcab")
        self.assertEqual(p.count("a"), 2)
        self.assertEqual(p.count("ab"), 0)

        # Wrong number of arguments
        self.assertRaises(TypeError, p.count)
        self.assertRaises(TypeError, p.count, 2, 3)

    @make_dynamo_test
    def test_index(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Checks index returns the position of a present item, raises
        # ValueError for a missing item, and raises TypeError when called without an argument.
        p = self.thetype("abc")
        self.assertEqual(p.index("a"), 0)
        self.assertRaises(ValueError, p.index, "e")

        # Wrong number of arguments
        self.assertRaises(TypeError, p.index)

    @make_dynamo_test
    def test_binop_imul(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Checks the multiply dunder method returns a new object of the same
        # type with the contents repeated twice, leaves the original unchanged, and raises
        # TypeError for a missing argument or a float factor. Note the test name says in-place
        # multiply but this version tests plain multiplication; ListTests overrides it.
        p = self.thetype([1, 2, 3])
        r = p.__mul__(2)
        self.assertIsInstance(r, self.thetype)
        self.assertEqual(r, self.thetype([1, 2, 3, 1, 2, 3]))
        self.assertEqual(p, self.thetype([1, 2, 3]))

        # Wrong number of arguments
        self.assertRaises(TypeError, p.__mul__)

        # can only multiply list by an integer
        self.assertRaises(TypeError, p.__mul__, 2.2)

    @make_dynamo_test
    def test_binop_add(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Checks the add operator and the add dunder method concatenate two
        # sequences into the same type, and that a missing argument or a dict operand raises
        # TypeError.
        p, q = map(self.thetype, ["abc", "bcd"])
        self.assertIsInstance(p + q, self.thetype)
        self.assertEqual(p + q, self.thetype("abcbcd"))
        self.assertEqual(p.__add__(q), self.thetype("abcbcd"))

        # Wrong number of arguments
        self.assertRaises(TypeError, p.__add__)

        # can only concatenate items of the same type
        self.assertRaises(TypeError, p.__add__, dict.fromkeys(q))

    @make_dynamo_test
    def test_cmp_eq(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Checks equality: the operator and the eq dunder method treat equal
        # contents as equal, different contents as not equal, and a missing argument raises
        # TypeError.
        p, q, r = map(self.thetype, ["ab", "abc", "ab"])
        self.assertTrue(p == p)
        self.assertTrue(p == r)
        self.assertEqual(p, p)
        self.assertEqual(p, r)
        self.assertNotEqual(p, q)
        self.assertTrue(p.__eq__(r))

        # Wrong number of arguments
        self.assertRaises(TypeError, p.__eq__)

    @make_dynamo_test
    def test_cmp_ne(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Checks inequality through the operator and the ne dunder method on
        # sequences of different content, and TypeError for a missing argument.
        p, q = map(self.thetype, ["ab", "abc"])
        self.assertTrue(p != q)
        self.assertNotEqual(p, q)
        self.assertTrue(p.__ne__(q))

        # Wrong number of arguments
        self.assertRaises(TypeError, p.__ne__)

    @make_dynamo_test
    def test_cmp_less_than(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Checks lexicographic less-than: a shorter prefix is less than the
        # longer sequence through the operator and the lt dunder method, the reverse is false, and
        # a missing argument raises TypeError.
        p, q = map(self.thetype, ["ab", "abc"])
        self.assertTrue(p < q)
        self.assertTrue(p.__lt__(q))
        self.assertFalse(q < p)

        # Wrong number of arguments
        self.assertRaises(TypeError, p.__lt__)

    @make_dynamo_test
    def test_cmp_greater_than(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Checks greater-than through the operator and the gt dunder method
        # for a longer sequence against its prefix, the reverse is false, and a missing argument
        # raises TypeError.
        p, q = map(self.thetype, ["ab", "abc"])
        self.assertTrue(q > p)
        self.assertTrue(q.__gt__(p))
        self.assertFalse(p > q)

        # Wrong number of arguments
        self.assertRaises(TypeError, p.__gt__)

    @make_dynamo_test
    def test_cmp_less_than_or_equal(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Checks less-or-equal through the operator and the le dunder method
        # for a prefix against the longer sequence, the reverse is false, and a missing argument
        # raises TypeError.
        p, q = map(self.thetype, ["ab", "abc"])
        self.assertTrue(p <= q)
        self.assertTrue(p.__le__(q))
        self.assertFalse(q <= p)

        # Wrong number of arguments
        self.assertRaises(TypeError, p.__le__)

    @make_dynamo_test
    def test_cmp_greater_than_or_equal(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Checks greater-or-equal through the operator and the ge dunder
        # method, the reverse is false, and a missing argument raises TypeError.
        p, q = map(self.thetype, ["ab", "abc"])
        self.assertTrue(q >= p)
        self.assertTrue(q.__ge__(p))
        self.assertFalse(p >= q)

        # Wrong number of arguments
        self.assertRaises(TypeError, p.__ge__)

    @make_dynamo_test
    def test___getitem__(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Checks the getitem dunder method returns the element at an index,
        # raises IndexError for an out of range index, and raises TypeError for zero or two
        # arguments.
        p = self.thetype("abc")
        self.assertEqual(p.__getitem__(2), "c")
        self.assertRaises(IndexError, p.__getitem__, 10)

        # Wrong number of arguments
        self.assertRaises(TypeError, p.__getitem__)
        self.assertRaises(TypeError, p.__getitem__, 1, 2)

    @make_dynamo_test
    def test___contains__(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Checks the contains dunder method reports membership, returns a real
        # bool, and raises TypeError for zero or two arguments.
        p = self.thetype("abc")
        self.assertTrue(p.__contains__("a"))
        self.assertIsInstance(p.__contains__("c"), bool)

        # Wrong number of arguments
        self.assertRaises(TypeError, p.__contains__)
        self.assertRaises(TypeError, p.__contains__, 1, 2)

    @make_dynamo_test
    def test___iter__(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Checks the iter dunder method yields an iterator whose first item is
        # the only element, and that taking the iterator of an iterator again still works.
        p = self.thetype([1])
        it = p.__iter__()
        self.assertEqual(next(it), 1)
        it = p.__iter__().__iter__()
        self.assertEqual(next(it), 1)

    @make_dynamo_test
    def test_list_mul_constant_tuple(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Multiplies a list holding one constant tuple by two and checks the
        # result is a list with two references to that tuple.
        tree = (1, 2)
        result = [tree] * 2
        self.assertEqual(result, [tree, tree])


class ListTests(TupleTests):
    # [CLASS DOC] Dynamo tests for list specific methods (append, copy, clear, extend, insert,
    # pop, remove, reverse, sort, in-place operators, setitem and delitem). Sets thetype to list
    # and inherits every TupleTests test, so the shared tests also run on lists.
    # List methods
    # + append
    # + copy
    # + clear
    # + extend
    # + insert
    # + pop
    # + remove
    # + reverse
    # + sort
    # BinOps:
    # +, <, >, <=, >=, ==, !=
    # Dunder methods:
    # + __setitem__
    # + __getitem__
    # + __contains__
    # + __delitem__

    thetype = list

    @make_dynamo_test
    def test_append(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Checks append returns None, adds the item at the end, and raises
        # TypeError for zero or two arguments.
        p = self.thetype("abc")
        self.assertIsNone(p.append("d"))
        self.assertEqual(p, ["a", "b", "c", "d"])

        # Wrong number of arguments
        self.assertRaises(TypeError, p.append)
        self.assertRaises(TypeError, p.append, 2, 3)

    @make_dynamo_test
    def test_copy(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Checks copy returns a list equal to the original and that an extra
        # argument raises TypeError.
        p = self.thetype("abc")
        self.assertEqual(p.copy(), p)

        # Wrong number of arguments
        self.assertRaises(TypeError, p.copy, 1)

    @make_dynamo_test
    def test_clear(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Checks clear returns None and empties the list so its length is
        # zero, and that an extra argument raises TypeError.
        p = self.thetype("abc")
        self.assertIsNone(p.clear())
        self.assertEqual(p, [])
        self.assertEqual(len(p), 0)

        # Wrong number of arguments
        self.assertRaises(TypeError, p.clear, 1)

    @make_dynamo_test
    def test_extend(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Checks extend returns None and appends all items of another list,
        # raises TypeError for a non-iterable and for wrong argument counts.
        p, q = map(self.thetype, ["ab", "cd"])
        self.assertIsNone(p.extend(q))
        self.assertEqual(p, self.thetype("abcd"))

        # extend needs an iterable
        self.assertRaises(TypeError, p.extend, 1)

        # Wrong number of arguments
        self.assertRaises(TypeError, p.extend)
        self.assertRaises(TypeError, p.extend, 2, 3)

    @make_dynamo_test
    def test_insert(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Checks insert returns None and puts the item at the given index, and
        # raises TypeError with zero, one or three arguments.
        p = self.thetype("abc")
        self.assertIsNone(p.insert(1, "ef"))
        self.assertEqual(p, ["a", "ef", "b", "c"])

        # Wrong number of arguments
        self.assertRaises(TypeError, p.insert)
        self.assertRaises(TypeError, p.insert, 1)
        self.assertRaises(TypeError, p.insert, 1, 2, 3)

    @make_dynamo_test
    def test_pop(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Checks pop with no argument removes the last item, pop with an index
        # removes that item, an out of range index raises IndexError, and two arguments raise
        # TypeError.
        p = self.thetype("abcd")
        self.assertEqual(p.pop(), "d")
        self.assertEqual(p.pop(1), "b")
        # The length of p is now 2, valid indices are 0, 1
        self.assertRaises(IndexError, p.pop, 2)

        # Wrong number of arguments
        self.assertRaises(TypeError, p.pop, 2, 3)

    @make_dynamo_test
    def test_remove(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Checks remove returns None and deletes the first matching item,
        # raises ValueError when not found, and TypeError for zero or two arguments.
        p = self.thetype("abad")
        self.assertIsNone(p.remove("a"))
        self.assertEqual(p, ["b", "a", "d"])
        self.assertRaises(ValueError, p.remove, "x")

        # Wrong number of arguments
        self.assertRaises(TypeError, p.remove)
        self.assertRaises(TypeError, p.remove, 2, 3)

    @make_dynamo_test
    def test_remove_uses_item_richcompare(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Checks remove uses the stored item's equality: a list holding an
        # always-equal object has it removed when given a never-equal object.
        p = [AlwaysEqualForListRemove()]
        self.assertIsNone(p.remove(NeverEqualForListRemove()))
        self.assertEqual(p, [])

    @make_dynamo_test
    def test_remove_matches_identity_before_richcompare(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Checks remove matches on identity before equality: the same never-
        # equal object that is stored is still removed.
        item = NeverEqualForListRemove()
        p = [item]
        self.assertIsNone(p.remove(item))
        self.assertEqual(p, [])

    @make_dynamo_test
    def test_reverse(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Checks reverse returns None and reverses the list in place, and that
        # an extra argument raises TypeError.
        p = self.thetype("abcd")
        self.assertIsNone(p.reverse())
        self.assertEqual(p, self.thetype("dcba"))

        # Wrong number of arguments
        self.assertRaises(TypeError, p.reverse, 1)

    @make_dynamo_test
    def test_sort(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Checks sort returns None and orders a list of characters ascending
        # in place.
        p = self.thetype("dbca")
        self.assertIsNone(p.sort())
        self.assertEqual(p, self.thetype("abcd"))

    @make_dynamo_test
    def test_sort_cmp_to_key(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Checks sort with a key built by cmp_to_key from a reversed
        # comparison function, in descending then (with reverse true) ascending order.
        from functools import cmp_to_key

        def revcmp(a, b):
            # [FUNCTION DOC] HELPER: Three way comparison for reversed order; returns a positive,
            # zero or negative integer from comparing the arguments, negated relative to normal
            # ordering.
            return (a < b) - (a > b)

        p = self.thetype([1, 3, 2])
        p.sort(key=cmp_to_key(revcmp))
        self.assertEqual(p, self.thetype([3, 2, 1]))
        p.sort(key=cmp_to_key(revcmp), reverse=True)
        self.assertEqual(p, self.thetype([1, 2, 3]))

    @make_dynamo_test
    def test_sort_user_defined_lt(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Sorts with the CmpKeyForListSort key, which uses a user defined
        # less-than hook comparing remainders modulo three, and checks both normal and reversed
        # order results.
        p = self.thetype([3, 4, 5])
        p.sort(key=CmpKeyForListSort)
        self.assertEqual(p, self.thetype([3, 4, 5]))
        p.sort(key=CmpKeyForListSort, reverse=True)
        self.assertEqual(p, self.thetype([5, 4, 3]))

    @make_dynamo_test
    def test_sort_modified_during_sort(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Sorts with a comparison function that appends to the list while
        # sorting; checks that ValueError is raised and that the appended items are discarded,
        # leaving the sorted original items.
        from functools import cmp_to_key

        p = self.thetype([1, 3, 2])

        def selfmodifying(a, b):
            # [FUNCTION DOC] HELPER: Comparison function that appends the value four to the
            # captured list p as a side effect, then returns a three way comparison result of its
            # arguments.
            p.append(4)
            return (a > b) - (a < b)

        self.assertRaises(ValueError, p.sort, key=cmp_to_key(selfmodifying))
        # items appended during the sort are discarded
        self.assertEqual(p, self.thetype([1, 2, 3]))

    @make_dynamo_test
    def test_sort_empty_during_sort(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Sorts with a key function that records the list length each time it
        # is called; checks the list sorts correctly and every recorded size is zero, showing that
        # the list appears empty during sorting.
        p = self.thetype([1, 3, 2])
        sizes = []

        def key(x):
            # [FUNCTION DOC] HELPER: Key function. Appends the current length of the captured list
            # p to the captured list sizes and returns its argument unchanged.
            sizes.append(len(p))
            return x

        p.sort(key=key)
        self.assertEqual(p, self.thetype([1, 2, 3]))
        self.assertEqual(sizes, [0, 0, 0])

    @make_dynamo_test
    def test_binop_imul(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Checks in-place multiply: the imul dunder method and the augmented
        # operator extend the same list by repetition, return a list, and a missing argument or
        # float factor raises TypeError.
        p = self.thetype([1, 2, 3])
        r = p.__imul__(2)
        self.assertIsInstance(r, self.thetype)
        self.assertEqual(r, self.thetype([1, 2, 3, 1, 2, 3]))
        self.assertEqual(p, self.thetype([1, 2, 3, 1, 2, 3]))

        p = self.thetype("ab")
        p *= 2
        self.assertEqual(p, self.thetype("abab"))

        # Wrong number of arguments
        self.assertRaises(TypeError, p.__imul__)

        # can only multiply list by an integer
        self.assertRaises(TypeError, p.__imul__, 2.2)

    def test_binop_imul_global_list(self):
        # [FUNCTION DOC] Not decorated with make_dynamo_test. Assigns a module global list, then
        # compiles with the eager backend and full graph a function that applies in-place multiply
        # by two and by three to the global and returns the sine of its input. Checks the output
        # matches and the global ends up repeated six times.
        global lst
        lst = self.thetype(["a", "b"])

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function. Multiplies the module global list in place
            # by two via the operator and by three via the imul dunder method, then returns the
            # sine of the input tensor.
            global lst
            lst *= 2
            lst.__imul__(3)
            return x.sin()

        x = torch.tensor(1.0)
        self.assertEqual(fn(x), x.sin())
        self.assertEqual(lst, ["a", "b"] * 6)

    @make_dynamo_test
    def test_binop_iadd(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Checks in-place add: the iadd dunder method returns the same
        # extended list, the augmented operator accepts a string and extends with its characters,
        # and a missing argument raises TypeError. KNOWN ISSUE: the last check calls the add
        # dunder method rather than the in-place one, so the comment about concatenating the same
        # type tests plain add.
        p, q = map(self.thetype, ["abc", "bcd"])
        r = p.__iadd__(q)
        self.assertIsInstance(r, self.thetype)
        self.assertEqual(r, self.thetype("abcbcd"))
        self.assertEqual(p, self.thetype("abcbcd"))

        p = self.thetype("ab")
        p += "cd"
        self.assertEqual(p, self.thetype("abcd"))

        # Wrong number of arguments
        self.assertRaises(TypeError, p.__iadd__)

        # can only concatenate items of the same type
        self.assertRaises(TypeError, p.__add__, dict.fromkeys(q))

    def test_binop_iadd_global_list(self):
        # [FUNCTION DOC] Not decorated with make_dynamo_test. Sets a module global to an empty
        # list, compiles with the eager backend and full graph a function that extends it in place
        # twice, and checks the output and that the global equals the list a, b.
        global lst
        lst = self.thetype([])

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function. Applies in-place add of a one item list to
            # the module global via the operator, then via the iadd dunder method, and returns the
            # sine of the input tensor.
            global lst
            lst += ["a"]
            lst.__iadd__(["b"])
            return x.sin()

        x = torch.tensor(1.0)
        self.assertEqual(fn(x), x.sin())
        self.assertEqual(lst, ["a", "b"])

    def test_binop_delitem_global_list(self):
        # [FUNCTION DOC] Not decorated with make_dynamo_test. Sets a module global to a three item
        # list, compiles with the eager backend and full graph a function that deletes the item at
        # index one from the global, and checks the output and that the global is left with the
        # first and last items.
        global lst
        lst = self.thetype(["a", "b", "c"])

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function. Deletes the element at index one of the
            # module global list and returns the sine of the input tensor.
            global lst
            del lst[1]
            return x.sin()

        x = torch.tensor(1.0)
        self.assertEqual(fn(x), x.sin())
        self.assertEqual(lst, ["a", "c"])

    @make_dynamo_test
    def test___setitem__(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Checks the setitem dunder method returns None and replaces an
        # element, slice assignment of an empty list clears the list, and wrong argument counts
        # raise TypeError.
        p = self.thetype("abc")
        self.assertIsNone(p.__setitem__(2, "a"))
        self.assertEqual(p, self.thetype("aba"))

        p[0:] = []
        self.assertEqual(p, [])

        # Wrong number of arguments
        self.assertRaises(TypeError, p.__setitem__)
        self.assertRaises(TypeError, p.__setitem__, 1)
        self.assertRaises(TypeError, p.__setitem__, 1, 2, 3)

    @make_dynamo_test
    def test___delitem__(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Checks the delitem dunder method returns None for an integer index
        # and for a slice, raises ValueError for a slice step of zero, and raises TypeError for a
        # missing argument, a float index or two arguments.
        p = self.thetype("abcdef")
        self.assertIsNone(p.__delitem__(1))
        self.assertEqual(p, self.thetype("acdef"))

        self.assertIsNone(p.__delitem__(slice(1, 3)))
        self.assertEqual(p, self.thetype("aef"))

        # Slice step == 0
        self.assertRaises(ValueError, p.__delitem__, slice(1, 1, 0))

        # Wrong number of arguments
        self.assertRaises(TypeError, p.__delitem__)
        self.assertRaises(TypeError, p.__delitem__, 1.1)
        self.assertRaises(TypeError, p.__delitem__, 1, 2)

    @make_dynamo_test
    def test___setitem___slice_non_iterable(self):
        # [FUNCTION DOC] Test decorated with make_dynamo_test, so it runs once eagerly and once
        # compiled by dynamo. Checks the error message when assigning a non-iterable to a slice.
        # Computes from sys.version_info whether this interpreter has the old CPython behavior
        # (before 3.10.20, 3.11.15 and 3.12.5) which says that an iterable is needed; otherwise
        # (and always for extended slices with a step) the message says an iterable must be
        # assigned to an extended slice. Finally checks a valid slice assignment works.
        # ref: https://github.com/python/cpython/pull/120442 (gh-120384),
        # which fixed an array-out-of-bounds crash by moving the
        # PySequence_Fast check ahead of the step==1 branch in
        # list_ass_subscript. Before this landed, simple (step is None or
        # 1) slices raised "can only assign an iterable" instead of the
        # unified extended-slice message. Verified empirically against
        # real CPython builds (cross-checked via two independent
        # toolchains): the fix shipped in 3.10.20, 3.11.15, and 3.12.5
        # (3.12.0 is the first 3.12 release, so 3.12.0-3.12.4 predate it);
        # 3.13+ never had the bug.
        v = sys.version_info
        has_bug = (
            (v[:2] == (3, 10) and v < (3, 10, 20))
            or (v[:2] == (3, 11) and v < (3, 11, 15))
            or (v[:2] == (3, 12) and v < (3, 12, 5))
        )
        p = self.thetype("abcdef")
        if has_bug:
            self.assertRaisesRegex(
                TypeError,
                "can only assign an iterable",
                p.__setitem__,
                slice(1, 3),
                1,
            )
        else:
            self.assertRaisesRegex(
                TypeError,
                "must assign iterable to extended slice",
                p.__setitem__,
                slice(1, 3),
                1,
            )

        # Extended slices (step != 1) always use the extended-slice message.
        self.assertRaisesRegex(
            TypeError,
            "must assign iterable to extended slice",
            p.__setitem__,
            slice(1, 5, 2),
            1,
        )

        # Valid iterable assignments are unaffected.
        p[1:3] = ["x", "y"]
        self.assertEqual(p, ["a", "x", "y", "d", "e", "f"])


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
