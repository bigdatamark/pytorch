# Owner(s): ["module: dynamo"]

# [MODULE DOC] Unit tests for MetricsContext and TopN from torch._dynamo.metrics_context. Covers
# use outside a context, nested contexts, set/update/add_to_set/set_key_value semantics and
# overwrite rules, and TopN ordering. No import-time side effects beyond the imports.
from torch._dynamo.metrics_context import MetricsContext, TopN
from torch._dynamo.test_case import run_tests, TestCase


class TestMetricsContext(TestCase):
    # [CLASS DOC] Test case exercising MetricsContext. State: self.metrics, a mapping reset to
    # empty in setUp and replaced by a copy of the accumulated metrics whenever the outermost
    # context exits.
    def setUp(self):
        # [FUNCTION DOC] Reset the per-test metrics mapping to empty after the base class setup.
        super().setUp()
        self.metrics = {}

    def _on_exit(self, start_ns, end_ns, metrics, exc_type, exc_value):
        # [FUNCTION DOC] HELPER: on_exit callback handed to MetricsContext. Ignores the timing
        # and exception arguments and stores a copy of the final metrics mapping in self.metrics
        # for later assertions.
        # Save away the metrics to be validated in the test.
        self.metrics = metrics.copy()

    def test_context_exists(self):
        # [FUNCTION DOC] Verify that increment, set and update each raise RuntimeError mentioning
        # use outside of a MetricsContext when the context has not been entered.
        """
        Setting a value without entering the context should raise.
        """
        context = MetricsContext(self._on_exit)
        with self.assertRaisesRegex(RuntimeError, "outside of a MetricsContext"):
            context.increment("m", 1)

        with self.assertRaisesRegex(RuntimeError, "outside of a MetricsContext"):
            context.set("m", 1)

        with self.assertRaisesRegex(RuntimeError, "outside of a MetricsContext"):
            context.update({"m", 1})

    def test_nested_context(self):
        # [FUNCTION DOC] Verify nesting: entering the same context twice only fires on_exit when
        # the outermost level exits (metrics still empty after the inner exit), and the final
        # metrics hold values set at both levels.
        """
        Only the outermost context should get an on_exit call, and it should
        include everything.
        """
        context = MetricsContext(self._on_exit)
        with context:
            with context:
                context.set("m1", 1)
            self.assertEqual(self.metrics, {})
            context.set("m2", 2)
        self.assertEqual(self.metrics, {"m1": 1, "m2": 2})

    def test_set(self):
        # [FUNCTION DOC] Verify that set and update (with distinct keys) accumulate into the
        # metrics mapping reported at exit.
        """
        Validate various ways to set metrics.
        """
        with MetricsContext(self._on_exit) as context:
            context.set("m1", 1)
            context.set("m2", 2)
            context.update({"m3": 3, "m4": 4})

        self.assertEqual(self.metrics, {"m1": 1, "m2": 2, "m3": 3, "m4": 4})

    def test_set_disallow_overwrite(self):
        # [FUNCTION DOC] Verify that setting an already-set metric without overwrite raises
        # RuntimeError saying it has already been set, and that the original value is kept.
        """
        Validate set won't overwrite.
        """
        with MetricsContext(self._on_exit) as context:
            context.set("m1", 1)
            with self.assertRaisesRegex(RuntimeError, "already been set"):
                context.set("m1", 2)

        self.assertEqual(self.metrics, {"m1": 1})

    def test_update_disallow_overwrite(self):
        # [FUNCTION DOC] Verify that update with a key that already exists raises RuntimeError
        # saying it has already been set when overwrite is not requested. Nothing is asserted
        # about the final metrics.
        """
        Validate update won't overwrite.
        """
        with MetricsContext(self._on_exit) as context:
            context.update({"m1": 1, "m2": 2})
            with self.assertRaisesRegex(RuntimeError, "already been set"):
                context.update({"m1": 7, "m3": 3})

    def test_update_allow_overwrite(self):
        # [FUNCTION DOC] Verify that update with overwrite enabled replaces existing values and
        # adds new keys.
        """
        Validate update will overwrite when given param.
        """
        with MetricsContext(self._on_exit) as context:
            context.update({"m1": 1, "m2": 2})
            context.update({"m1": 7, "m3": 3}, overwrite=True)

        self.assertEqual(self.metrics, {"m1": 7, "m2": 2, "m3": 3})

    def test_add_to_set(self):
        # [FUNCTION DOC] Verify add_to_set accumulates values into a set per metric name
        # (duplicates across calls merge into one set) and that the stored values are set
        # instances.
        """
        Validate add_to_set.
        """
        with MetricsContext(self._on_exit) as context:
            context.add_to_set("m1", 1)
            context.add_to_set("m1", 2)
            context.add_to_set("m2", 3)
            context.add_to_set("m2", 4)

        self.assertEqual(self.metrics, {"m1": {1, 2}, "m2": {3, 4}})
        self.assertTrue(isinstance(self.metrics["m1"], set))
        self.assertTrue(isinstance(self.metrics["m2"], set))

    def test_set_key_value(self):
        # [FUNCTION DOC] Verify set_key_value stores entries in a nested mapping under the metric
        # name, and that an existing key may be overwritten (last value wins).
        with MetricsContext(self._on_exit) as context:
            context.set_key_value("feature_usage", "k", True)
            # Overrides allowed
            context.set_key_value("feature_usage", "k2", True)
            context.set_key_value("feature_usage", "k2", False)

        self.assertEqual(self.metrics, {"feature_usage": {"k": True, "k2": False}})

    def test_top_n(self):
        # [FUNCTION DOC] Verify TopN with capacity three: after adding five key/value pairs, it
        # holds three entries and iterating yields the three largest by value in descending
        # order. Also prints the list.
        top_n = TopN(3)
        for k, v in (("seven", 7), ("four", 4), ("five", 5), ("six", 6), ("eight", 8)):
            top_n.add(k, v)

        self.assertEqual(len(top_n), 3)
        print(list(top_n))
        self.assertEqual(list(top_n), [("eight", 8), ("seven", 7), ("six", 6)])


if __name__ == "__main__":
    run_tests()
