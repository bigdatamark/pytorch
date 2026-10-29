# Owner(s): ["module: dynamo"]
# [MODULE DOC] Dynamo tests for tracing HuggingFace transformers ModelOutput containers and
# PretrainedConfig objects. Imports of transformers are optional: when the import fails, a module
# level marker stays unset and every HF-dependent test is skipped through a decorator. Also
# instantiates a device-generic BERT test class for each available device type. No import-time
# side effects beyond that instantiation.
import dataclasses
import unittest.mock

import torch
import torch._dynamo.testing
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.testing import same
from torch.testing._internal.common_device_type import instantiate_device_type_tests


try:
    from transformers import modeling_outputs
    from transformers.configuration_utils import PretrainedConfig
    from transformers.modeling_outputs import (
        BaseModelOutput,
        BaseModelOutputWithPastAndCrossAttentions,
        BaseModelOutputWithPoolingAndCrossAttentions,
        CausalLMOutputWithPast,
    )
    from transformers.utils import ModelOutput
except ImportError:
    modeling_outputs = None


def maybe_skip(fn):
    # [FUNCTION DOC] HELPER: Decorator for tests that need HuggingFace transformers. Returns the
    # given function wrapped in a unittest skip with the reason 'requires HuggingFace' when the
    # transformers import failed, otherwise returns the function unchanged.
    if modeling_outputs is None:
        return unittest.skip("requires HuggingFace")(fn)
    return fn


class TestHFPretrained(TestCase):
    # [CLASS DOC] Tests that Dynamo can treat a HuggingFace PretrainedConfig object as a constant
    # holder whose attributes can be read during full-graph compilation.
    @maybe_skip
    def test_pretrained(self):
        # [FUNCTION DOC] Compiles with the eager backend in full-graph mode a function that
        # branches on attribute presence and on the return_dict flag of a PretrainedConfig and
        # scales by its chunk size, then checks the result matches the uncompiled one. Skipped
        # without transformers.
        def fn(a, tmp):
            # [FUNCTION DOC] HELPER: Function under test. Adds one to the tensor if the config has
            # a 'somekey' attribute, and if the config's return_dict flag is true returns the
            # tensor plus a ones vector of length two scaled by chunk_size_feed_forward; otherwise
            # returns the tensor.
            if hasattr(tmp, "somekey"):
                a = a + 1
            if tmp.return_dict:
                return a + torch.ones(2) * tmp.chunk_size_feed_forward
            return a

        x = torch.randn(2)
        tmp = PretrainedConfig(return_dict=True, chunk_size_feed_forward=20)
        ref = fn(x, tmp)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x, tmp)
        self.assertTrue(same(ref, res))

    @maybe_skip
    def test_pretrained_non_const_attr(self):
        # [FUNCTION DOC] Checks that a default PretrainedConfig, whose attribute_map attribute is
        # a non-constant-looking mapping, can be used as a branch condition in a full-graph eager
        # compile, with the compiled result equal to eager. Skipped without transformers.
        def fn(a, tmp):
            # [FUNCTION DOC] HELPER: Returns the tensor plus one when the config's attribute_map
            # is truthy, otherwise the tensor minus one.
            if tmp.attribute_map:
                return a + 1
            else:
                return a - 1

        x = torch.randn(2)
        tmp = PretrainedConfig()
        ref = fn(x, tmp)
        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
        res = opt_fn(x, tmp)
        self.assertTrue(same(ref, res))


class TestModelOutput(TestCase):
    # [CLASS DOC] Tests Dynamo handling of HuggingFace ModelOutput dataclass-like containers:
    # construction, attribute and item assignment, attribute and item reads, tuple conversion,
    # integer indexing, custom subclasses, graph breaks, and bytecode reconstruction of outputs.
    @maybe_skip
    def test_mo_create(self):
        # [FUNCTION DOC] Verifies that creating a BaseModelOutput inside a compiled function, with
        # two tensor arguments and two tensor additions, goes through the standard Dynamo test
        # helper with an expected operation count of two. Skipped without transformers.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Builds a BaseModelOutput with the first input plus one as the
            # hidden state and the second input plus three as attentions, and returns it.
            tmp = BaseModelOutput(a + 1, attentions=b + 3)
            return tmp

        torch._dynamo.testing.standard_test(self, fn=fn, nargs=2, expected_ops=2)

    @maybe_skip
    def test_mo_assign(self):
        # [FUNCTION DOC] Builds a BaseModelOutput inside a function, then assigns a field through
        # attribute assignment and another through item assignment, and compares the three fields
        # of the eager and the optimize-assert compiled results. Expects exactly one compiled
        # frame and four captured operations. Skipped without transformers.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Creates a BaseModelOutput holding the second input plus
            # three, sets its hidden_states attribute to the first input plus seven, sets the
            # attentions item to the sum of both inputs plus six, and returns the object.
            tmp = BaseModelOutput(last_hidden_state=b + 3)
            tmp.hidden_states = a + 7
            tmp["attentions"] = a + b + 6
            return tmp

        args = [torch.randn(10), torch.randn(10)]
        obj1 = fn(*args)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(cnts)(fn)
        obj2 = opt_fn(*args)
        self.assertTrue(same(obj1.last_hidden_state, obj2.last_hidden_state))
        self.assertTrue(same(obj1.hidden_states, obj2.hidden_states))
        self.assertTrue(same(obj1.attentions, obj2.attentions))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 4)

    def _common(self, fn, op_count):
        # [FUNCTION DOC] HELPER: Shared checker for the getter tests. Builds a BaseModelOutput
        # with random length-ten last_hidden_state and attentions tensors, runs the given function
        # eagerly and under the optimize-assert compile counter backend, asserts the results
        # match, that exactly one frame was compiled, and that the operation count equals the
        # given op_count.
        args = [
            BaseModelOutput(
                last_hidden_state=torch.randn(10), attentions=torch.randn(10)
            )
        ]
        obj1 = fn(*args)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(cnts)(fn)
        obj2 = opt_fn(*args)
        self.assertTrue(same(obj1, obj2))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, op_count)

    @maybe_skip
    def test_mo_getattr(self):
        # [FUNCTION DOC] Reads fields from a BaseModelOutput by attribute inside a compiled
        # function, adding hidden_states and attentions only when they are not None, and expects
        # two captured operations through the shared checker. Skipped without transformers.
        def fn(obj: BaseModelOutput):
            # [FUNCTION DOC] HELPER: Multiplies last_hidden_state by ten and adds hidden_states
            # and attentions when each is not None, returning the resulting tensor.
            x = obj.last_hidden_state * 10
            if obj.hidden_states is not None:
                x += obj.hidden_states
            if obj.attentions is not None:
                x += obj.attentions
            return x

        self._common(fn, 2)

    @maybe_skip
    def test_mo_getattr_missing(self):
        # [FUNCTION DOC] Checks that reading a missing attribute with a default and guarding on it
        # does not break compilation; expects a single captured operation via the shared checker.
        # Skipped without transformers.
        def fn(obj: BaseModelOutput):
            # [FUNCTION DOC] HELPER: If the object has a non-None attribute named asdf it
            # increments it (never taken here), then returns attentions plus one.
            if getattr(obj, "asdf", None) is not None:
                obj.asdf += 1
            return obj.attentions + 1

        self._common(fn, 1)

    @maybe_skip
    def test_mo_getitem(self):
        # [FUNCTION DOC] Reads fields with item access and key membership tests inside a
        # compiled function, expecting two captured operations via the shared checker. Skipped
        # without transformers.
        # KNOWN ISSUE: the membership test uses the key hidden_stats (a typo for hidden_states),
        # so the hidden_states branch is never exercised.
        def fn(obj: BaseModelOutput):
            # [FUNCTION DOC] HELPER: Multiplies the last_hidden_state item by ten, then adds the
            # hidden_states item if one membership test passes and the attentions item if the
            # other passes, and returns the sum.
            x = obj["last_hidden_state"] * 10
            if "hidden_stats" in obj:
                x += obj["hidden_states"]
            if "attentions" in obj:
                x += obj["attentions"]
            return x

        self._common(fn, 2)

    @maybe_skip
    def test_mo_tuple(self):
        # [FUNCTION DOC] Checks that converting a ModelOutput to a tuple and unpacking it inside a
        # compiled function works; expects two captured operations via the shared checker. Skipped
        # without transformers.
        def fn(obj: BaseModelOutput):
            # [FUNCTION DOC] HELPER: Unpacks the tuple form of the object into two tensors and
            # returns the first plus ten times the second.
            a, b = obj.to_tuple()
            return a + b * 10

        self._common(fn, 2)

    @maybe_skip
    def test_mo_index(self):
        # [FUNCTION DOC] Checks integer indexing of a ModelOutput inside a compiled function,
        # expecting two captured operations via the shared checker. Skipped without transformers.
        def fn(obj: BaseModelOutput):
            # [FUNCTION DOC] HELPER: Returns the item at position zero times ten plus the item at
            # position one.
            return obj[0] * 10 + obj[1]

        self._common(fn, 2)

    @maybe_skip
    def test_mo_init(self):
        # [FUNCTION DOC] Defines a ModelOutput dataclass subclass with one required and four
        # optional fields, and compiles a function that introspects its dataclass fields and sums
        # the set ones. Compares against eager on a fresh instance built from three tensors,
        # expecting one frame and two operations. Skipped without transformers.
        @dataclasses.dataclass
        class MyDataClass(ModelOutput):
            # [CLASS DOC] HELPER: Local ModelOutput dataclass with a required tensor field a and
            # four optional tensor fields b, c, d and e that default to None.
            a: torch.Tensor
            b: torch.Tensor = None
            c: torch.Tensor = None
            d: torch.Tensor = None
            e: torch.Tensor = None

        def fn(obj):
            # [FUNCTION DOC] HELPER: Asserts via plain assertions that the object has dataclass
            # fields, that all fields after the first default to None, and that not all of them
            # are None on the instance. Then sums the first field with every non-None later field
            # and returns it.
            class_fields = dataclasses.fields(obj)
            assert len(class_fields)  # noqa: S101
            assert all(field.default is None for field in class_fields[1:])  # noqa: S101
            other_fields_are_none = all(
                getattr(obj, field.name) is None for field in class_fields[1:]
            )
            assert not other_fields_are_none  # noqa: S101

            total = getattr(obj, class_fields[0].name)
            for field in class_fields[1:]:
                v = getattr(obj, field.name)
                if v is not None:
                    total += v

            return total

        tensors = [torch.randn(10), torch.randn(10), torch.randn(10)]
        obj1 = MyDataClass(*tensors)
        correct1 = fn(obj1)

        obj2 = MyDataClass(*tensors)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertTrue(same(opt_fn(obj2), correct1))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    @maybe_skip
    def test_mo_init2(self):
        # [FUNCTION DOC] Checks a ModelOutput subclass with a single optional field, which takes a
        # different post-initialization code path, can be constructed in a full-graph eager
        # compile. Compares the field value to eager. Skipped without transformers.
        # this ModelOutput subclass runs a different __post_init__ codepath
        @dataclasses.dataclass
        class MyDataClass(ModelOutput):
            # [CLASS DOC] HELPER: Local ModelOutput dataclass with one optional float tensor
            # field x.
            x: torch.FloatTensor = None

        def fn(x):
            # [FUNCTION DOC] HELPER: Constructs the dataclass with the input times three and
            # returns it.
            obj = MyDataClass(x=x * 3)
            return obj

        inp = torch.randn(3, 3)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(inp).x, opt_fn(inp).x)

    @maybe_skip
    def test_mo_init_with_disable(self):
        # [FUNCTION DOC] Minimal repro for a graph break from a super call to the object attribute
        # setter: constructs a ModelOutput subclass inside a function that is marked disabled
        # non-recursively and called through a compiled wrapper. Compares the field to the eager
        # result. Skipped without transformers.
        # Can result in "non-function or method super: <slot wrapper '__setattr__' of 'object' objects>"
        # graph breaks (although it may not be the first)
        # Minimal repro for https://github.com/pytorch/pytorch/issues/126028
        @dataclasses.dataclass
        class MyDataClass(ModelOutput):
            # [CLASS DOC] HELPER: Local ModelOutput dataclass with one optional float tensor
            # field x.
            x: torch.FloatTensor = None

        @torch._dynamo.disable(recursive=False)
        def fn(x):
            # [FUNCTION DOC] HELPER: Disabled for Dynamo tracing of callees. Builds the dataclass
            # with the input as field x and returns it.
            return MyDataClass(x=x)

        inp = torch.randn(3, 3)
        opt_fn = torch.compile(fn, backend="eager")
        self.assertEqual(fn(inp).x, opt_fn(inp).x)

    @maybe_skip
    def test_mo_newkey(self):
        # [FUNCTION DOC] Creates an empty BaseModelOutput, adds a new key through item assignment
        # outside compilation, and checks that a full-graph eager compile can read that key and
        # add one. Skipped without transformers.
        obj = BaseModelOutput()

        def fn(obj):
            # [FUNCTION DOC] HELPER: Returns the item stored under the key wwww plus one.
            return obj["wwww"] + 1

        inp = torch.randn(3, 3)
        obj["wwww"] = inp
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(obj), opt_fn(obj))

    @maybe_skip
    def test_mo_from_outside(self):
        # [FUNCTION DOC] Checks that a BaseModelOutput constructed outside the compiled region and
        # passed in as an argument can have its attentions field read in a full-graph eager
        # compile. Skipped without transformers.
        def fn(obj):
            # [FUNCTION DOC] HELPER: Returns the attentions attribute of the object plus one.
            return obj.attentions + 1

        obj = BaseModelOutput(attentions=torch.randn(3, 3))
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(obj), opt_fn(obj))

    @maybe_skip
    def test_mo_reconstruct_bytecode(self):
        # [FUNCTION DOC] Checks that a BaseModelOutput created inside a compiled function and
        # returned is correctly reconstructed in the bytecode of the output, comparing attentions
        # with eager. Skipped without transformers.
        def fn(inp):
            # [FUNCTION DOC] HELPER: Returns a new BaseModelOutput whose attentions is the input
            # plus one.
            return BaseModelOutput(attentions=inp + 1)

        inp = torch.randn(3, 3)
        opt_fn = torch.compile(fn, backend="eager")
        self.assertEqual(fn(inp).attentions, opt_fn(inp).attentions)

    @maybe_skip
    def test_none(self):
        # [FUNCTION DOC] Compiles a module that returns the first element of a
        # CausalLMOutputWithPast whose loss is None, in full-graph eager mode, and compares it
        # with eager. Skipped without transformers.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward adds one to the input, wraps it in a
            # CausalLMOutputWithPast with no loss and returns its first element.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds one to the input, builds the output container with
                # loss None and logits set to the result, and returns the item at position zero.
                x = x + 1
                return CausalLMOutputWithPast(loss=None, logits=x)[0]

        model = Model()
        opt_model = torch.compile(model, backend="eager", fullgraph=True)
        x = torch.randn(1, 1, 1, 1)

        self.assertTrue(same(model(x), opt_model(x)))

    @maybe_skip
    def test_reconstruction(self):
        # [FUNCTION DOC] Exports a module that returns a CausalLMOutputWithPast with a tensor loss
        # and logits set to None using the Dynamo export with an ATen graph, and checks the
        # exported graph module result matches eager. Skipped without transformers.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward returns an output container holding the
            # input plus one as loss and no logits.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds one to the input and returns it as the loss of a
                # CausalLMOutputWithPast with logits None.
                x = x + 1
                return CausalLMOutputWithPast(loss=x, logits=None)

        model = Model()
        x = torch.randn(1, 1, 1, 1)
        eo = torch._dynamo.export(Model(), aten_graph=True)(x)
        self.assertTrue(same(model(x), eo.graph_module(x)))


class TestModelOutputBert(TestCase):
    # [CLASS DOC] Device-generic test class. Instantiated per device type at module level, so each
    # test method receives a device argument.
    @maybe_skip
    def test_HF_bert_model_output(self, device):
        # [FUNCTION DOC] Builds a small BERT-like model (encoder returning a ModelOutput, pooler
        # with a linear layer and tanh) on the given device and checks an eager-backend compile
        # gives the same last_hidden_state and pooler_output as eager. Exercises item reads, tuple
        # conversion, construction, attribute assignment and item assignment on a ModelOutput.
        # Skipped without transformers.
        class BertPooler(torch.nn.Module):
            # [CLASS DOC] HELPER: Pooling module holding a 768 by 768 linear layer on the test
            # device and a tanh activation.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the dense layer on the test device and the
                # activation module.
                super().__init__()
                self.dense = torch.nn.Linear(768, 768).to(device)
                self.activation = torch.nn.Tanh()

            def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Takes the hidden state of the first token for every
                # batch element, applies the dense layer and the activation, and returns the
                # pooled tensor.
                # We "pool" the model by simply taking the hidden state corresponding
                # to the first token.
                first_token_tensor = hidden_states[:, 0]
                pooled_output = self.dense(first_token_tensor)
                pooled_output = self.activation(pooled_output)
                return pooled_output

        class BertEncoder(torch.nn.Module):
            # [CLASS DOC] HELPER: Stub encoder with no parameters whose forward wraps its input in
            # a BaseModelOutputWithPastAndCrossAttentions.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the parent module constructor.
                super().__init__()

            def forward(
                self,
                hidden_states: torch.Tensor,
            ) -> BaseModelOutputWithPastAndCrossAttentions:
                # [FUNCTION DOC] HELPER: Returns a BaseModelOutputWithPastAndCrossAttentions whose
                # last_hidden_state is the input and whose other fields are None.
                return BaseModelOutputWithPastAndCrossAttentions(
                    last_hidden_state=hidden_states,
                    past_key_values=None,
                    hidden_states=None,
                    attentions=None,
                    cross_attentions=None,
                )

        class BertModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Model combining the stub encoder and the pooler.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the encoder and pooler submodules.
                super().__init__()
                self.encoder = BertEncoder()
                self.pooler = BertPooler()

            def forward(
                self,
                sequence_output: torch.Tensor,
            ) -> BaseModelOutputWithPoolingAndCrossAttentions:
                # [FUNCTION DOC] HELPER: Runs the encoder, reads item zero of its output, pools it
                # when a pooler exists, builds a pooling ModelOutput from the pieces, then sets
                # pooler_output both by attribute and by item assignment, and returns it.
                encoder_outputs = self.encoder(sequence_output)
                # test __getitem__ and to_tuple
                sequence_output = encoder_outputs[0]
                pooled_output = (
                    self.pooler(sequence_output) if self.pooler is not None else None
                )
                # test CustomDictVariable.create
                result = BaseModelOutputWithPoolingAndCrossAttentions(
                    last_hidden_state=sequence_output,
                    pooler_output=pooled_output,
                    past_key_values=encoder_outputs.past_key_values,
                    hidden_states=encoder_outputs.hidden_states,
                    attentions=encoder_outputs.attentions,
                    cross_attentions=encoder_outputs.cross_attentions,
                )
                # test __setattr__
                result.pooler_output = pooled_output
                # test __setitem__
                result["pooler_output"] = pooled_output
                return result

        sequence_output = torch.rand(1, 12, 768).to(device)
        model = BertModel()
        orig_result = model(sequence_output)
        compiled_model = torch.compile(model, backend="eager")
        compiled_result = compiled_model(sequence_output)
        self.assertTrue(
            torch.allclose(
                orig_result.last_hidden_state, compiled_result.last_hidden_state
            )
        )
        self.assertTrue(
            torch.allclose(orig_result.pooler_output, compiled_result.pooler_output)
        )


instantiate_device_type_tests(TestModelOutputBert, globals(), allow_xpu=True)

if __name__ == "__main__":
    run_tests()
