# mypy: allow-untyped-defs
# [MODULE DOC] Prototype machinery that records ("captures") operations performed on a
# placeholder DataFrame object so they can be replayed on each real DataFrame flowing
# through a datapipe. Contains the Capture expression-node classes (attribute, item, call,
# arithmetic, variable assignment), the global CaptureControl switch, the traced datapipe
# DataFrameTracedOps, and DataFrameTracer registered as the functional datapipe
# trace_as_dataframe. Import registers that functional name; no other side effects.
from typing import Any, NoReturn

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.dataframe.structures import DataChunkDF
from torch.utils.data.datapipes.datapipe import DFIterDataPipe, IterDataPipe


# TODO(VitalyFedyunin): Add error when two different traces get combined

__all__ = [
    "Capture",
    "CaptureA",
    "CaptureAdd",
    "CaptureCall",
    "CaptureControl",
    "CaptureDataFrame",
    "CaptureDataFrameWithDataPipeOps",
    "CaptureF",
    "CaptureGetAttr",
    "CaptureGetItem",
    "CaptureInitial",
    "CaptureLikeMock",
    "CaptureMul",
    "CaptureSetItem",
    "CaptureSub",
    "CaptureVariable",
    "CaptureVariableAssign",
    "DataFrameTracer",
    "DataFrameTracedOps",
    "disable_capture",
    "get_val",
]


def disable_capture() -> None:
    # [FUNCTION DOC] Globally turns capture off by setting the CaptureControl.disabled
    # flag; afterwards creating a CaptureVariable raises RuntimeError. Process-wide, with
    # no way to re-enable in this module.
    CaptureControl.disabled = True


class CaptureControl:
    # [CLASS DOC] HELPER: Holder of the global class-level boolean flag disabled, read by
    # CaptureVariable creation and set by disable_capture. Never instantiated.
    disabled = False


class DataFrameTracedOps(DFIterDataPipe):
    # [CLASS DOC] Datapipe that replays recorded operations on every upstream DataFrame.
    # State: source_datapipe and output_var (the captured variable whose context holds the
    # operation list).
    def __init__(self, source_datapipe, output_var) -> None:
        # [FUNCTION DOC] Stores the upstream datapipe and the output capture variable.
        super().__init__()
        self.source_datapipe = source_datapipe
        self.output_var = output_var

    def __iter__(self):
        # [FUNCTION DOC] For each upstream item, yields the result of applying all recorded
        # operations through output_var.apply_ops.
        for item in self.source_datapipe:
            yield self.output_var.apply_ops(item)


#  TODO(VitalyFedyunin): Extract this list from the DFIterDataPipe registered functions
DATAPIPES_OPS = [
    "_dataframes_as_tuples",
    "groupby",
    "_dataframes_filter",
    "map",
    "to_datapipe",
    "shuffle",
    "concat",
    "batch",
    "_dataframes_per_row",
    "_dataframes_concat",
    "_dataframes_shuffle",
]

UNIMPLEMENTED_ATTR = ["__deepcopy__", "__setstate__", "is_shardable", "apply_sharding"]


class Capture:
    # [CLASS DOC] Base node of the recorded-operation graph. State: ctx, a mapping with a
    # shared list of operations, a shared list of variables and an optional schema_df. Nodes
    # created from one another share the same ctx object. Operator overloads (attribute,
    # item get/set, add, sub, mul, call) do not compute; they append nodes to ctx and return
    # new capture objects. Subclasses provide execute to compute a value.
    # TODO: All operations are shared across entire InitialCapture, need to figure out what if we join two captures

    def __init__(self, schema_df=None) -> None:
        # [FUNCTION DOC] Creates a fresh context with empty operation and variable lists
        # and the given schema DataFrame (used to infer columns).
        self.ctx = {"operations": [], "variables": [], "schema_df": schema_df}

    def __str__(self) -> str:
        # [FUNCTION DOC] Returns the textual listing of all recorded operations.
        return self._ops_str()

    def _ops_str(self):
        # [FUNCTION DOC] HELPER: Joins the string forms of all recorded operations with
        # newlines.
        res = ""
        for op in self.ctx["operations"]:
            if len(res) > 0:
                res += "\n"
            res += str(op)
        return res

    def __getstate__(self):
        # [FUNCTION DOC] Pickle support. Side effect: clears the schema DataFrame in the
        # shared context and the cached calculated value of every variable (mutating self),
        # then returns a mapping copy of the instance attributes.
        # TODO(VitalyFedyunin): Currently can't pickle (why?)
        self.ctx["schema_df"] = None
        for var in self.ctx["variables"]:
            var.calculated_value = None
        state = {}
        for item in self.__dict__:
            state[item] = getattr(self, item)
        return state

    def __setstate__(self, state):
        # [FUNCTION DOC] Unpickle support: sets every saved attribute on the instance.
        for k, v in state.items():
            setattr(self, k, v)

    def __getattr__(self, attrname):
        # [FUNCTION DOC] Called for attributes not found normally. Raises RuntimeError for
        # kwarg/kwargs and AttributeError for __deepcopy__; otherwise returns a
        # CaptureGetAttr node recording the attribute access.
        if attrname == "kwarg" or attrname == "kwargs":
            raise RuntimeError("no kwargs!")
        if attrname == "__deepcopy__":
            raise AttributeError
        result = CaptureGetAttr(self, attrname, ctx=self.ctx)
        return result

    def __getitem__(self, key):
        # [FUNCTION DOC] Returns a CaptureGetItem node recording item access with key.
        return CaptureGetItem(self, key, ctx=self.ctx)

    def __setitem__(self, key, value) -> None:
        # [FUNCTION DOC] Appends a CaptureSetItem operation (assignment of value at key)
        # to the shared operation list. Nothing is evaluated.
        self.ctx["operations"].append(CaptureSetItem(self, key, value, ctx=self.ctx))

    def __add__(self, add_val):
        # [FUNCTION DOC] Records an addition: appends a variable assignment of the sum node
        # to the operations and returns the new CaptureVariable holding the result.
        res = CaptureAdd(self, add_val, ctx=self.ctx)
        var = CaptureVariable(res, ctx=self.ctx)
        self.ctx["operations"].append(
            CaptureVariableAssign(variable=var, value=res, ctx=self.ctx)
        )
        return var

    def __sub__(self, add_val):
        # [FUNCTION DOC] Records a subtraction like __add__; returns the result variable.
        res = CaptureSub(self, add_val, ctx=self.ctx)
        var = CaptureVariable(res, ctx=self.ctx)
        self.ctx["operations"].append(
            CaptureVariableAssign(variable=var, value=res, ctx=self.ctx)
        )
        return var

    def __mul__(self, add_val):
        # [FUNCTION DOC] Records a multiplication like __add__; returns the result variable.
        res = CaptureMul(self, add_val, ctx=self.ctx)
        var = CaptureVariable(res, ctx=self.ctx)
        t = CaptureVariableAssign(variable=var, value=res, ctx=self.ctx)
        self.ctx["operations"].append(t)
        return var

    def _is_context_empty(self):
        # [FUNCTION DOC] HELPER: True when the shared context has no operations and no
        # variables.
        return len(self.ctx["operations"]) == 0 and len(self.ctx["variables"]) == 0

    def apply_ops_2(self, dataframe) -> None:
        # [FUNCTION DOC] Seeds the first variable of the context with dataframe, then
        # executes every recorded operation in order. Mutates variables' cached values; not
        # thread safe. Returns nothing.
        # TODO(VitalyFedyunin): Make this calculation thread safe (as currently it updates pointer)
        self.ctx["variables"][0].calculated_value = dataframe
        for op in self.ctx["operations"]:
            op.execute()

    @property
    def columns(self):
        # [FUNCTION DOC] Property: replays the operations on the schema DataFrame and
        # returns the columns of the resulting value, to expose the output schema.
        # KNOWN ISSUE: the base class defines no execute method, so self.execute resolves
        # through __getattr__ to a recorded attribute node rather than a real computation
        # unless a subclass defines execute.
        self.apply_ops_2(self.ctx["schema_df"])
        value = self.execute()
        return value.columns

    # TODO(VitalyFedyunin): Add tests
    # TODO(VitalyFedyunin): Need to join context if one of them are empty because we used capture

    def __call__(self, *args, **kwargs):
        # [FUNCTION DOC] Records a call: if this node's context is empty, adopts the context
        # of the first non-empty Capture found in args or kwargs. Appends an assignment of a
        # CaptureCall node to a new variable and returns that variable.
        # KNOWN ISSUE: the kwargs loop tests whether the key (not only the value) is a
        # Capture, which is likely a typo since keyword names are strings.
        # TODO: Check if args or kwargs have more than one different context
        if self._is_context_empty():
            # TODO: Allow CaptureA to take context from mock
            for arg in args:
                if isinstance(arg, Capture) and not arg._is_context_empty():
                    self.ctx = arg.ctx
                    break
            if self._is_context_empty():
                for k, v in kwargs.items():
                    if isinstance(k, Capture) and not k._is_context_empty():
                        self.ctx = k.ctx
                        break
                    if isinstance(v, Capture) and not v._is_context_empty():
                        self.ctx = v.ctx
                        break

        res = CaptureCall(self, ctx=self.ctx, args=args, kwargs=kwargs)
        var = CaptureVariable(None, ctx=self.ctx)
        t = CaptureVariableAssign(ctx=self.ctx, variable=var, value=res)
        self.ctx["operations"].append(t)
        return var


class CaptureF(Capture):
    # [CLASS DOC] Capture node carrying keyword data in kwargs, with an optional shared
    # context (a new empty one is created if none is given).
    def __init__(self, ctx=None, **kwargs) -> None:
        # [FUNCTION DOC] Uses ctx if given, else a fresh context; stores kwargs.
        super().__init__()
        if ctx is None:
            self.ctx = {"operations": [], "variables": []}
        else:
            self.ctx = ctx
        self.kwargs = kwargs


class CaptureA(CaptureF):
    # [CLASS DOC] Capture standing for a patched-in real attribute; kwargs hold name and
    # real_attribute.
    def __str__(self) -> str:
        # [FUNCTION DOC] Returns the recorded name.
        return f"{self.kwargs['name']}"

    def execute(self):
        # [FUNCTION DOC] Returns the stored real attribute (the original, unpatched object).
        value = self.kwargs["real_attribute"]
        return value


class CaptureLikeMock:
    # [CLASS DOC] Context manager that temporarily replaces an importable attribute (given
    # by dotted name) with a CaptureA node, so calls to it get recorded; restores the
    # original on exit. State: get_target, attribute, name, save.
    def __init__(self, name) -> None:
        # [FUNCTION DOC] Resolves the dotted name into a target getter and attribute name
        # using the private mock helper, and stores them. Nothing is patched yet.
        import unittest.mock as mock

        # TODO(VitalyFedyunin): Do not use private function here, copy own implementation instead.
        get_target, attribute = mock._get_target(name)  # type: ignore[attr-defined]
        self.get_target = get_target
        self.attribute = attribute
        self.name = name

    def __enter__(self):
        # [FUNCTION DOC] Saves the original attribute and replaces it with a CaptureA node
        # wrapping it. Side effect: mutates the target module or class. Returns None.
        self.save = getattr(self.get_target(), self.attribute)
        capt = CaptureA(name=self.name, real_attribute=self.save)
        setattr(self.get_target(), self.attribute, capt)

    def __exit__(self, *exc_info):
        # [FUNCTION DOC] Restores the saved original attribute; exceptions are not
        # suppressed.
        setattr(self.get_target(), self.attribute, self.save)


class CaptureCall(Capture):
    # [CLASS DOC] Node for a recorded call of a captured callable. State: callable and
    # kwargs holding the call args and kwargs.
    def __init__(self, callable, ctx=None, **kwargs) -> None:
        # [FUNCTION DOC] Uses ctx if given else a fresh context; stores callable and kwargs.
        super().__init__()
        if ctx is None:
            self.ctx = {"operations": [], "variables": []}
        else:
            self.ctx = ctx
        self.kwargs = kwargs
        self.callable = callable

    def __str__(self) -> str:
        # [FUNCTION DOC] Returns text of the form callable(args,kwargs).
        return "{callable}({args},{kwargs})".format(
            callable=self.callable, **self.kwargs
        )

    def execute(self):
        # [FUNCTION DOC] Evaluates the callable and calls it with positional args, where
        # Capture args are executed first. Keyword args are passed through unevaluated.
        # TODO: VitalyFedyunin execute kwargs and maybe nested structures
        executed_args = []
        for arg in self.kwargs["args"]:
            if isinstance(arg, Capture):
                executed_args.append(arg.execute())
            else:
                executed_args.append(arg)
        left = get_val(self.callable)
        return left(*executed_args, **self.kwargs["kwargs"])


class CaptureVariableAssign(CaptureF):
    # [CLASS DOC] Operation node assigning the evaluated value node to a variable; kwargs
    # hold variable and value.
    def __str__(self) -> str:
        # [FUNCTION DOC] Returns text of the form variable = value.
        variable = self.kwargs["variable"]
        value = self.kwargs["value"]
        return f"{variable} = {value}"

    def execute(self) -> None:
        # [FUNCTION DOC] Evaluates the value node and caches the result on the variable.
        self.kwargs["variable"].calculated_value = self.kwargs["value"].execute()


class CaptureVariable(Capture):
    # [CLASS DOC] Named variable node registered in the shared context. State: value,
    # unique name (var_N from a class-wide counter), and calculated_value set when
    # operations run. Creation raises RuntimeError when capture is disabled.
    # TODO(VitalyFedyunin): This should be atomic and thread safe
    names_idx = 0

    def __init__(self, value, ctx) -> None:
        # [FUNCTION DOC] Registers the variable in ctx using the next global counter value
        # for its name. Raises RuntimeError if capture is disabled. Side effects: increments
        # the class counter (not thread safe) and appends to the context variables.
        super().__init__()
        if CaptureControl.disabled:
            raise RuntimeError("Attempting to create capture variable with capture off")
        self.ctx = ctx
        self.value = value
        self.name = f"var_{CaptureVariable.names_idx}"
        CaptureVariable.names_idx += 1
        self.ctx["variables"].append(self)

    def __str__(self) -> str:
        # [FUNCTION DOC] Returns the variable name.
        return self.name

    def execute(self):
        # [FUNCTION DOC] Returns the cached calculated value.
        return self.calculated_value

    def apply_ops(self, dataframe):
        # [FUNCTION DOC] Seeds the first context variable with dataframe, executes all
        # recorded operations in order, and returns this variable's calculated value. Not
        # thread safe since it mutates cached values.
        # TODO(VitalyFedyunin): Make this calculation thread safe (as currently it updates pointer)
        self.ctx["variables"][0].calculated_value = dataframe
        for op in self.ctx["operations"]:
            op.execute()
        return self.calculated_value


class CaptureGetItem(Capture):
    # [CLASS DOC] Node for a recorded item read from left at key.
    def __init__(self, left, key, ctx) -> None:
        # [FUNCTION DOC] Stores the shared ctx, the left node and the key.
        super().__init__()
        self.ctx = ctx
        self.left = left
        self.key = key

    def __str__(self) -> str:
        # [FUNCTION DOC] Returns text of the form left[key] (string keys are quoted).
        return f"{self.left}[{get_val(self.key)}]"

    def execute(self):
        # [FUNCTION DOC] Executes the left node and returns its element at key.
        left = self.left.execute()
        return left[self.key]


class CaptureSetItem(Capture):
    # [CLASS DOC] Operation node for a recorded item assignment of value into left at key.
    def __init__(self, left, key, value, ctx) -> None:
        # [FUNCTION DOC] Stores the shared ctx, the left node, the key and the value node.
        super().__init__()
        self.ctx = ctx
        self.left = left
        self.key = key
        self.value = value

    def __str__(self) -> str:
        # [FUNCTION DOC] Returns text of the form left[key] = value.
        return f"{self.left}[{get_val(self.key)}] = {self.value}"

    def execute(self) -> None:
        # [FUNCTION DOC] Executes left and value nodes, then assigns the value at key in
        # the left object (in-place mutation of the real DataFrame).
        left = self.left.execute()
        value = self.value.execute()
        left[self.key] = value


class CaptureAdd(Capture):
    # [CLASS DOC] Node for a recorded addition of left and right operands.
    def __init__(self, left, right, ctx) -> None:
        # [FUNCTION DOC] Stores the shared ctx and the two operands.
        super().__init__()
        self.ctx = ctx
        self.left = left
        self.right = right

    def __str__(self) -> str:
        # [FUNCTION DOC] Returns text of the form left + right.
        return f"{self.left} + {self.right}"

    def execute(self):
        # [FUNCTION DOC] Evaluates both operands (captures are executed) and returns their sum.
        return get_val(self.left) + get_val(self.right)


class CaptureMul(Capture):
    # [CLASS DOC] Node for a recorded multiplication of left and right operands.
    def __init__(self, left, right, ctx) -> None:
        # [FUNCTION DOC] Stores the shared ctx and the two operands.
        super().__init__()
        self.ctx = ctx
        self.left = left
        self.right = right

    def __str__(self) -> str:
        # [FUNCTION DOC] Returns text of the form left * right.
        return f"{self.left} * {self.right}"

    def execute(self):
        # [FUNCTION DOC] Evaluates both operands and returns their product.
        return get_val(self.left) * get_val(self.right)


class CaptureSub(Capture):
    # [CLASS DOC] Node for a recorded subtraction of right from left.
    def __init__(self, left, right, ctx) -> None:
        # [FUNCTION DOC] Stores the shared ctx and the two operands.
        super().__init__()
        self.ctx = ctx
        self.left = left
        self.right = right

    def __str__(self) -> str:
        # [FUNCTION DOC] Returns text of the form left - right.
        return f"{self.left} - {self.right}"

    def execute(self):
        # [FUNCTION DOC] Evaluates both operands and returns left minus right.
        return get_val(self.left) - get_val(self.right)


class CaptureGetAttr(Capture):
    # [CLASS DOC] Node for a recorded attribute read of name from src.
    def __init__(self, src, name, ctx) -> None:
        # [FUNCTION DOC] Stores the shared ctx, the source node and the attribute name.
        super().__init__()
        self.ctx = ctx
        self.src = src
        self.name = name

    def __str__(self) -> str:
        # [FUNCTION DOC] Returns text of the form src.name.
        return f"{self.src}.{self.name}"

    def execute(self):
        # [FUNCTION DOC] Evaluates the source and returns its attribute name.
        val = get_val(self.src)
        return getattr(val, self.name)


def get_val(capture):
    # [FUNCTION DOC] Resolves a value: a Capture is executed to its result, a string is
    # returned wrapped in double quotes, and anything else is returned unchanged.
    # KNOWN ISSUE: because strings are quoted, executing a node whose operand is a plain
    # string yields the quoted text, which suits printing but not real computation.
    if isinstance(capture, Capture):
        return capture.execute()
    elif isinstance(capture, str):
        return f'"{capture}"'
    else:
        return capture


class CaptureInitial(CaptureVariable):
    # [CLASS DOC] Root input variable of a capture, owning a brand-new context that carries
    # the schema DataFrame; its name is prefixed with input_.
    def __init__(self, schema_df=None) -> None:
        # [FUNCTION DOC] Creates a new context (empty operations and variables plus
        # schema_df), registers this variable in it, and prefixes the name with input_.
        # pyrefly: ignore [bad-assignment]
        new_ctx: dict[str, list[Any]] = {
            "operations": [],
            "variables": [],
            "schema_df": schema_df,
        }
        super().__init__(None, new_ctx)
        self.name = f"input_{self.name}"


class CaptureDataFrame(CaptureInitial):
    # [CLASS DOC] Public placeholder for a DataFrame whose operations are captured; adds
    # no behavior over CaptureInitial.
    pass


class CaptureDataFrameWithDataPipeOps(CaptureDataFrame):
    # [CLASS DOC] Captured DataFrame that also behaves like a datapipe: datapipe methods
    # (batch, groupby, shuffle, filter, collate) are mapped to DataFrame-aware pipes, and
    # names listed in DATAPIPES_OPS are forwarded to the traced datapipe.
    def as_datapipe(self):
        # [FUNCTION DOC] Returns a DataFrameTracedOps pipe over the first context
        # variable's source_datapipe that replays the captured operations on each item.
        return DataFrameTracedOps(self.ctx["variables"][0].source_datapipe, self)

    def raw_iterator(self):
        # [FUNCTION DOC] Returns an iterator over the traced DataFrames (not split to rows).
        return self.as_datapipe().__iter__()

    def __iter__(self):
        # [FUNCTION DOC] Returns an iterator over individual row tuples, via the
        # as-tuples datapipe.
        return iter(self._dataframes_as_tuples())

    def batch(self, batch_size=10, drop_last: bool = False, wrapper_class=DataChunkDF):
        # [FUNCTION DOC] Re-chunks the stream into DataFrames of batch_size rows (per-row
        # split then concat), wraps them one per batch in wrapper_class, and flags the
        # result as containing DataFrames. Returns the resulting datapipe.
        # KNOWN ISSUE: drop_last is applied to the outer batch of size 1, so it never
        # drops a partial DataFrame of fewer than batch_size rows.
        dp = self._dataframes_per_row()._dataframes_concat(batch_size)
        dp = dp.as_datapipe().batch(1, drop_last=drop_last, wrapper_class=wrapper_class)
        dp._dp_contains_dataframe = True
        return dp

    def groupby(
        self,
        group_key_fn,
        *,
        buffer_size=10000,
        group_size=None,
        guaranteed_group_size=None,
        drop_remaining=False,
    ):
        # [FUNCTION DOC] Splits the stream into rows and applies the regular datapipe
        # groupby with the given key function and buffer and group size options. Returns
        # the grouped datapipe.
        dp = self._dataframes_per_row()
        dp = dp.as_datapipe().groupby(
            group_key_fn,
            buffer_size=buffer_size,
            group_size=group_size,
            guaranteed_group_size=guaranteed_group_size,
            drop_remaining=drop_remaining,
        )
        return dp

    def shuffle(self, *args, **kwargs):
        # [FUNCTION DOC] Forwards all arguments to the DataFrame shuffle datapipe.
        return self._dataframes_shuffle(*args, **kwargs)

    def filter(self, *args, **kwargs):
        # [FUNCTION DOC] Forwards all arguments to the DataFrame filter datapipe.
        return self._dataframes_filter(*args, **kwargs)

    def collate(self, *args, **kwargs) -> NoReturn:
        # [FUNCTION DOC] Always raises RuntimeError; collating needs batched DataFrames.
        raise RuntimeError("Can't collate unbatched DataFrames stream")

    def __getattr__(self, attrname):  # ?
        # [FUNCTION DOC] Raises AttributeError for names in UNIMPLEMENTED_ATTR; forwards
        # names in DATAPIPES_OPS to the traced datapipe; otherwise falls back to the
        # capture behavior of recording the attribute access.
        if attrname in UNIMPLEMENTED_ATTR:
            raise AttributeError("Attempting to get ", attrname)
        if attrname in DATAPIPES_OPS:
            return (self.as_datapipe()).__getattr__(attrname)
        return super().__getattr__(attrname)


@functional_datapipe("trace_as_dataframe")
class DataFrameTracer(CaptureDataFrameWithDataPipeOps, IterDataPipe):  # type: ignore[misc]
    # [CLASS DOC] Entry point (functional name trace_as_dataframe) that wraps a source
    # datapipe of DataFrames as a capturable DataFrame object. State: source_datapipe.
    # Not shardable.
    source_datapipe: Any | None = None

    # TODO(VitalyFedyunin): Must implement all special functions of datapipes

    def set_shuffle_settings(self, *args, **kwargs) -> None:
        # [FUNCTION DOC] No-op; accepts and ignores shuffle settings.
        pass

    def is_shardable(self) -> bool:
        # [FUNCTION DOC] Always returns False; this pipe cannot be sharded.
        return False

    def __init__(self, source_datapipe, schema_df=None) -> None:
        # [FUNCTION DOC] Stores the source datapipe. If schema_df is not given, it is the
        # first item taken from the source (consumes one iteration of it). Then sets up
        # the capture context via the base constructor.
        self.source_datapipe = source_datapipe
        if schema_df is None:
            schema_df = next(iter(self.source_datapipe))
        super().__init__(schema_df=schema_df)
