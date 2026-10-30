# [MODULE DOC] Utilities to record the inputs of ATen operators during a model run and to replay
# them later for operator microbenchmarks. Defines abbreviation tables for dtypes, a call-
# description serializer, a dispatch mode that tallies unique operator inputs, a text-file
# writer, and a loader that parses those files back into actual tensors. Import-time side
# effects: creates a module logger, computes the directories for the timm, huggingface and
# torchbench log folders next to this file, and looks up the aten namespace and the tensor type.
import functools
import logging
import math
import os
from collections import Counter, defaultdict
from collections.abc import Generator, Iterable
from functools import partial
from typing import Any

import torch
from torch.testing import make_tensor
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map


log = logging.getLogger(__name__)

OP_INP_DIRECTORY = os.path.join(os.path.dirname(__file__), "operator_inp_logs")

TIMM_DIR = os.path.join(OP_INP_DIRECTORY, "timm_train")
HF_DIR = os.path.join(OP_INP_DIRECTORY, "hf_train")
TORCHBENCH_DIR = os.path.join(OP_INP_DIRECTORY, "torchbench_train")

aten = torch.ops.aten
tensor_type = torch._C.TensorType.get()

dtype_abbrs = {
    torch.bfloat16: "bf16",
    torch.float64: "f64",
    torch.float32: "f32",
    torch.float16: "f16",
    torch.complex32: "c32",
    torch.complex64: "c64",
    torch.complex128: "c128",
    torch.int8: "i8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.bool: "b8",
    torch.uint8: "u8",
}

dtype_abbrs_parsing = {value: key for key, value in dtype_abbrs.items()}


def truncate_inp(arg):
    # [FUNCTION DOC] HELPER: Shorten one value for serialization: a dtype in the abbreviation
    # table becomes its short string, a device becomes its type string, anything else is
    # returned unchanged.
    if arg in dtype_abbrs:
        return dtype_abbrs[arg]
    elif isinstance(arg, torch.device):
        return arg.type
    else:
        return arg


# Serialize Function Call
class FuncCallWrapper:
    # [CLASS DOC] HELPER: Description of a call to a named constructor such as a
    # tensor or sparse tensor builder. Holds the call name, the positional arguments and the
    # keyword arguments, each with dtype and device values shortened. Its textual form is a call
    # expression whose dtype arguments appear unquoted, so the text can be evaluated later with
    # the deserializer helpers in scope.
    def __init__(self, call, *args, **kwargs):
        # [FUNCTION DOC] Store the call name and shorten dtypes and devices inside the
        # positional and keyword arguments with a tree traversal.
        self.call = call
        self.args = tree_map(truncate_inp, args)
        self.kwargs = tree_map(truncate_inp, kwargs) if kwargs is not None else {}

    def __repr__(self):
        # [FUNCTION DOC] Build the textual call description: positional args joined by commas,
        # keyword args appended as name equals value text, wrapped in the call name, with outer
        # double quotes stripped and quoted dtype abbreviations unquoted. Returns the string.
        args = ", ".join([repr(arg) for arg in self.args])
        kwargs = "".join(
            [f", {str(key)}={value}" for key, value in self.kwargs.items()]
        )
        out = f"{self.call}({args}{kwargs})".strip('"')
        # f strings introduce quotations we dont want
        for key in dtype_abbrs_parsing:
            out = out.replace(f"'{key}'", key)
        return out


def serialize_sparse_tensor(e):
    # [FUNCTION DOC] Describe a sparse tensor as a call wrapper holding shape list, dtype,
    # layout and coalesced flag, with the number of nonzeros appended unless the tensor is fake.
    if isinstance(e, torch._subclasses.FakeTensor):
        return FuncCallWrapper("ST", list(e.shape), e.dtype, e.layout, e.is_coalesced())
    else:
        return FuncCallWrapper(
            "ST", list(e.shape), e.dtype, e.layout, e.is_coalesced(), e._nnz()
        )


def deserialize_sparse_tensor(size, dtype, layout, is_coalesced, nnz=None):
    # [FUNCTION DOC] Placeholder for rebuilding a sparse tensor from its description; always
    # raises NotImplementedError.
    raise NotImplementedError


def deserialize_tensor(size, dtype, stride=None):
    # [FUNCTION DOC] Rebuild a CPU tensor from a recorded size, dtype and optional stride.
    # Allocates an empty (strided if a stride is given) tensor and tries to fill it with random
    # values of that dtype; if filling fails the error is printed and the unfilled tensor is
    # returned. Returns the tensor.
    if stride is not None:
        out = torch.empty_strided(size, stride, dtype=dtype)
    else:
        out = torch.empty(size, dtype=dtype)
    try:
        out.copy_(make_tensor(size, dtype=dtype, device="cpu"))
    except Exception as e:
        print(e)
        return out
    return out


def serialize_tensor(e):
    # [FUNCTION DOC] Describe a dense tensor as a call wrapper holding shape list and dtype,
    # adding strides only if it is not contiguous.
    if not e.is_contiguous():
        return FuncCallWrapper("T", list(e.shape), e.dtype, stride=e.stride())
    else:
        return FuncCallWrapper("T", list(e.shape), e.dtype)


def serialize_torch_args(e):
    # [FUNCTION DOC] Serialize one dispatch argument: sparse tensors and dense tensors become
    # call wrappers, any other value is shortened by the truncation helper.
    if isinstance(e, torch.Tensor):
        if e.is_sparse:
            return serialize_sparse_tensor(e)
        return serialize_tensor(e)
    else:
        return truncate_inp(e)


def contains_tensor(elems):
    # [FUNCTION DOC] Return True if any leaf of the nested structure is a tensor, else False.
    for elem in pytree.tree_leaves(elems):
        if isinstance(elem, torch.Tensor):
            return True
    return False


def skip_args(elems):
    # [FUNCTION DOC] Return True if any leaf of the nested structure is a memory format or an
    # untyped storage (values seen only in constructors and similar ops), else False.
    for i in pytree.tree_leaves(elems):
        # only shows up in constructors and ops like that
        if isinstance(i, (torch.memory_format, torch.storage.UntypedStorage)):
            return True
    return False


def contains_tensor_types(type):
    # [FUNCTION DOC] Recursively test whether a TorchScript type is a subtype of tensor or
    # contains such a type among its contained types (for example a list of tensors). Returns a
    # bool.
    return type.isSubtypeOf(tensor_type) or any(
        contains_tensor_types(e) for e in type.containedTypes()
    )


@functools.cache
def non_compute_operator(op):
    # [FUNCTION DOC] Cached predicate deciding whether an operator should be excluded from
    # benchmarks. True for ops with no tensor-typed argument (constructors), ops whose name
    # contains the like suffix, and ops where the single tensor output aliases a tensor input.
    # False for mutating ops and for ops without exactly one tensor output. Takes an OpOverload;
    # the result is memoized per op.
    schema = op._schema

    # skip constructors
    if not any(contains_tensor_types(arg.type) for arg in schema.arguments):
        return True
    if "_like" in op.name():
        return True

    # allow in place writes
    if schema.is_mutable:
        return False

    tensor_inps = [arg for arg in schema.arguments if arg.type is tensor_type]
    tensor_outputs = [ret for ret in schema.returns if ret.type is tensor_type]

    # skip aliasing unless there are multiple outputs
    if len(tensor_outputs) != 1:
        return False

    for inp in tensor_inps:
        if inp.alias_info and tensor_outputs[0].alias_info:
            if inp.alias_info.before_set.intersection(
                tensor_outputs[0].alias_info.after_set
            ):
                return True

    return False


class OperatorInputsMode(TorchDispatchMode):
    # [CLASS DOC] Dispatch mode that records, for every intercepted operator call, its
    # serialized inputs. State is a mapping from operator name text to a counter of serialized
    # input strings (pass an existing mapping to accumulate across runs). Only calls with
    # tensors in both inputs and outputs and without memory format or storage arguments are
    # recorded. The call itself is always executed unchanged.
    def __init__(self, func_db=None):
        # [FUNCTION DOC] Use the given operator database or start a fresh mapping of counters.
        self.func_db = defaultdict(Counter) if func_db is None else func_db

    def __torch_dispatch__(self, func_overload, types, args=(), kwargs=None):
        # [FUNCTION DOC] Serialize the arguments, execute the real operator, and when inputs and
        # outputs contain tensors and there is no memory format or storage argument, increment
        # the count for this operator and serialized input text. Returns the operator output.
        kwargs = kwargs if kwargs else {}
        arg_meta, kwarg_meta = tree_map(serialize_torch_args, (args, kwargs))

        out = func_overload(*args, **kwargs)

        inps = (args, kwargs)
        if contains_tensor(inps) and not skip_args(inps) and contains_tensor(out):
            serialized_str = repr((arg_meta, kwarg_meta))
            self.func_db[str(func_overload)][serialized_str] += 1

        return out

    def log_to_file(self, output_filename, *, skip_non_compute_operators=True):
        # [FUNCTION DOC] Write the recorded database to a text file, overwriting it. Operators
        # are sorted by name; unless disabled, non-compute operators are skipped (the name text
        # is evaluated into an operator object, relying on the aten name in scope). Each
        # operator gets a header line followed by one line per unique input with its count, with
        # quoted dtype abbreviations unquoted.
        sorted_operators = sorted(self.func_db.keys())
        with open(output_filename, "w") as f:
            for operator in sorted_operators:
                if skip_non_compute_operators and non_compute_operator(eval(operator)):
                    continue
                f.write(f"Operator: {operator}\n")
                operator_inputs = self.func_db[operator]
                for inps, count in operator_inputs.items():
                    f.write(f"cnt: {count}, ")
                    # repr will add quotation marks around the dtype strings
                    for dtype_abbr in dtype_abbrs.values():
                        inps = inps.replace("'" + dtype_abbr + "'", dtype_abbr)
                    f.write(inps)
                    f.write("\n")


def map_to_device(e, device):
    # [FUNCTION DOC] Map a value onto a target device: tensors are moved, device objects are
    # replaced by the target, the strings cuda and cpu become the target device type, other
    # values are returned unchanged.
    # KNOWN ISSUE: any other string value reaches the string branch and falls off the end, so
    #     None is returned instead of the string.
    if isinstance(e, torch.Tensor):
        return e.to(device)
    elif isinstance(e, torch.device):
        return device
    elif isinstance(e, str):
        if e == "cuda" or e == "cpu":
            return device.type
    else:
        return e


def map_to_dtype(e, dtype):
    # [FUNCTION DOC] Cast floating point tensors to the given dtype, replace dtype objects by
    # it, and leave everything else unchanged.
    if isinstance(e, torch.Tensor) and e.is_floating_point():
        return e.to(dtype)
    elif isinstance(e, torch.dtype):
        return dtype
    else:
        return e


def deserialize_args(inps):
    # [FUNCTION DOC] Turn a serialized input line back into Python objects by evaluating it with
    # a namespace providing the tensor builder, the sparse builder, torch, infinity and the
    # dtype abbreviations (after unquoting those abbreviations). Returns the evaluated pair of
    # positional args and keyword args. Evaluates arbitrary text, so only trusted files should
    # be used.
    inps = inps.strip().strip("'")
    global_vals = {
        "T": deserialize_tensor,
        "ST": deserialize_sparse_tensor,
        "th": torch,
        "inf": math.inf,
        "torch": torch,
        **dtype_abbrs_parsing,
    }
    # f strings introduce quotations we dont want
    for key in dtype_abbrs_parsing:
        inps = inps.replace(f"'{key}'", key)
    return eval(inps.strip().strip("'").strip('"'), global_vals)


class OperatorInputsLoader:
    # [CLASS DOC] Loader that reads an operator input log file into a mapping from operator name
    # text to a counter of serialized input strings with counts, and replays or summarizes them.
    # Several loaders can be merged. Factory static methods load the timm, huggingface and
    # torchbench log folders.
    def __init__(self, json_file_path):
        # [FUNCTION DOC] Parse the text file made by the log writer: each operator header line
        # is followed by lines with a count and the serialized inputs, which are summed per
        # input. The sum operator overload with a symbolic int is renamed to the dimension list
        # overload. Raises an AssertionError if a block does not begin with an operator header
        # line.
        self.operator_db = defaultdict(Counter)

        with open(json_file_path) as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            op_line = lines[i].strip("\n")
            if "Operator: " not in op_line:
                raise AssertionError(
                    f"Expected 'Operator: ' in line, but got: {op_line}"
                )
            operator = op_line[len("Operator: ") :]
            operator = (
                operator if operator != "aten.sum.SymInt" else "aten.sum.dim_IntList"
            )
            op_inps = Counter()
            i += 1
            while i < len(lines) and "Operator: " not in lines[i]:
                line = lines[i]
                cnt = eval(line[len("cnt: ") : line.find(",")])
                inps = line[line.find(",") + 2 :].strip("'")
                op_inps[inps] += cnt
                i += 1
            self.operator_db[operator] = op_inps

    def get_inputs_for_operator(
        self, operator, dtype=None, device="cuda"
    ) -> Generator[tuple[Iterable[Any], dict[str, Any]], None, None]:
        # [FUNCTION DOC] Generator over recorded inputs of one operator overload (given as an
        # object or its name). For each unique recorded input it rebuilds tensors, casts
        # floating tensors to the requested dtype unless it is absent or float16, moves
        # everything to the device (default cuda; skipped when empty) and yields the args and
        # kwargs pair. Embedding operators log a warning and yield a single None. Raises an
        # AssertionError, on first iteration, if the operator is unknown. Counts are ignored.
        if str(operator) not in self.operator_db:
            raise AssertionError(f"Could not find {operator}, must provide overload")

        if "embedding" in str(operator):
            log.warning("Embedding inputs NYI, input data cannot be randomized")
            yield
            return

        # line[1] represents number of times these inputs occurred, ignored for now
        for line in self.operator_db[str(operator)].items():
            inps = line[0]

            args, kwargs = deserialize_args(inps)

            # Backwards require some inputs to be float16 and some to be float32
            # So we record on half and upcast to float when specified
            if dtype and dtype != torch.float16:
                to_dtype = partial(map_to_dtype, dtype=dtype)
                args, kwargs = tree_map(to_dtype, (args, kwargs))

            if device:
                to_device = partial(map_to_device, device=torch.device(device))
                args, kwargs = tree_map(to_device, (args, kwargs))

            yield args, kwargs

    def get_all_ops(self):
        # [FUNCTION DOC] Generator over all recorded operators converted to operator objects by
        # evaluating their names; names that fail to evaluate with an attribute error are logged
        # as warnings and skipped.
        for key in self.operator_db:
            try:
                op = eval(key)
            except AttributeError:
                log.warning("Evaluating an op name into an OpOverload", exc_info=True)
                continue
            yield op

    def get_call_frequency(self, op):
        # [FUNCTION DOC] Total number of recorded calls for an operator, summing its input
        # counts. Raises an AssertionError if the operator is unknown.
        if str(op) not in self.operator_db:
            raise AssertionError(f"Could not find {op}, must provide overload")

        count = 0
        for counter in self.operator_db[str(op)].values():
            count += counter
        return count

    def merge(self, other):
        # [FUNCTION DOC] Add all counts from another loader into this one in place.
        for operator, counter_dict in other.operator_db.items():
            for inps, cnt in counter_dict.items():
                self.operator_db[operator][inps] += cnt

    @staticmethod
    def get_timm_loader():
        # [FUNCTION DOC] Return a loader merged from every log file of the timm training log
        # folder.
        return OperatorInputsLoader._load_directory(TIMM_DIR)

    @staticmethod
    def get_huggingface_loader():
        # [FUNCTION DOC] Return a loader merged from every log file of the huggingface training
        # log folder.
        return OperatorInputsLoader._load_directory(HF_DIR)

    @staticmethod
    def get_torchbench_loader():
        # [FUNCTION DOC] Return a loader merged from every log file of the torchbench training
        # log folder.
        return OperatorInputsLoader._load_directory(TORCHBENCH_DIR)

    @staticmethod
    def _load_directory(inp_dir):
        # [FUNCTION DOC] HELPER: Load and merge every text file in a directory into one loader.
        # Raises an AssertionError if the directory is missing. Returns None if the directory
        # has no text files.
        if not os.path.isdir(inp_dir):
            raise AssertionError(f"Directory does not exist: {inp_dir}")
        union = None
        for inp in os.listdir(inp_dir):
            if inp[-4:] != ".txt":
                continue
            path = os.path.join(inp_dir, inp)
            if union is None:
                union = OperatorInputsLoader(path)
            else:
                union.merge(OperatorInputsLoader(path))
        return union
