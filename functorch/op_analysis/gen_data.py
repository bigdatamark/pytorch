# [MODULE DOC] Script that produces CSV-like analysis files of ATen operator coverage
# (vmap.txt and decompositions.txt) by categorizing every op in native_functions.yaml.
# Import-time side effects: runs immediately on import. Needs a modified PyTorch build exposing
# dispatch registrations, native_functions.yaml at a relative path, and input files
# annotated_ops, run_ops.txt, count_ops.txt, run_decompositions.txt and public_api in the
# current directory; writes the two output files there.
import csv
from collections import defaultdict

import yaml

import torch


def get_ops_for_key(key):
    # [FUNCTION DOC] Returns the set of ATen operator names registered for a dispatch key (all
    # registrations when key is None). Only entries containing the aten:: prefix are kept, with
    # the first six characters removed and whitespace stripped.
    # Needs modified PyTorch C++ code to work
    if key is None:
        ops = torch._C._dispatch_get_registrations_for_dispatch_key()
    else:
        ops = torch._C._dispatch_get_registrations_for_dispatch_key(key)
    cleaned_ops = []
    for i in ops:
        if "aten::" not in i:
            continue
        cleaned_ops.append(i[6:].strip())
    return set(cleaned_ops)


def gen_data(special_op_lists, analysis_name):
    # [FUNCTION DOC] Loads native_functions.yaml and the annotated_ops file, derives name,
    # full name and return type for each op, labels each op with a category (see annotate_ops),
    # and writes one comma-separated line per op to the file analysis_name: full name, category,
    # whether it is composite implicit autograd, then the result of each predicate function in
    # special_op_lists. Reads files relative to the current directory; I/O errors propagate.
    # KNOWN ISSUE: the unique op list and overload bookkeeping (uniq_ops, overload_types, cnt)
    # are built but never used.
    all_ops = get_ops_for_key(None)
    composite_ops = get_ops_for_key("CompositeImplicitAutograd")
    noncomposite_ops = all_ops - composite_ops
    with open("../../aten/src/ATen/native/native_functions.yaml") as f:
        ops = yaml.load(f.read(), Loader=yaml.CLoader)

    with open("annotated_ops") as f:
        annotated_ops = {a.strip(): b.strip() for a, b in csv.reader(f)}

    uniq_ops = []
    uniq_names = set()
    overload_types = defaultdict(list)
    cnt = 0
    for op in ops:
        func_str = op["func"]
        name = func_str[: func_str.index("(")]
        if "." in name:
            uniq_name = name[: name.index(".")]
            overload_types[name[name.index(".") + 1 :]].append(name)
        else:
            uniq_name = name
        op["name"] = uniq_name
        full_name = func_str[: func_str.index("(")]
        op["full_name"] = full_name
        ret_type = func_str[func_str.index("->") + 3 :]
        op["ret_type"] = ret_type
        cnt += 1
        if uniq_name in uniq_names:
            continue
        uniq_names.add(uniq_name)
        uniq_ops.append(op)

    def annotate_ops(ops, is_unique):
        # [FUNCTION DOC] HELPER: Sets a meta label on each op mapping in place, using the first
        # matching rule: inplace, out (only when not is_unique), conv, pool, backward, private,
        # batch_norm, non_tensor, backend, or core (with the annotation if known, else unknown).
        # Returns a mapping of category to count.
        categorization = defaultdict(int)
        for op in ops:
            if op["name"][-1] == "_":
                categorization["inplace"] += 1
                op["meta"] = "inplace"
                continue
            if not is_unique and "a!" in op["func"].lower():
                categorization["out"] += 1
                op["meta"] = "out"
                continue
            if "conv" in op["name"]:
                categorization["conv"] += 1
                op["meta"] = "conv"
                continue
            if "pool" in op["name"]:
                categorization["pool"] += 1
                op["meta"] = "pool"
                continue
            if "backward" in op["name"]:
                categorization["backward"] += 1
                op["meta"] = "backward"
                continue
            if op["name"][0] == "_" and op["name"][1] != "_":
                categorization["private"] += 1
                op["meta"] = "private"
                continue
            if "batch_norm" in op["name"]:
                categorization["batch_norm"] += 1
                op["meta"] = "batch_norm"
                continue
            if "Tensor" not in op["func"] or "Tensor" not in op["ret_type"]:
                categorization["non_tensor"] += 1
                op["meta"] = "non_tensor"
                continue
            if (
                "cudnn" in op["name"]
                or "mkldnn" in op["name"]
                or "miopen" in op["name"]
                or "native" in op["name"]
                or "thnn" in op["name"]
                or "slow" in op["name"]
            ):
                categorization["backend"] += 1
                op["meta"] = "backend"
                continue
            if op["name"] in annotated_ops:
                categorization["core"] += 1
                op["meta"] = "core " + annotated_ops[op["name"]]
                continue
            categorization["core"] += 1
            op["meta"] = "core unknown"
        return categorization

    annotate_ops(ops, is_unique=False)
    with open(f"{analysis_name}", "w") as f:
        for op in ops:
            info = [
                op["full_name"],
                op["meta"],
                op["full_name"] not in noncomposite_ops,
            ] + [check(op) for check in special_op_lists]
            f.write(",".join([str(i) for i in info]) + "\n")


def name_check(lst):
    # [FUNCTION DOC] HELPER: Returns a predicate that tells whether an op's name is in lst.
    return lambda x: x["name"] in lst


def full_name_check(lst):
    # [FUNCTION DOC] HELPER: Returns a predicate that tells whether an op's full (overload
    # qualified) name is in lst.
    return lambda x: x["full_name"] in lst


# Generates batching rule data
gen_data([full_name_check(get_ops_for_key("FuncTorchBatched"))], "vmap.txt")


def remove_suffix(input_string, suffix):
    # [FUNCTION DOC] HELPER: Returns input_string without the trailing suffix if it ends with
    # a non-empty suffix; otherwise returns it unchanged.
    if suffix and input_string.endswith(suffix):
        return input_string[: -len(suffix)]
    return input_string


def remove_prefix(input_string, prefix):
    # [FUNCTION DOC] HELPER: Returns input_string without the leading prefix if it starts with
    # a non-empty prefix; otherwise returns it unchanged.
    if prefix and input_string.startswith(prefix):
        return input_string[len(prefix) :]
    return input_string


if True:
    with open("run_ops.txt") as f:
        opinfo_ops = [remove_suffix(i.strip(), ".default") for i in f]
    with open("count_ops.txt") as f:
        opinfo_counts = [i.strip() for i in f]
        opinfo_counts = defaultdict(int, dict(zip(opinfo_ops, opinfo_counts)))

    def count_fn(x):
        # [FUNCTION DOC] HELPER: Looks up the OpInfo test count for the op's full name (zero
        # when absent). Note counts are kept as the strings read from count_ops.txt.
        return opinfo_counts[x["full_name"]]

    with open("run_decompositions.txt") as f:
        decomposed_ops = [remove_suffix(i.strip(), ".default") for i in f]

    with open("public_api") as f:
        ref_api = [i.strip() for i in f]

    def has_ref_impl(x):
        # [FUNCTION DOC] HELPER: True if the op has a reference implementation listed in the
        # public_api file. Strips linalg_ and special_ prefixes from the name, then checks the
        # bare name and the name under nn.functional, fft, special and linalg namespaces.
        name = x["name"]
        for prefix in ["linalg_", "special_"]:
            name = remove_prefix(name, prefix)
        prefixes = ["nn.functional", "fft", "special", "linalg"]
        return (
            any(f"{prefix}.{name}" in ref_api for prefix in prefixes) or name in ref_api
        )

    gen_data(
        [
            full_name_check(opinfo_ops),
            full_name_check(decomposed_ops),
            count_fn,
            has_ref_impl,
        ],
        "decompositions.txt",
    )
