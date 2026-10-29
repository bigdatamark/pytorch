# [MODULE DOC] Developer script (not a test) that suggests xfail and skip entries for the
# functorch op-db tests. Executes at import time: opens result.txt from the current
# directory (raises FileNotFoundError if missing), keeps lines starting with FAILED,
# extracts the test names with a regex, groups failures by test family (the base_names set),
# and prints for each family a banner and the suggested OpInfo-style entries. Run after
# collecting pytest output as described in the string literal below the imports. No files
# are written.
import re

import torch


"""
Instructions:

1. pytest -n 8 test/test_vmap.py test/test_ops.py test/test_aotdispatch.py > result.txt
2. python test/xfail_suggester.py
"""

with open("result.txt") as f:
    lines = f.readlines()

failed = [line for line in lines if line.startswith("FAILED")]
p = re.compile("FAILED test/test_\w+.py::\w+::(\S+)")  # noqa: W605


def get_failed_test(line):
    # [FUNCTION DOC] HELPER: Extracts the test method name from one pytest summary line
    # using the module-level regex, which only matches FAILED lines for test files named
    # test_*.py. Returns the captured name, or None if the line does not match.
    m = p.match(line)
    if m is None:
        return None
    return m.group(1)


base_names = {
    "test_grad_",
    "test_vjp_",
    "test_vmapvjp_",
    "test_vmapvjp_has_batch_rule_",
    "test_vjpvmap_",
    "test_jvp_",
    "test_vmapjvp_",
    "test_vmapjvpall_has_batch_rule_",
    "test_vmapjvpall_",
    "test_jvpvjp_",
    "test_vjpvjp_",
    "test_decomposition_",
    "test_make_fx_exhaustive_",
    "test_vmap_exhaustive_",
    "test_op_has_batch_rule_",
    "test_vmap_autograd_grad_",
}

failed_tests = [get_failed_test(line) for line in lines]
failed_tests = [match for match in failed_tests if match is not None]
failed_tests = sorted(failed_tests)

suggested_xfails = {}


def remove_device_dtype(test):
    # [FUNCTION DOC] HELPER: Strips the trailing device and dtype components from a test
    # name by splitting on underscores, dropping the last two pieces and rejoining the rest.
    return "_".join(test.split("_")[:-2])


def belongs_to_base(test, base):
    # [FUNCTION DOC] HELPER: Returns True if the test name starts with the given family
    # prefix and does not start with any other, longer family prefix from base_names (so
    # more specific families win). Returns False otherwise.
    if not test.startswith(base):
        return False
    candidates = [try_base for try_base in base_names if len(try_base) > len(base)]
    for candidate in candidates:
        if test.startswith(candidate):
            return False
    return True


def parse_namespace(base):
    # [FUNCTION DOC] HELPER: Splits a test suffix into an optional torch namespace and the
    # rest. Checks a fixed mapping of name prefixes (nn_functional_, fft_, linalg_,
    # _masked_, sparse_, special_) in order; on the first prefix match returns the namespace
    # name and the remainder with the prefix removed. Otherwise returns None and the
    # unchanged input.
    mappings = {
        "nn_functional_": "nn.functional",
        "fft_": "fft",
        "linalg_": "linalg",
        "_masked_": "_masked",
        "sparse_": "sparse",
        "special_": "special",
    }
    for heading in mappings:
        if base.startswith(heading):
            return mappings[heading], base[len(heading) :]
    return None, base


def get_torch_module(namespace):
    # [FUNCTION DOC] HELPER: Maps a namespace name to the torch module object: None gives
    # torch itself, the nn.functional name gives torch.nn.functional, anything else is
    # looked up as an attribute of torch (raises AttributeError if absent).
    if namespace is None:
        return torch
    if namespace == "nn.functional":
        return torch.nn.functional
    return getattr(torch, namespace)


def parse_base(base):
    # [FUNCTION DOC] HELPER: Splits a test suffix into namespace, API name and variant. Gets
    # the namespace and remainder, lists the attribute names of that torch module sorted
    # longest first, and picks the first name that prefixes the remainder; the variant is
    # the text after that name plus one separator character. If nothing matches the whole
    # remainder is the API and the variant is empty. Prints the base, namespace, API and
    # variant as a debug side effect. Returns the tuple of namespace, API, variant.
    namespace, rest = parse_namespace(base)

    apis = dir(get_torch_module(namespace))
    apis = sorted(apis, key=lambda x: -len(x))

    api = rest
    variant = ""
    for candidate in apis:
        if rest.startswith(candidate):
            api = candidate
            variant = rest[len(candidate) + 1 :]
            break
    print(base, namespace, api, variant)
    return namespace, api, variant


def any_starts_with(strs, thing):
    # [FUNCTION DOC] HELPER: Returns True if any string in the iterable starts with the
    # given prefix, else False. Not referenced elsewhere in this script.
    for s in strs:
        if s.startswith(thing):
            return True
    return False


def get_suggested_xfails(base, tests):
    # [FUNCTION DOC] HELPER: Builds the suggestion lines for one test family. Keeps the
    # failed test names belonging to the family (with the family prefix removed), reduces
    # them to distinct names without device and dtype, and for each one parses namespace,
    # API and variant. Emits an xfail line with no device if both the cpu and cuda float32
    # variants failed, an xfail line restricted to cpu or to cuda if only that one failed,
    # or otherwise a skip line. Returns the list of strings.
    # KNOWN ISSUE: the skip line is emitted without a closing parenthesis, unlike the xfail
    #     lines, and the loop variable shadows the base parameter.
    result = []
    tests = [test[len(base) :] for test in tests if belongs_to_base(test, base)]

    base_tests = {remove_device_dtype(test) for test in tests}
    tests = set(tests)
    for base in base_tests:
        cpu_variant = base + "_cpu_float32"
        cuda_variant = base + "_cuda_float32"
        namespace, api, variant = parse_base(base)
        if namespace is not None:
            api = f"{namespace}.{api}"
        if cpu_variant in tests and cuda_variant in tests:
            result.append(f"xfail('{api}', '{variant}'),")
            continue
        if cpu_variant in tests:
            result.append(f"xfail('{api}', '{variant}', device_type='cpu'),")
            continue
        if cuda_variant in tests:
            result.append(f"xfail('{api}', '{variant}', device_type='cuda'),")
            continue
        result.append(f"skip('{api}', '{variant}',")
    return result


result = {base: get_suggested_xfails(base, failed_tests) for base in base_names}
for k, v in result.items():
    print("=" * 50)
    print(k)
    print("=" * 50)
    print("\n".join(v))
