# [MODULE DOC] Utilities for writing operator micro-benchmarks: config generation helpers
# (cross product, config lists, random sampling, op lists), argument parsing helpers
# (boolean parsing, operator range, comma lists), thread environment setters and a numpy
# random tensor helper. Module constants define the reserved config keywords (probs,
# total_samples, tags) and the supported device names (cpu, cuda, mps, xpu).
import argparse
import bisect
import itertools
import os
import random

import numpy as np


"""Performance microbenchmarks's utils.

This module contains utilities for writing microbenchmark tests.
"""

# Here are the reserved keywords in the benchmark suite
_reserved_keywords = {"probs", "total_samples", "tags"}
_supported_devices = {"cpu", "cuda", "mps", "xpu"}


def shape_to_string(shape):
    # [FUNCTION DOC] Returns a comma plus space joined string of the elements of a shape.
    return ", ".join([str(x) for x in shape])


def str2bool(v):
    # [FUNCTION DOC] Converts a command line value to a boolean for argparse. Booleans pass
    # through; the strings yes, true, t, y and 1 (any case) give True, and no, false, f, n
    # and 0 give False. Anything else raises argparse.ArgumentTypeError.
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def numpy_random(dtype, *shapes):
    # [FUNCTION DOC] Returns a numpy array of uniform random values in the
    # half-open range zero to one with the given shape dimensions, cast to the
    # given numpy dtype. Uses the global numpy random state.
    """Return a random numpy tensor of the provided dtype.
    Args:
        shapes: int or a sequence of ints to defining the shapes of the tensor
        dtype: use the dtypes from numpy
            (https://numpy.org/doc/stable/user/basics.types.html)
    Return:
        numpy tensor of dtype
    """
    # TODO: consider more complex/custom dynamic ranges for
    # comprehensive test coverage.
    return np.random.rand(*shapes).astype(dtype)


def set_omp_threads(num_threads):
    # [FUNCTION DOC] Sets the OMP_NUM_THREADS environment variable of this
    # process to the given count, printing a notice if a different value
    # was already set. Side effect: mutates os.environ.
    existing_value = os.environ.get("OMP_NUM_THREADS", "")
    if existing_value != "":
        print(
            f"Overwriting existing OMP_NUM_THREADS value: {existing_value}; Setting it to {num_threads}."
        )
    os.environ["OMP_NUM_THREADS"] = str(num_threads)


def set_mkl_threads(num_threads):
    # [FUNCTION DOC] Sets the MKL_NUM_THREADS environment variable of this process to the given
    # count, printing a notice if a value was already set. Side effect: mutates os.environ.
    existing_value = os.environ.get("MKL_NUM_THREADS", "")
    if existing_value != "":
        print(
            f"Overwriting existing MKL_NUM_THREADS value: {existing_value}; Setting it to {num_threads}."
        )
    os.environ["MKL_NUM_THREADS"] = str(num_threads)


def cross_product(*inputs):
    # [FUNCTION DOC] Returns the cartesian product of the given iterables as a list of
    # tuples (the docstring's generator-style example is only illustrative).
    """
    Return a list of cartesian product of input iterables.
    For example, cross_product(A, B) returns ((x,y) for x in A for y in B).
    """
    return list(itertools.product(*inputs))


def get_n_rand_nums(min_val, max_val, n):
    # [FUNCTION DOC] Returns a list of n distinct integers sampled from the range from min_val up
    # to but excluding max_val. Side effect: reseeds the global random module with a fixed value
    # first, so the result is deterministic but also resets the global random state.
    random.seed((1 << 32) - 1)
    return random.sample(range(min_val, max_val), n)


def generate_configs(**configs):
    # [FUNCTION DOC] Generates config combinations with a user supplied sampling function.
    # Requires a sample_func keyword (AssertionError otherwise). For each other keyword it
    # builds a list of single-entry mappings (name to one value) and then returns the result of
    # calling the sampling function with those lists as positional arguments.
    """
    Given configs from users, we want to generate different combinations of
    those configs
    For example, given M = ((1, 2), N = (4, 5)) and sample_func being cross_product,
    we will generate (({'M': 1}, {'N' : 4}),
                      ({'M': 1}, {'N' : 5}),
                      ({'M': 2}, {'N' : 4}),
                      ({'M': 2}, {'N' : 5}))
    """
    if "sample_func" not in configs:
        raise AssertionError("Missing sample_func to generate configs")
    result = []
    for key, values in configs.items():
        if key == "sample_func":
            continue
        tmp_result = []
        for value in values:
            tmp_result.append({key: value})
        result.append(tmp_result)

    results = configs["sample_func"](*result)
    return results


def cross_product_configs(**configs):
    # [FUNCTION DOC] Returns the cartesian product of all keyword value lists as a list of
    # tuples, where each tuple holds one single-entry mapping per keyword. First validates
    # the configs (device names). Converts the lazy product into a list.
    """
    Given configs from users, we want to generate different combinations of
    those configs
    For example, given M = ((1, 2), N = (4, 5)),
    we will generate (({'M': 1}, {'N' : 4}),
                      ({'M': 1}, {'N' : 5}),
                      ({'M': 2}, {'N' : 4}),
                      ({'M': 2}, {'N' : 5}))
    """
    _validate(configs)
    configs_attrs_list = []
    for key, values in configs.items():
        tmp_results = [{key: value} for value in values]
        configs_attrs_list.append(tmp_results)

    # TODO(mingzhe0908) remove the conversion to list.
    # itertools.product produces an iterator that produces element on the fly
    # while converting to a list produces everything at the same time.
    generated_configs = list(itertools.product(*configs_attrs_list))
    return generated_configs


def _validate(configs):
    # [FUNCTION DOC] HELPER: Checks the user configs. If a device entry is present, every listed
    # device must be one of the supported devices, otherwise AssertionError is raised.
    """Validate inputs from users."""
    if "device" in configs:
        for v in configs["device"]:
            if v not in _supported_devices:
                raise AssertionError(
                    f"Device must be one of {_supported_devices}, got {v}"
                )


def config_list(**configs):
    # [FUNCTION DOC] Builds the config list from explicit attribute rows. Requires attrs,
    # attr_names and tags keywords (ValueError if any is missing) and validates the device
    # entry. Each row of attrs becomes a list of single-entry mappings keyed by attr_names,
    # followed by a tags entry holding the tags joined by underscores. If
    # cross_product_configs is given, every row is repeated once per combination of that
    # cross product, with the combination appended. Returns the list of lists.
    """Generate configs based on the list of input shapes.
    This function will take input shapes specified in a list from user. Besides
    that, all other parameters will be cross produced first and each of the
    generated list will be merged with the input shapes list.

    Reserved Args:
        attr_names(reserved): a list of names for input shapes.
        attrs(reserved): a list of values for each input shape.
        corss_product: a dictionary of attributes which will be
                       cross produced with the input shapes.
        tags(reserved): a tag used to filter inputs.

    Here is an example:
    attrs = [
        [1, 2],
        [4, 5],
    ],
    attr_names = ['M', 'N'],
    cross_product_configs={
        'device': ['cpu', 'cuda'],
    },

    we will generate [[{'M': 1}, {'N' : 2}, {'device' : 'cpu'}],
                      [{'M': 1}, {'N' : 2}, {'device' : 'cuda'}],
                      [{'M': 4}, {'N' : 5}, {'device' : 'cpu'}],
                      [{'M': 4}, {'N' : 5}, {'device' : 'cuda'}]]
    """
    generated_configs = []
    reserved_names = ["attrs", "attr_names", "tags"]
    if any(attr not in configs for attr in reserved_names):
        raise ValueError("Missing attrs in configs")

    _validate(configs)

    cross_configs = None
    if "cross_product_configs" in configs:
        cross_configs = cross_product_configs(**configs["cross_product_configs"])

    for inputs in configs["attrs"]:
        tmp_result = [
            {configs["attr_names"][i]: input_value}
            for i, input_value in enumerate(inputs)
        ]
        # TODO(mingzhe0908):
        # If multiple 'tags' were provided, do they get concat?
        # If a config has both ['short', 'medium'], it should match
        # both 'short' and 'medium' tag-filter?
        tmp_result.append({"tags": "_".join(configs["tags"])})
        if cross_configs:
            generated_configs += [tmp_result + list(config) for config in cross_configs]
        else:
            generated_configs.append(tmp_result)

    return generated_configs


def attr_probs(**probs):
    # [FUNCTION DOC] Returns the keyword arguments as a mapping; used to declare sampling weights.
    """return the inputs in a dictionary"""
    return probs


class RandomSample:
    # [CLASS DOC] Weighted random sampler over config value lists. Holds the configs mapping
    # (value lists per name plus a probs mapping of weights) and a cache of cumulative
    # distributions per name. get_one_set_of_inputs draws one value for every non-reserved name.
    def __init__(self, configs):
        # [FUNCTION DOC] Stores the configs and creates an
        # empty cache of cumulative distributions.
        self.saved_cum_distribution = {}
        self.configs = configs

    def _distribution_func(self, key, weights):
        # [FUNCTION DOC] HELPER: Returns the normalized cumulative
        # distribution for the weights of a name, computed on first use and
        # cached under that name. The last value is one up to rounding.
        """this is a cumulative distribution function used for random sampling inputs"""
        if key in self.saved_cum_distribution:
            return self.saved_cum_distribution[key]

        total = sum(weights)
        result = []
        cumsum = 0
        for w in weights:
            cumsum += w
            result.append(cumsum / total)
        self.saved_cum_distribution[key] = result
        return result

    def _random_sample(self, key, values, weights):
        # [FUNCTION DOC] HELPER: Picks one value with probability proportional to its weight.
        # Raises AssertionError if the numbers of values and weights differ or the found
        # position is beyond the list. Draws a uniform number from the global random module,
        # finds its position in the cumulative distribution, and clamps the position to the
        # last value to absorb rounding error. Returns the chosen value.
        """given values and weights, this function randomly sample values based their weights"""
        # TODO(mingzhe09088): cache the results to avoid recalculation overhead
        if len(values) != len(weights):
            raise AssertionError(
                f"values and weights must have same length, got {len(values)} and {len(weights)}"
            )
        _distribution_func_vals = self._distribution_func(key, weights)
        x = random.random()
        idx = bisect.bisect(_distribution_func_vals, x)

        if idx > len(values):
            raise AssertionError(
                f"Wrong index value is returned: idx={idx}, len(values)={len(values)}"
            )
        # Due to numerical property, the last value in cumsum could be slightly
        # smaller than 1, and lead to the (index == len(values)).
        if idx == len(values):
            idx -= 1
        return values[idx]

    def get_one_set_of_inputs(self):
        # [FUNCTION DOC] Draws one sample for each config name other than the reserved
        # keywords, using the weights stored under probs for that name (a name without
        # weights raises KeyError). Returns a list of single-entry mappings.
        tmp_attr_list = []
        for key, values in self.configs.items():
            if key in _reserved_keywords:
                continue
            value = self._random_sample(key, values, self.configs["probs"][str(key)])
            tmp_results = {key: value}
            tmp_attr_list.append(tmp_results)
        return tmp_attr_list


def random_sample_configs(**configs):
    # [FUNCTION DOC] Generates total_samples randomly sampled configs. Each is a
    # list of single-entry mappings for every value list plus a tags entry (tags
    # joined by underscores). Raises ValueError when probs is missing.
    # KNOWN ISSUE: the docstring says probs is optional and defaults to
    # equal weights, but the code raises ValueError if probs is absent and
    # also fails with KeyError for any name lacking weights.
    """
    This function randomly sample <total_samples> values from the given inputs based on
    their weights.
    Here is an example showing what are the expected inputs and outputs from this function:
    M = [1, 2],
    N = [4, 5],
    K = [7, 8],
    probs = attr_probs(
        M = [0.7, 0.2],
        N = [0.5, 0.2],
        K = [0.6, 0.2],
    ),
    total_samples=10,
    this function will generate
    [
        [{'K': 7}, {'M': 1}, {'N': 4}],
        [{'K': 7}, {'M': 2}, {'N': 5}],
        [{'K': 8}, {'M': 2}, {'N': 4}],
        ...
    ]
    Note:
    The probs is optional. Without them, it implies everything is 1. The probs doesn't
    have to reflect the actual normalized probability, the implementation will
    normalize it.
    TODO (mingzhe09088):
    (1):  a lambda that accepts or rejects a config as a sample. For example: for matmul
    with M, N, and K, this function could get rid of (M * N * K > 1e8) to filter out
    very slow benchmarks.
    (2): Make sure each sample is unique. If the number of samples are larger than the
    total combinations, just return the cross product. Otherwise, if the number of samples
    is close to the number of cross-products, it is numerical safer to generate the list
    that you don't want, and remove them.
    """
    if "probs" not in configs:
        raise ValueError(
            "probs is missing. Consider adding probs or using other config functions"
        )

    configs_attrs_list = []
    randomsample = RandomSample(configs)
    for i in range(configs["total_samples"]):
        tmp_attr_list = randomsample.get_one_set_of_inputs()
        tmp_attr_list.append({"tags": "_".join(configs["tags"])})
        configs_attrs_list.append(tmp_attr_list)
    return configs_attrs_list


def op_list(**configs):
    # [FUNCTION DOC] Turns the attrs rows and attr_names into a list of mappings from each name
    # to the row's value, for example operator name and function. Raises ValueError if attrs is
    # missing; a missing attr_names would fail with KeyError. Returns the list.
    """Generate a list of ops organized in a specific format.
    It takes two parameters which are "attr_names" and "attr".
    attrs stores the name and function of operators.
    Args:
        configs: key-value pairs including the name and function of
        operators. attrs and attr_names must be present in configs.
    Return:
        a sequence of dictionaries which stores the name and function
        of ops in a specifal format
    Example:
    attrs = [
        ["abs", torch.abs],
        ["abs_", torch.abs_],
    ]
    attr_names = ["op_name", "op"].

    With those two examples,
    we will generate (({"op_name": "abs"}, {"op" : torch.abs}),
                      ({"op_name": "abs_"}, {"op" : torch.abs_}))
    """
    generated_configs = []
    if "attrs" not in configs:
        raise ValueError("Missing attrs in configs")
    for inputs in configs["attrs"]:
        tmp_result = {
            configs["attr_names"][i]: input_value
            for i, input_value in enumerate(inputs)
        }
        generated_configs.append(tmp_result)
    return generated_configs


def get_operator_range(chars_range):
    # [FUNCTION DOC] Parses an operator range option into a set of lower-case characters. Returns
    # None for None or the string None. Raises ValueError if the text contains neither a comma nor
    # a dash. Items are comma separated: a single character is added as is, a dash pair adds every
    # character from start to end inclusive. Used to filter operators by first letter.
    """Generates the characters from chars_range inclusive."""
    if chars_range == "None" or chars_range is None:
        return None

    if all(item not in chars_range for item in [",", "-"]):
        raise ValueError(
            "The correct format for operator_range is "
            "<start>-<end>, or <point>, <start>-<end>"
        )

    ops_start_chars_set = set()
    ranges = chars_range.split(",")
    for item in ranges:
        if len(item) == 1:
            ops_start_chars_set.add(item.lower())
            continue
        start, end = item.split("-")
        ops_start_chars_set.update(
            chr(c).lower() for c in range(ord(start), ord(end) + 1)
        )
    return ops_start_chars_set


def process_arg_list(arg_list):
    # [FUNCTION DOC] Parses a comma separated option string. Returns None for
    # the string None, otherwise a list of stripped non-empty items.
    if arg_list == "None":
        return None

    return [fr.strip() for fr in arg_list.split(",") if len(fr.strip()) > 0]
