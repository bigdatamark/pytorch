# [MODULE DOC] Minimal parser for the text report printed by cuda-memcheck. Defines a parse error
# type, a report container, an error container and a parse function that splits a report into
# per-error sections plus the final error count summary. No import-time side effects.
# this file contains a simple parser that parses report
# from cuda-memcheck


class ParseError(Exception):
    # [CLASS DOC] Exception raised by this module when the report text cannot be interpreted: a
    # missing error summary, or a summary count that disagrees with the number of parsed errors.
    """Whenever the simple parser is unable to parse the report, this exception will be raised"""


class Report:
    # [CLASS DOC] Parsed result of a whole report: holds the summary text, the integer error count
    # read from it, and the list of Error objects. Invariant: the count equals the length of the
    # list, except that a count above 10000 is clamped to 10000 when exactly 10000 errors were
    # listed.
    """A report is a container of errors, and a summary on how many errors are found"""

    def __init__(self, text, errors):
        # [FUNCTION DOC] Stores the summary line and error list, and reads the error count from
        # the third whitespace separated token of the summary line (a non-integer there raises
        # ValueError). Raises ParseError when the count differs from the number of errors, unless
        # exactly 10000 errors were listed and the summary claims more, in which case the count is
        # clamped to 10000 (cuda-memcheck prints at most that many).
        # text is something like
        # ERROR SUMMARY: 1 error
        # or
        # ERROR SUMMARY: 2 errors
        self.text = text
        self.num_errors = int(text.strip().split()[2])
        self.errors = errors
        if len(errors) != self.num_errors:
            if len(errors) == 10000 and self.num_errors > 10000:
                # When there are more than 10k errors, cuda-memcheck only display 10k
                self.num_errors = 10000
            else:
                raise ParseError("Number of errors does not match")


class Error:
    # [CLASS DOC] One error section of the report: attributes are the message (first line) and the
    # stack (list of stripped backtrace lines). Created from the list of lines of the section by
    # the parser.
    """Each error is a section in the output of cuda-memcheck.
    Each error in the report has an error message and a backtrace. It looks like:

    ========= Program hit cudaErrorInvalidValue (error 1) due to "invalid argument" on CUDA API call to cudaGetLastError.
    =========     Saved host backtrace up to driver entry point at error
    =========     Host Frame:/usr/lib/x86_64-linux-gnu/libcuda.so.1 [0x38c7b3]
    =========     Host Frame:/usr/local/cuda/lib64/libcudart.so.10.1 (cudaGetLastError + 0x163) [0x4c493]
    =========     Host Frame:/home/xgao/anaconda3/lib/python3.7/site-packages/torch/lib/libtorch.so [0x5b77a05]
    =========     Host Frame:/home/xgao/anaconda3/lib/python3.7/site-packages/torch/lib/libtorch.so [0x39d6d1d]
    =========     .....
    """

    def __init__(self, lines):
        # [FUNCTION DOC] Takes the section lines with the common prefix already removed; the first
        # line becomes the message, the second line (the saved-backtrace header) is dropped, and
        # the remaining lines are stripped of surrounding whitespace and stored as the stack.
        # Assumes at least one line is given; an empty list raises IndexError.
        self.message = lines[0]
        lines = lines[2:]
        self.stack = [l.strip() for l in lines]


def parse(message):
    # [FUNCTION DOC] Parses a cuda-memcheck report given as one string. Ignores everything until
    # the line announcing the tool, then considers only lines starting with the equals-sign
    # prefix. The first prefixed line after a blank prefixed line starts a new error, further
    # lines are appended to it, and a blank prefixed line closes it. The line starting with the
    # summary label ends parsing and a Report is returned. Raises ParseError if the text has no
    # summary line (or the report is otherwise inconsistent, via Report).
    """A simple parser that parses the report of cuda-memcheck. This parser is meant to be simple
    and it only split the report into separate errors and a summary. Where each error is further
    split into error message and backtrace. No further details are parsed.

    A report contains multiple errors and a summary on how many errors are detected. It looks like:

    ========= CUDA-MEMCHECK
    ========= Program hit cudaErrorInvalidValue (error 1) due to "invalid argument" on CUDA API call to cudaPointerGetAttributes.
    =========     Saved host backtrace up to driver entry point at error
    =========     Host Frame:/usr/lib/x86_64-linux-gnu/libcuda.so.1 [0x38c7b3]
    =========     Host Frame:/usr/local/cuda/lib64/libcudart.so.10.1 (cudaPointerGetAttributes + 0x1a9) [0x428b9]
    =========     Host Frame:/home/xgao/anaconda3/lib/python3.7/site-packages/torch/lib/libtorch.so [0x5b778a9]
    =========     .....
    =========
    ========= Program hit cudaErrorInvalidValue (error 1) due to "invalid argument" on CUDA API call to cudaGetLastError.
    =========     Saved host backtrace up to driver entry point at error
    =========     Host Frame:/usr/lib/x86_64-linux-gnu/libcuda.so.1 [0x38c7b3]
    =========     Host Frame:/usr/local/cuda/lib64/libcudart.so.10.1 (cudaGetLastError + 0x163) [0x4c493]
    =========     .....
    =========
    ========= .....
    =========
    ========= Program hit cudaErrorInvalidValue (error 1) due to "invalid argument" on CUDA API call to cudaGetLastError.
    =========     Saved host backtrace up to driver entry point at error
    =========     Host Frame:/usr/lib/x86_64-linux-gnu/libcuda.so.1 [0x38c7b3]
    =========     .....
    =========     Host Frame:python (_PyEval_EvalFrameDefault + 0x6a0) [0x1d0ad0]
    =========     Host Frame:python (_PyEval_EvalCodeWithName + 0xbb9) [0x116db9]
    =========
    ========= ERROR SUMMARY: 4 errors
    """
    errors = []
    HEAD = "========="
    headlen = len(HEAD)
    started = False
    in_message = False
    message_lines = []
    lines = message.splitlines()
    for l in lines:
        if l == HEAD + " CUDA-MEMCHECK":
            started = True
            continue
        if not started or not l.startswith(HEAD):
            continue
        l = l[headlen + 1 :]
        if l.startswith("ERROR SUMMARY:"):
            return Report(l, errors)
        if not in_message:
            in_message = True
            message_lines = [l]
        elif l == "":
            errors.append(Error(message_lines))
            in_message = False
        else:
            message_lines.append(l)
    raise ParseError("No error summary found")
