# mypy: allow-untyped-defs
# [MODULE DOC] Thin Python wrapper exposing the C++ stack trace of the current thread.
# Contains one function, get_cpp_backtrace. Import has no side effects beyond importing
# the native helper from torch._C.
from torch._C import _get_cpp_backtrace

def get_cpp_backtrace(frames_to_skip=0, maximum_number_of_frames=64) -> str:
    # [FUNCTION DOC] Returns the C++ stack trace of the calling thread as a string.
    # Args: frames_to_skip, number of top frames to drop; maximum_number_of_frames, cap on
    # frames returned. Delegates entirely to the native _get_cpp_backtrace; no side effects.
    r"""
    Return a string containing the C++ stack trace of the current thread.

    Args:
        frames_to_skip (int): the number of frames to skip from the top of the stack
        maximum_number_of_frames (int): the maximum number of frames to return
    """
    return _get_cpp_backtrace(frames_to_skip, maximum_number_of_frames)
