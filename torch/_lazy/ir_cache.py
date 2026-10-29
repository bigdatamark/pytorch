# [MODULE DOC] Wrappers for dumping and clearing the lazy IR TrieCache.
import torch._C._lazy


def dump(dot_file_name: str) -> None:
    # [FUNCTION DOC] Writes the IR TrieCache in dot format to the given file name.
    """Dump TrieCache in the dot format"""
    return torch._C._lazy._dump_ir_cache(dot_file_name)


def reset() -> None:
    # [FUNCTION DOC] Clears the IR TrieCache (used in tests to avoid node reuse across tests).
    """Clear TrieCache. This is needed in testing to avoid
    node reusing between different tests.
    """
    return torch._C._lazy._clear_ir_cache()
