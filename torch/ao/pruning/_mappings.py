# mypy: allow-untyped-defs
# [MODULE DOC] Provides lazily built mappings from dense module types to their sparse
# quantized (static and dynamic) counterparts. Exports two getter functions; the
# sparse module package is imported only when a getter runs.
__all__ = [
    "get_static_sparse_quantized_mapping",
    "get_dynamic_sparse_quantized_mapping",
]


def get_static_sparse_quantized_mapping():
    # [FUNCTION DOC] Imports the sparse module package and returns a new mapping from the
    # Linear type to the static sparse quantized Linear type.
    import torch.ao.nn.sparse

    _static_sparse_quantized_mapping = {
        torch.nn.Linear: torch.ao.nn.sparse.quantized.Linear,
    }
    return _static_sparse_quantized_mapping


def get_dynamic_sparse_quantized_mapping():
    # [FUNCTION DOC] Imports the sparse module package and returns a new mapping from the
    # Linear type to the dynamic sparse quantized Linear type.
    import torch.ao.nn.sparse

    _dynamic_sparse_quantized_mapping = {
        torch.nn.Linear: torch.ao.nn.sparse.quantized.dynamic.Linear,
    }
    return _dynamic_sparse_quantized_mapping
