# mypy: allow-untyped-defs
# [MODULE DOC] Small tensor-shape helpers for autograd functions: maybe_view and maybe_unexpand.


def maybe_view(tensor, size, check_same_size=True):
    # [FUNCTION DOC] Return the tensor itself if check_same_size is set and its size already equals size,
    # otherwise return a contiguous view of the tensor with the given size.
    if check_same_size and tensor.size() == size:
        return tensor
    return tensor.contiguous().view(size)


def maybe_unexpand(tensor, old_size, check_same_size=True):
    # [FUNCTION DOC] Reduce a broadcast-expanded tensor (typically a gradient) back to old_size.
    # Returns it unchanged if sizes already match (and check_same_size is set). Otherwise sums away the
    # leading extra dimensions, then sums with kept dims over every dimension whose size differs from old_size.
    if check_same_size and tensor.size() == old_size:
        return tensor
    num_unsqueezed = tensor.dim() - len(old_size)
    expanded_dims = [
        dim
        for dim, (expanded, original) in enumerate(
            zip(tensor.size()[num_unsqueezed:], old_size)
        )
        if expanded != original
    ]

    for _ in range(num_unsqueezed):
        tensor = tensor.sum(0, keepdim=False)
    for dim in expanded_dims:
        tensor = tensor.sum(dim, keepdim=True)
    return tensor
