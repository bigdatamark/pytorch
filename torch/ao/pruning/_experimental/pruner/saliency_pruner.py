# mypy: allow-untyped-defs
# [MODULE DOC] Defines SaliencyPruner, a structured pruner that removes rows with the
# smallest L1 norm. No import-time side effects.
from .base_structured_sparsifier import BaseStructuredSparsifier


class SaliencyPruner(BaseStructuredSparsifier):
    # [CLASS DOC] Structured pruner scoring each row (output channel) of an N-dimensional
    # weight by its L1 norm and masking the lowest-scoring fraction given by sparsity_level.
    """
    Prune rows based on the saliency (L1 norm) of each row.

    This pruner works on N-Dimensional weight tensors.
    For each row, we will calculate the saliency, which is the sum of the L1 norm of all weights in that row.
    We expect that the resulting saliency vector has the same shape as our mask.
    We then pick elements to remove until we reach the target sparsity_level.
    """

    def update_mask(self, module, tensor_name, **kwargs):
        # [FUNCTION DOC] Computes the L1 norm of each row of module.tensor_name, then sets to
        # False (in place) the mask entries of the int(mask length times kwargs
        # sparsity_level) rows with the smallest norm. Raises a generic Exception for
        # weights with fewer than two dimensions and AssertionError if the saliency
        # shape differs from the mask shape.
        # tensor_name will give you the FQN, all other entries in sparse config is present in kwargs
        weights = getattr(module, tensor_name)
        mask = getattr(module.parametrizations, tensor_name)[0].mask

        # use negative weights so we can use topk (we prune out the smallest)
        if weights.dim() <= 1:
            raise Exception(  # noqa: TRY002
                "Structured pruning can only be applied to a 2+dim weight tensor!"
            )
        saliency = -weights.norm(dim=tuple(range(1, weights.dim())), p=1)
        if saliency.shape != mask.shape:
            raise AssertionError(
                f"saliency shape ({saliency.shape}) must match mask shape ({mask.shape})"
            )

        num_to_pick = int(len(mask) * kwargs["sparsity_level"])
        prune = saliency.topk(num_to_pick).indices

        # Set the mask to be false for the rows we want to prune
        mask.data[prune] = False
