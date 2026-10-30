# [MODULE DOC] Reveal-type sample for named-tuple return types of tensor ops. It sorts a small
# 2x2 tensor and indexes the result both by position and by the indices and values field names,
# expecting the revealed type torch.return_types.sort; then it runs the linalg QR decomposition,
# uses position and the Q field, and reveals Any (a TODO in the file says the precise tuple type
# is not yet inferred). The comparisons are bare expressions with noqa markers; the expected
# types are in the marker comments.
import torch


t = torch.tensor([[3.0, 1.5], [2.0, 1.5]])

t_sort = t.sort()
t_sort[0][0, 0] == 1.5  # noqa: B015
t_sort.indices[0, 0] == 1  # noqa: B015
t_sort.values[0, 0] == 1.5  # noqa: B015
reveal_type(t_sort)  # E: torch.return_types.sort

t_qr = torch.linalg.qr(t)
t_qr[0].shape == [2, 2]  # noqa: B015
t_qr.Q.shape == [2, 2]  # noqa: B015
# TODO: Fixme, should be Tuple[{Tensor}, {Tensor}, fallback=torch.return_types.linalg_qr]
reveal_type(t_qr)  # E: Any
