# mypy: ignore-errors
# [PACKAGE DOC] Aggregates the per-area OpInfo definition modules into the global databases.
# Exports op_db (all OpInfo entries from fft, linalg, signal, special and _masked) and
# python_ref_db (Python reference OpInfos from fft, linalg and special) as plain lists
# built at import time by concatenation. Importing this package imports those submodules,
# which constructs every OpInfo they define.

from torch.testing._internal.opinfo.core import OpInfo
from torch.testing._internal.opinfo.definitions import (
    _masked,
    fft,
    linalg,
    signal,
    special,
)


# Operator database
op_db: list[OpInfo] = [
    *fft.op_db,
    *linalg.op_db,
    *signal.op_db,
    *special.op_db,
    *_masked.op_db,
]

python_ref_db: list[OpInfo] = [
    *fft.python_ref_db,
    *linalg.python_ref_db,
    *special.python_ref_db,
]
