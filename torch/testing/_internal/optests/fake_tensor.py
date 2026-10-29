# mypy: ignore-errors
# [MODULE DOC] Fake-tensor consistency check used by opcheck: run an operator under
# CrossRefFakeMode. Contains is_builtin and fake_check. No import-time side effects.

import torch._subclasses


def is_builtin(op):
    # [FUNCTION DOC] HELPER: Returns True if the operator's namespace is aten, prims or prim.
    return op.namespace in ('aten', 'prims', 'prim')


def fake_check(op, args, kwargs):
    # [FUNCTION DOC] Runs the operator once on the given args and kwargs under a cross-ref
    # fake tensor mode, which compares real and fake execution results. Built-in
    # operators (see is_builtin) are skipped by the comparison. Returns nothing; errors
    # raised by the mode (mismatch between real and fake) or by the op propagate.
    with torch._subclasses.CrossRefFakeMode(ignore_op_fn=is_builtin):
        op(*args, **kwargs)
