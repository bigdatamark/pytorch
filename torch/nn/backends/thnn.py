# mypy: allow-untyped-defs
# this is for historical pickle deserialization, it is not used otherwise
# [MODULE DOC] Legacy stub module kept only so old pickled models that reference it can still
# be loaded. Defines one no-op function; no import-time side effects.


def _get_thnn_function_backend() -> None:
    # [FUNCTION DOC] HELPER: No-op placeholder for historical pickle compatibility; returns None.
    pass
