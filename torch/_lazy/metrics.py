# [MODULE DOC] Wrappers over lazy runtime metrics: reset, counter names/values, and the combined report.
import torch._C._lazy


def reset() -> None:
    # [FUNCTION DOC] Resets all metric counters.
    """Resets all metric counters."""
    torch._C._lazy._reset_metrics()


def counter_names() -> list[str]:
    # [FUNCTION DOC] Returns the names of all active counters.
    """Retrieves all the currently active counter names."""
    return torch._C._lazy._counter_names()


def counter_value(name: str) -> int:
    # [FUNCTION DOC] Returns the integer value of the named counter.
    """Return the value of the counter with the specified name"""
    return torch._C._lazy._counter_value(name)


def metrics_report() -> str:
    # [FUNCTION DOC] Returns the combined lazy-core and backend metrics report as text.
    """Return the combined (lazy core and backend) metric report"""
    return torch._C._lazy._metrics_report()
