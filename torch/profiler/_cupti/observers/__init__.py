# [PACKAGE DOC] Package of observers (consumers) for the in-process CUPTI activity monitor.
# Contents: base.py (CuptiMonitorObserver base class, ObserverAnnotationSettings and the graph
# resolver types), node_timer.py (NodeTimerObserver buffering per-node kernel/copy/set spans),
# observation_window.py (WindowFinalizerMixin for boundary-based window finalization).
# This __init__ defines no names and re-exports nothing; import the submodules directly.
# No import-time side effects.
"""Observers for the in-process CUPTI activity monitor.

Each observer registers the activity kinds it needs with the shared monitor
(``torch.profiler._cupti.monitor.CuptiMonitor()``) and consumes the decoded columns
delivered to it, aggregating or assembling whatever it exposes to callers.
``CuptiMonitorObserver`` is the shared base.
"""
