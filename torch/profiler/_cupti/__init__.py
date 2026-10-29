# [PACKAGE DOC] HELPER: Placeholder package for in-process CUPTI activity collection; it only
# carries a docstring and exports nothing yet.
"""In-process CUPTI activity collection for torch.profiler.

``monitor`` owns the GIL-free CUPTI activity buffer collection (backed by the
native ``CuptiMonitorBuffers``); ``monitor_trace`` builds Chrome traces from it.
(The multiplexer + observers land here next.)
"""
