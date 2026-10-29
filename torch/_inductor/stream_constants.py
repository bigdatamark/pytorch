# [MODULE DOC] Constants for naming CUDA streams in Inductor codegen: DEFAULT_STREAM (name of the
# default stream), DEFAULT_STREAM_IDX (its index, 0) and STREAM_NAME_TEMPLATE (format string that
# turns an integer stream index into a stream variable name).
"""Constants for CUDA stream naming in Inductor codegen."""

DEFAULT_STREAM: str = "default_stream"
DEFAULT_STREAM_IDX: int = 0
STREAM_NAME_TEMPLATE: str = "stream{stream_idx:d}"
