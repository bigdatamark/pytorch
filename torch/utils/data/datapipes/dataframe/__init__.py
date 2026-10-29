# [PACKAGE DOC] DataFrame support for datapipes: lazily captured DataFrame operations
# (CaptureDataFrame), the DFIterDataPipe base type, and DataFramesAsTuplesPipe which
# flattens DataFrame batches into row tuples. Exports these three names via __all__ and
# raises an error at import time if __all__ is not sorted.
from torch.utils.data.datapipes.dataframe.dataframes import (
    CaptureDataFrame,
    DFIterDataPipe,
)
from torch.utils.data.datapipes.dataframe.datapipes import DataFramesAsTuplesPipe


__all__ = ["CaptureDataFrame", "DFIterDataPipe", "DataFramesAsTuplesPipe"]

# Please keep this list sorted
if __all__ != sorted(__all__):
    raise AssertionError("__all__ is not sorted")
