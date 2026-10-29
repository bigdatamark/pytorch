# [PACKAGE DOC]
# Map-style DataPipes (index to item). Re-exports Batcher, Concater, Mapper,
# SequenceWrapper, Shuffler and Zipper under short names via __all__. Importing it also
# registers their functional names. Raises AssertionError at import time if __all__ is not
# sorted.
# Functional DataPipe
from torch.utils.data.datapipes.map.callable import MapperMapDataPipe as Mapper
from torch.utils.data.datapipes.map.combinatorics import (
    ShufflerIterDataPipe as Shuffler,
)
from torch.utils.data.datapipes.map.combining import (
    ConcaterMapDataPipe as Concater,
    ZipperMapDataPipe as Zipper,
)
from torch.utils.data.datapipes.map.grouping import BatcherMapDataPipe as Batcher
from torch.utils.data.datapipes.map.utils import (
    SequenceWrapperMapDataPipe as SequenceWrapper,
)


__all__ = ["Batcher", "Concater", "Mapper", "SequenceWrapper", "Shuffler", "Zipper"]

# Please keep this list sorted
if __all__ != sorted(__all__):
    raise AssertionError("__all__ is not sorted")
