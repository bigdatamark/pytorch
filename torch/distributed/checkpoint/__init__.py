# [PACKAGE DOC]
# Distributed checkpoint (DCP) package: saving and loading sharded state dicts across ranks.
# Re-exports the save/load entry points (save, load, async_save and deprecated
# save_state_dict, load_state_dict), planners (default and base planner types, plans, items),
# storage readers and writers (file system, HuggingFace, quantized HuggingFace), metadata
# types, the optimizer state loading helper, CheckpointException and CheckpointableTensor.
from . import _extension
from .api import CheckpointException
from .default_planner import DefaultLoadPlanner, DefaultSavePlanner
from .filesystem import FileSystemReader, FileSystemWriter
from .hf_storage import HuggingFaceStorageReader, HuggingFaceStorageWriter
from .metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    Metadata,
    TensorStorageMetadata,
)
from .optimizer import load_sharded_optimizer_state_dict
from .planner import LoadPlan, LoadPlanner, ReadItem, SavePlan, SavePlanner, WriteItem
from .protocol import CheckpointableTensor
from .quantized_hf_storage import QuantizedHuggingFaceStorageReader

# pyrefly: ignore [deprecated]
from .state_dict_loader import load, load_state_dict

# pyrefly: ignore [deprecated]
from .state_dict_saver import async_save, save, save_state_dict
from .storage import StorageReader, StorageWriter
