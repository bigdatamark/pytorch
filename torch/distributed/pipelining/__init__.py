# Copyright (c) Meta Platforms, Inc. and affiliates
# [PACKAGE DOC] Pipeline parallelism package. Re-exports the graph-splitting API
# (Pipe, pipe_split, SplitPoint, pipeline), the stage classes (PipelineStage,
# build_stage) and the schedule classes (GPipe, 1F1B, interleaved 1F1B, looped BFS,
# zero-bubble variants, DualPipeV, listed in __all__). Importing has no side effects
# beyond importing the submodules.
from ._IR import Pipe, pipe_split, pipeline, SplitPoint
from .schedules import (
    _ScheduleForwardOnly,
    Schedule1F1B,
    ScheduleDualPipeV,
    ScheduleGPipe,
    ScheduleInterleaved1F1B,
    ScheduleInterleavedZeroBubble,
    ScheduleLoopedBFS,
    ScheduleZBVZeroBubble,
)
from .stage import build_stage, PipelineStage


__all__ = [
    "Pipe",
    "pipe_split",
    "SplitPoint",
    "pipeline",
    "PipelineStage",
    "build_stage",
    "Schedule1F1B",
    "ScheduleGPipe",
    "ScheduleInterleaved1F1B",
    "ScheduleLoopedBFS",
    "ScheduleInterleavedZeroBubble",
    "ScheduleZBVZeroBubble",
    "ScheduleDualPipeV",
]
