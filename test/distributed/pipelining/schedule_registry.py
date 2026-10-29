# [MODULE DOC] Hand-written schedules for testing torch.distributed.pipelining. Defines
# shorthand aliases F, B, W, I for forward, full backward, backward-weight and backward-input
# computation types, and four schedule classes with fixed per-rank action orders.
# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
# This file is a Schedule zoo for testing torch.distributed.pipelining.
# It includes schedules designed purely for testing purposes
from collections.abc import Callable

from torch.distributed.pipelining.schedules import (
    _Action,
    _ComputationType,
    _PipelineScheduleRuntime,
    PipelineScheduleMulti,
    RECV_B,
    RECV_F,
    SEND_B,
    SEND_F,
)
from torch.distributed.pipelining.stage import _PipelineStageBase


F = _ComputationType.FORWARD
B = _ComputationType.FULL_BACKWARD
W = _ComputationType.BACKWARD_WEIGHT
I = _ComputationType.BACKWARD_INPUT


class ScheduleVShaped(PipelineScheduleMulti):
    # [CLASS DOC] Test schedule with 4 stages on 2 ranks in a V shape (rank 0 owns stages 0
    # and 3, rank 1 owns 1 and 2), one microbatch. pipeline_order gives each rank a list of
    # actions, with None meaning an idle slot.
    n_stages = 4
    rank_stages = {
        0: [0, 3],
        1: [1, 2],
    }

    def __init__(
        self,
        stages: list[_PipelineStageBase],
        n_microbatches: int,
        loss_fn: Callable | None = None,
        scale_grads: bool = True,
    ):
        # [FUNCTION DOC] Calls the multi-stage schedule constructor with the stages,
        # microbatch count, loss function and grad scaling, then sets the fixed
        # pipeline_order for the V layout and validates it and sets the stage mapping.
        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            scale_grads=scale_grads,
        )

        # Go through one microbatch
        # Note(whc) - it might be easier to work with this schedules by writing them as a list of
        # ["0F0", ...] and then parsing them in the test infra to turn them into actions.
        self.pipeline_order = {
            0: [
                _Action(0, F, 0),
                None,
                None,
                _Action(3, F, 0),
                _Action(3, B, 0),
                None,
                None,
                _Action(0, B, 0),
            ],
            1: [
                None,
                _Action(1, F, 0),
                _Action(2, F, 0),
                None,
                None,
                _Action(2, B, 0),
                _Action(1, B, 0),
                None,
            ],
        }
        self._validate_and_set_stage_mapping(self.pipeline_order)


class ScheduleUnbalanced(PipelineScheduleMulti):
    # [CLASS DOC] Test schedule with 5 stages over 2 ranks (rank 0: stages 0, 1, 4; rank 1:
    # stages 2, 3), a single microbatch, with a fixed pipeline_order containing idle slots.
    n_stages = 5
    rank_stages = {
        0: [0, 1, 4],
        1: [2, 3],
    }

    def __init__(
        self,
        stages: list[_PipelineStageBase],
        n_microbatches: int,
        loss_fn: Callable | None = None,
        scale_grads: bool = True,
    ):
        # [FUNCTION DOC] Calls the multi-stage schedule constructor, then sets the fixed
        # unbalanced pipeline_order and validates it and sets the stage mapping.
        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            scale_grads=scale_grads,
        )

        self.pipeline_order = {
            0: [
                _Action(0, F, 0),
                _Action(1, F, 0),
                None,
                None,
                _Action(4, F, 0),
                _Action(4, B, 0),
                None,
                None,
                _Action(1, B, 0),
                _Action(0, B, 0),
            ],
            1: [
                None,
                None,
                _Action(2, F, 0),
                _Action(3, F, 0),
                None,
                None,
                _Action(3, B, 0),
                _Action(2, B, 0),
                None,
                None,
            ],
        }
        self._validate_and_set_stage_mapping(self.pipeline_order)


class ScheduleWithW(PipelineScheduleMulti):
    # [CLASS DOC] Test schedule with 4 stages over 2 ranks (rank 0: 0, 2; rank 1: 1, 3) and
    # two microbatches that splits backward into input (I) and weight (W) actions, as in
    # zero-bubble schedules.
    n_stages = 4
    num_microbatches = 2
    rank_stages = {
        0: [0, 2],
        1: [1, 3],
    }

    def __init__(
        self,
        stages: list[_PipelineStageBase],
        n_microbatches: int,
        loss_fn: Callable | None = None,
        enable_zero_bubble: bool = True,
        scale_grads: bool = True,
    ):
        # [FUNCTION DOC] Calls the multi-stage schedule constructor, sets use_full_backward
        # to False (required for schedules using W), then sets the two-microbatch
        # pipeline_order and validates it and sets the stage mapping.
        # KNOWN ISSUE: the enable_zero_bubble argument is accepted but never used.
        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            scale_grads=scale_grads,
        )

        # Needs to be updated as part of all schedules using "W"
        self.use_full_backward = False

        # Go through two microbatches
        self.pipeline_order = {
            0: [
                _Action(0, F, 0),
                _Action(0, F, 1),
                _Action(2, F, 0),
                _Action(2, F, 1),
                None,
                _Action(2, I, 0),
                _Action(2, W, 0),
                _Action(0, I, 0),
                _Action(2, I, 1),
                _Action(0, W, 0),
                _Action(0, I, 1),
                _Action(2, W, 1),
                _Action(0, W, 1),
            ],
            1: [
                None,
                _Action(1, F, 0),
                _Action(1, F, 1),
                _Action(3, F, 0),
                _Action(3, I, 0),
                _Action(3, F, 1),
                _Action(1, I, 0),
                _Action(3, I, 1),
                _Action(3, W, 0),
                _Action(1, I, 1),
                _Action(1, W, 0),
                _Action(3, W, 1),
                _Action(1, W, 1),
            ],
        }
        self._validate_and_set_stage_mapping(self.pipeline_order)


class ScheduleWithReorderedB(_PipelineScheduleRuntime):
    # [CLASS DOC] Test schedule on the runtime schedule base class with 2 stages (one per
    # rank) and two microbatches, defined by explicit communication actions in
    # pipeline_order_with_comms; the backward receives are issued before the backward compute.
    n_stages = 2
    num_microbatches = 2
    rank_stages = {
        0: [0],
        1: [1],
    }

    def __init__(
        self,
        stages: list[_PipelineStageBase],
        n_microbatches: int,
        loss_fn: Callable | None = None,
        scale_grads: bool = True,
    ):
        # [FUNCTION DOC] Calls the runtime schedule constructor, then sets
        # pipeline_order_with_comms: per rank, lists of forward, send/recv and backward
        # actions for both microbatches. No validation is called here.
        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            scale_grads=scale_grads,
        )
        # Go through two microbatches
        self.pipeline_order_with_comms = {
            0: [
                _Action(0, F, 0),
                _Action(0, F, 1),
                _Action(0, SEND_F, 0),
                _Action(0, SEND_F, 1),
                _Action(0, RECV_B, 0),
                _Action(0, RECV_B, 1),
                _Action(0, B, 0),
                _Action(0, B, 1),
            ],
            1: [
                _Action(1, RECV_F, 0),
                _Action(1, RECV_F, 1),
                _Action(1, F, 0),
                _Action(1, F, 1),
                _Action(1, B, 0),
                _Action(1, B, 1),
                _Action(1, SEND_B, 0),
                _Action(1, SEND_B, 1),
            ],
        }
