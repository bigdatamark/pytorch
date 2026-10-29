# mypy: allow-untyped-defs
# [MODULE DOC] Defines ModelReportObserver, an observer module used by the FX model report tooling.
# It records activation range, per-channel min/max and percentile-ratio statistics so that
# detectors can recommend dynamic vs static quantization and spot outliers. No import-time side effects.
import torch
from torch.ao.quantization.observer import ObserverBase


class ModelReportObserver(ObserverBase):
    # [CLASS DOC] Observer (subclass of ObserverBase, qint8 dtype) that passes data through unchanged while
    # accumulating statistics. Plain attribute: num_batches_tracked and average_batch_activation_range (running mean of per-batch
    # max-min). Buffers: epoch_activation_min/max (global range), min_val/max_val (per channel, along ch_axis),
    # comp_percentile, average_percentile_ratio, percentile_batches_tracked, constant_channels (all per channel,
    # empty until first batch). Key methods: forward, get_batch_to_epoch_ratio, reset_batch_and_epoch_values.
    r"""This observer is used to record additional information regarding keeping track
    of S = average_batch_activation_range/epoch_activation_range.

    The purpose of this information is to prepare a report to present to users on whether
    Dynamic or Static Quantization is more appropriate for their model given the general
    distributions of their data.

    Args:
        ch_axis (int, optional): The channel axis for which the range and outlier stats are computed
            Default: 1
        comp_percentile (float, optional): The percentile to compare against 100 percentile to find outliers
            Should be between 0 and 1 exclusive
            Default: 0.9

    * :attr:`num_batches_tracked` specifies number of batches passed through the observer

    * :attr:`average_batch_activation_range` defines average across the ranges of each batch passed through

    * :attr:`epoch_activation_min` defines the minimum value passed through the observer

    * :attr:`epoch_activation_max` defines the maximum value passed through the observer

    * :attr:`ch_axis` defines the channel being used to compute per channel min max stats

    * :attr:`min_val` defines the per channel minimum values passed through

    * :attr:`max_val` defines the per channel maximum values passed through

    * :attr:`comp_percentile` defines comparison percentile to find outliers

    * :attr:`average_percentile_ratio` defines the per channel average percentile ratios

    * :attr:`percentile_batches_tracked` defines the number of percentile batches tracked for each channel

    * :attr:`constant_channels` defines the number of batches that aren't constant channels per channel

    Note: this tool is meant for FX Graph Mode Quantization
    """

    epoch_activation_min: torch.Tensor
    epoch_activation_max: torch.Tensor
    min_val: torch.Tensor
    max_val: torch.Tensor
    comp_percentile: torch.Tensor
    average_percentile_ratio: torch.Tensor
    percentile_batches_tracked: torch.Tensor
    constant_channels: torch.Tensor

    def __init__(self, ch_axis: int = 1, comp_percentile: float = 0.9):
        # [FUNCTION DOC] Initializes the observer with the channel axis (ch_axis) and the comparison percentile
        # (comp_percentile, expected to be strictly between 0 and 1; not validated here). Registers the statistic buffers:
        # epoch min is +inf, epoch max is -inf, per-channel buffers are empty tensors.
        super().__init__(torch.qint8)
        self.num_batches_tracked = 0

        # keep track of the min and mix of the range for average batch and epoch as a whole
        self.average_batch_activation_range: torch.Tensor = torch.tensor(float(0))
        self.register_buffer("epoch_activation_min", torch.tensor(float("inf")))
        self.register_buffer("epoch_activation_max", torch.tensor(float("-inf")))

        # keep track of per channel min max information using the given channel
        self.ch_axis: int = ch_axis
        self.register_buffer("min_val", torch.tensor([]))
        self.register_buffer("max_val", torch.tensor([]))

        # keep track of percentile ratio information per channel
        self.register_buffer("comp_percentile", torch.tensor([comp_percentile]))
        self.register_buffer("average_percentile_ratio", torch.tensor([]))
        self.register_buffer("percentile_batches_tracked", torch.tensor([]))
        self.register_buffer("constant_channels", torch.tensor([]))

    def forward(self, x):
        # [FUNCTION DOC] Updates all statistics from the input tensor (detached, cast to the epoch buffer dtype)
        # by calling the range, min/max and percentile helpers in order, and returns the original input unchanged.
        # Side effects: mutates the observer's buffers and counters.
        x_copy = x.detach()  # avoid keeping autograd tape
        x_copy = x_copy.to(self.epoch_activation_min.dtype)

        x_copy = self._calculate_range_stats(x_copy)
        x_copy = self._calculate_min_max_stats(x_copy)
        x_copy = self._calculate_percentile_stats(x_copy)

        # return the passed in the value
        return x

    def _calculate_range_stats(self, x_copy):
        # [FUNCTION DOC] HELPER: Updates the global epoch min/max buffers and the running average of per-batch range
        # (max minus min) using the input, increments num_batches_tracked, and returns the input tensor.
        r"""Calculates and stores range stats with forward values.

        Args
            x_copy: A copy of the forward data

        Returns the passed in x_copy
        """
        # get the min, max values of the data
        min_val_cur, max_val_cur = torch.aminmax(x_copy)

        # calculate new epoch range values
        epoch_min_val = torch.min(self.epoch_activation_min, min_val_cur)
        epoch_max_val = torch.max(self.epoch_activation_max, max_val_cur)

        self.epoch_activation_min.copy_(epoch_min_val)
        self.epoch_activation_max.copy_(epoch_max_val)

        # calculate the average batch activation range
        current_batch_range = max_val_cur - min_val_cur
        new_range = (
            self.average_batch_activation_range * self.num_batches_tracked
            + current_batch_range
        ) / (self.num_batches_tracked + 1)

        self.average_batch_activation_range = new_range
        self.num_batches_tracked += 1  # new batch was processed

        return x_copy

    def _calculate_min_max_stats(self, x_copy):
        # [FUNCTION DOC] HELPER: Updates per-channel min_val/max_val buffers along ch_axis. Permutes the channel axis to the front,
        # flattens the rest, and takes min/max per channel (combined with the stored values if already
        # initialized). Resizes and overwrites the buffers in place; returns the input tensor.
        r"""Calculates and stores the per_channel min, max stats with forward values.
        Does calculation based on channel axis: self.ch_axis

        Args
            x_copy: A copy of the forward data

        Returns the passed in x_copy
        """
        # get the current min and max vals
        min_val = self.min_val
        max_val = self.max_val
        x_dim = x_copy.size()

        new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
        new_axis_list[self.ch_axis] = 0
        new_axis_list[0] = self.ch_axis
        y = x_copy.permute(new_axis_list)
        # Need to match dtype of min/max because the updates to buffers
        # are done in place and types need to match for comparisons
        y = y.to(self.min_val.dtype)
        y = torch.flatten(y, start_dim=1)
        if min_val.numel() == 0 or max_val.numel() == 0:
            min_val, max_val = torch.aminmax(y, dim=1)
        else:
            min_val_cur, max_val_cur = torch.aminmax(y, dim=1)
            min_val = torch.min(min_val_cur, min_val)
            max_val = torch.max(max_val_cur, max_val)

        self.min_val.resize_(min_val.shape)
        self.max_val.resize_(max_val.shape)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)

        return x_copy

    def _calculate_percentile_stats(self, x_copy):
        # [FUNCTION DOC] HELPER: Updates per-channel percentile statistics. Computes the 0th, comp_percentile and 100th quantiles
        # (lower interpolation, on CPU) per channel, then maintains a running average of the ratio of 100th to
        # comp percentile quantile, counting only batches where a channel is not all zeros, and counts batches where the channel was
        # constant. Lazily initializes the percentile buffers on first call. Returns the input tensor.
        # KNOWN ISSUE: the quantile call uses dim=ch_axis after the channel axis was moved to the front and flattened, so for
        # a ch_axis other than 0 or 1 the reduced dimension may not be the intended one.
        r"""Calculates and stores the per_channel percentile stats with forward values.
        Does calculation based on channel axis: self.ch_axis

        Args
            x_copy: A copy of the forward data

        Returns the passed in x_copy
        """
        # get the dimension of the copy
        x_dim = x_copy.size()

        new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
        new_axis_list[self.ch_axis] = 0
        new_axis_list[0] = self.ch_axis
        y = x_copy.permute(new_axis_list)
        # Need to match dtype of min/max because the updates to buffers
        # are done in place and types need to match for comparisons
        y = y.to(self.min_val.dtype)
        y = torch.flatten(y, start_dim=1)
        y = y.to(dtype=self.min_val.dtype, device="cpu")

        # find the percentile values along the axis
        # we want both 100th percentile and comp_percentile
        # we also want to find 0th quartile to see if we have constant channel
        quantiles_list = [0, self.comp_percentile, 1.00]
        quantiles_to_find = torch.tensor(quantiles_list, dtype=self.min_val.dtype)

        # find the quantiles
        desired_quantiles = torch.quantile(
            y, quantiles_to_find, dim=self.ch_axis, interpolation="lower"
        )
        zero_quantile = desired_quantiles[0]
        comp_quantile = desired_quantiles[1]
        hundreth_quartile = desired_quantiles[2]

        # if any of the channels have 0s, we ignore that channel for this calculation
        any_non_zero_quantile_value: torch.Tensor = (
            comp_quantile != torch.tensor([0])
        ) | (hundreth_quartile != torch.tensor([0]))
        any_non_zero_quantile_value = (
            any_non_zero_quantile_value.int()
        )  # transform boolean values to int values

        # we also check if we have a constant channel
        any_constant_channels: torch.Tensor = (
            hundreth_quartile - zero_quantile
        ) == torch.tensor([0])
        any_constant_channels = (
            any_constant_channels.int()
        )  # transform boolean values to int values

        # possibilities to get nan as an answer
        #   will ignore any of these three cases with 0s and just not deal with them for now
        # case (1) 0 in numerator: issue if 0 is largest, all negative, and rest are really negative
        # case (2) 0 in denominator: is possible unless case 3, we just ignore
        # case (3) 0 in both: not outlier, channel just kinda useless, ignore

        # get the ratio and get rid of nan values
        quantile_ratios = hundreth_quartile / comp_quantile
        quantile_ratios = torch.nan_to_num(quantile_ratios)
        # update averages, remembering to only update if didn't have zeros
        ratio_if_not_zero = any_non_zero_quantile_value * quantile_ratios

        # if num_batches and average_ratio are not initialized, we want to initialize them
        if (
            self.percentile_batches_tracked.shape[0] == 0
            or self.average_percentile_ratio.shape[0] == 0
        ):
            self.percentile_batches_tracked = torch.zeros_like(
                any_non_zero_quantile_value
            )
            self.average_percentile_ratio = torch.zeros_like(ratio_if_not_zero)

        # also initialize the constant channel var if that is not initialized separately
        if self.constant_channels.shape[0] == 0:
            self.constant_channels = torch.zeros_like(any_constant_channels)

        # get current num batches and average ratio
        num_batches = self.percentile_batches_tracked
        average_ratio = self.average_percentile_ratio

        # calculate new_number of batches, new_ratios, and get rid of nans because of 0 size batches
        new_number_of_batches: torch.Tensor = num_batches + any_non_zero_quantile_value
        new_ratios: torch.Tensor = (
            (average_ratio * num_batches) + ratio_if_not_zero
        ) / new_number_of_batches
        new_ratios = torch.nan_to_num(new_ratios)

        # update the number of non-constant channels
        new_constant_count: torch.Tensor = (
            self.constant_channels + any_constant_channels
        )

        # update the values locally
        self.percentile_batches_tracked.copy_(new_number_of_batches)
        self.average_percentile_ratio.copy_(new_ratios)
        self.constant_channels.copy_(new_constant_count)

        return x_copy

    @torch.jit.export
    def get_batch_to_epoch_ratio(self):
        # [FUNCTION DOC] Returns the average per-batch activation range divided by the whole-epoch range (max minus min).
        # Raises ValueError if the epoch range is zero, or if it is infinite (no data observed).
        epoch_activation_range = self.epoch_activation_max - self.epoch_activation_min

        if epoch_activation_range == torch.tensor(float(0)):
            raise ValueError("Range for Epoch is 0")
        elif epoch_activation_range == torch.tensor(float("inf")):
            raise ValueError(
                "No data has been run through observer or infinity value present"
            )
        else:
            return self.average_batch_activation_range / epoch_activation_range

    @torch.jit.export
    def reset_batch_and_epoch_values(self):
        # [FUNCTION DOC] Resets all counters and statistics to their initial values for a new epoch, keeping the device of
        # the max_val buffer. Replaces buffers with fresh tensors; note comp_percentile is left unchanged.
        # set all the values back to their original defaults for a new epoch
        # keep device
        device = self.max_val.device
        self.num_batches_tracked = 0
        self.average_batch_activation_range = torch.tensor(float(0), device=device)
        self.epoch_activation_min = torch.tensor(float("inf"), device=device)
        self.epoch_activation_max = torch.tensor(float("-inf"), device=device)
        self.min_val = torch.tensor([], device=device)
        self.max_val = torch.tensor([], device=device)
        self.average_percentile_ratio = torch.tensor([], device=device)
        self.percentile_batches_tracked = torch.tensor([], device=device)
        self.constant_channels = torch.tensor([], device=device)

    @torch.jit.export
    def calculate_qparams(self):  # type: ignore[override]
        # [FUNCTION DOC] Always raises a generic Exception: qparams are not meaningful for this observer.
        raise Exception(  # noqa: TRY002
            "calculate_qparams should not be called for ModelReportObserver"
        )
