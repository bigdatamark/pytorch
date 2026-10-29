# mypy: allow-untyped-defs
# [MODULE DOC] Defines _remote_device, a parsed, hashable description of a device on a
# possibly remote worker (worker name or rank plus a torch.device). No import-time side effects.

import torch


class _remote_device:
    # [CLASS DOC] HELPER: Immutable-by-convention value object parsed from a string or
    # torch.device. Holds _worker_name, _rank (one of the two or neither) and _device
    # (always a torch.device after construction). Supports equality, hashing and repr.
    """
    Represents a device on a remote worker.

    Args:
        remote_device (str or torch.device): Represents a device on a remote worker.
            The string format should be one of the following:

                1. "<workername>/<device>", where the device field can be parsed as torch.device type.
                   E.g., "trainer0/cpu", "trainer0", "ps0/cuda:0".
                   In addition, the device field can be optional and the default value is "cpu".
                2. "rank:<rank>/<device>", where <rank> is the rank of the
                   process and device can be parsed as torch.device type.
                   E.g., "rank:0/cpu", "rank:0", "rank:0/cuda:0"
                3. <workername> and <rank> are optional and formats like "cpu"
                    and "cuda:1", just represent local devices.
    """

    def __init__(self, remote_device: str | torch.device):
        # [FUNCTION DOC] Parses the input. A torch.device means local with no worker. A string
        # is split on a slash: two parts are worker and device; one part is a local device if
        # valid, else a worker name with cpu. Worker names of the form rank:N set the rank and
        # clear the name. Raises ValueError on malformed strings, empty worker name, bad rank
        # format or extra colons; TypeError for other input types; torch.device may raise on
        # an invalid device string.
        PARSE_ERROR = (
            f"Could not parse remote_device: {remote_device}. The valid format is "
            "'<workername>/<device>' or 'rank:<rank>/<device>' or '<device>'"
        )
        self._worker_name = None
        self._rank = None
        self._device: str | int | torch.device | None = None

        if isinstance(remote_device, torch.device):
            self._device = remote_device
        elif isinstance(remote_device, str):
            fields = remote_device.split("/")
            if len(fields) == 2:
                self._worker_name, self._device = fields
            elif len(fields) == 1:
                # Check if this is a valid device.
                if _remote_device._is_valid_local_device(fields[0]):
                    self._device = fields[0]
                else:
                    self._worker_name = fields[0]
                    self._device = "cpu"
            else:
                raise ValueError(PARSE_ERROR)
        else:
            raise TypeError(f"Invalid type for remote_device: {type(remote_device)}")

        # Do some basic sanity check (no empty string)
        if self._worker_name is not None and not self._worker_name:
            raise ValueError(PARSE_ERROR)

        # Validate the device.
        self._device = torch.device(self._device)

        # Check for rank based format.
        if self._worker_name is not None:
            fields = self._worker_name.split(":")
            if len(fields) == 2:
                # rank:<rank>/device format, extract rank
                if fields[0] == "rank" and fields[1].isdigit():
                    self._rank = int(fields[1])  # type: ignore[assignment]
                    self._worker_name = None
                else:
                    raise ValueError(PARSE_ERROR)
            elif len(fields) > 2:
                raise ValueError(PARSE_ERROR)

    @staticmethod
    def _is_valid_local_device(device):
        # [FUNCTION DOC] HELPER: Returns True if the argument can be turned into a torch.device,
        # False if constructing it raises any exception.
        # Check for torch.device
        try:
            torch.device(device)
            return True
        except Exception:
            return False

    def worker_name(self) -> str | None:
        # [FUNCTION DOC] Returns the worker name, or None when absent (local or rank form).
        """Return the name of remote worker representing the remote device and ``None`` if no worker name is available."""
        return self._worker_name

    def rank(self) -> int | None:
        # [FUNCTION DOC] Returns the integer rank, or None when not specified.
        """
        Returns the rank of remote worker representing the remote device.
        Returns ``None`` if no rank is available.
        """
        return self._rank

    def device(self) -> torch.device:
        # [FUNCTION DOC] Returns the torch.device on the remote worker.
        """Return the local device on the remote worker."""
        return self._device  # type: ignore[return-value]

    def __repr__(self):
        # [FUNCTION DOC] Formats as worker/device, rank:N/device, or just the device. The
        # branch for a missing device (name or rank only) is effectively unreachable since the
        # device is always set in the constructor; with nothing set it raises RuntimeError.
        if self._device is not None:
            if self._worker_name is not None:
                return f"{self._worker_name}/{self._device}"
            elif self._rank is not None:
                return f"rank:{self._rank}/{self._device}"
            else:
                return str(self._device)
        else:
            if self._worker_name is not None:
                return f"{self._worker_name}"
            elif self._rank is not None:
                return f"{self._rank}"
            else:
                raise RuntimeError("Invalid state!")

    def __eq__(self, other):
        # [FUNCTION DOC] Equal when other is a _remote_device with the same worker name, device
        # and rank.
        return isinstance(other, _remote_device) and (
            self._worker_name == other._worker_name
            and self._device == other._device
            and self._rank == other._rank
        )

    def __hash__(self):
        # [FUNCTION DOC] Combines hashes of worker name, device and rank with bitwise xor.
        return hash(self._worker_name) ^ hash(self._device) ^ hash(self._rank)
