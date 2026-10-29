# [MODULE DOC] HELPER: Registers the "torch_profile" debug server handler at import time
# (a side effect on the C++ handler registry).
import pathlib
import tempfile
import time

from torch._C._distributed_c10d import _register_handler, _Request, _Response
from torch.profiler import _ExperimentalConfig, profile


def _torch_profile(req: _Request, resp: _Response) -> None:
    # [FUNCTION DOC] HELPER: Debug server handler. Reads the required "duration" request
    # parameter (seconds, float; a missing or invalid value raises), profiles all threads with
    # shape recording for that long (blocking sleep), exports a chrome trace to a temporary
    # file and sets the response body to its JSON bytes with status 200.
    experimental_config = _ExperimentalConfig(
        profile_all_threads=True,
    )
    duration = float(req.get_param("duration"))
    with profile(record_shapes=True, experimental_config=experimental_config) as prof:
        time.sleep(duration)

    with tempfile.NamedTemporaryFile(prefix="torch_debug", suffix=".json") as f:
        prof.export_chrome_trace(f.name)
        resp.set_content(pathlib.Path(f.name).read_bytes(), "application/json")
        resp.set_status(200)


_register_handler("torch_profile", _torch_profile)
