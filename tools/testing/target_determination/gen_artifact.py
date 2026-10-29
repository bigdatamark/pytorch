# [MODULE DOC] Helper for writing the target determination exclusions artifact that CI
# uploads, as a JSON file in the test reports directory of the repo.
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]


def gen_ci_artifact(included: list[Any], excluded: list[Any]) -> None:
    # [FUNCTION DOC] Writes a JSON file holding the included and excluded lists to the test
    # reports directory under the repo test directory, with a random hex suffix in the name
    # to avoid collisions. Raises a file-not-found error if that directory does not exist.
    file_name = f"td_exclusions-{os.urandom(10).hex()}.json"
    with open(REPO_ROOT / "test" / "test-reports" / file_name, "w") as f:
        json.dump({"included": included, "excluded": excluded}, f)
