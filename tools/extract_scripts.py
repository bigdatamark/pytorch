#!/usr/bin/env python3

# [MODULE DOC]
# Command line tool that extracts the inline shell and github-script code from every step
# of the GitHub Actions workflows under .github/workflows into standalone script files in
# an output directory (one directory per workflow and job), so they can be linted. Exits
# with an error if any extracted script still contains a GitHub Actions expression.
# Defines the Step alias and the Script typed dict. Runs main only as a script.
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any
from typing_extensions import TypedDict  # Python 3.11+

import yaml


Step = dict[str, Any]


class Script(TypedDict):
    # [CLASS DOC] Typed dictionary describing one extracted script: the file extension to
    # use (including the dot) and the full script text.
    extension: str
    script: str


def extract(step: Step) -> Script | None:
    # [FUNCTION DOC] Extracts the script from one workflow step mapping. If the step has a
    # run field and its shell (default bash) is known, returns the extension for that shell
    # and the run text, with a shebang and strict-mode header prepended for bash and sh.
    # Otherwise, if the step uses the github-script action and has a script input, returns
    # a .js script with that text. Returns None when the step has no extractable script.
    # KNOWN ISSUE: an unrecognized shell value yields no extension, so the run text is
    # silently skipped.
    run = step.get("run")

    # https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions#using-a-specific-shell
    shell = step.get("shell", "bash")
    extension = {
        "bash": ".sh",
        "pwsh": ".ps1",
        "python": ".py",
        "sh": ".sh",
        "cmd": ".cmd",
        "powershell": ".ps1",
    }.get(shell)

    is_gh_script = step.get("uses", "").startswith("actions/github-script@")
    gh_script = step.get("with", {}).get("script")

    if run is not None and extension is not None:
        script = {
            "bash": f"#!/usr/bin/env bash\nset -eo pipefail\n{run}",
            "sh": f"#!/usr/bin/env sh\nset -e\n{run}",
        }.get(shell, run)
        return {"extension": extension, "script": script}  # type: ignore[typeddict-item]
    elif is_gh_script and gh_script is not None:
        return {"extension": ".js", "script": gh_script}
    else:
        return None


def main() -> None:
    # [FUNCTION DOC] Entry point. Requires a command line option for the output directory
    # and exits with a message if it already exists. Loads every file in the workflows
    # directory as YAML, and for each job step with an extractable script writes it to a
    # file under the output directory, named with a zero-padded step index and a sanitized
    # step name. Scripts containing a GitHub Actions expression marker are reported on
    # stderr, and after all files are written the process exits with an error message.
    # Side effects: creates directories and files, writes stderr, may terminate the process.
    # KNOWN ISSUE: the index width is computed from the last step index, but steps are
    # numbered from 1, so the final index can be one digit wider than the padding.
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} already exists; aborting to avoid overwriting")

    gha_expressions_found = False

    for p in Path(".github/workflows").iterdir():
        with open(p, "rb") as f:
            workflow = yaml.safe_load(f)

        for job_name, job in workflow["jobs"].items():
            job_dir = out / p / job_name
            if "steps" not in job:
                continue
            steps = job["steps"]
            index_chars = len(str(len(steps) - 1))
            for i, step in enumerate(steps, start=1):
                extracted = extract(step)
                if extracted:
                    script = extracted["script"]
                    step_name = step.get("name", "")
                    if "${{" in script:
                        gha_expressions_found = True
                        print(
                            f"{p} job `{job_name}` step {i}: {step_name}",
                            file=sys.stderr,
                        )

                    job_dir.mkdir(parents=True, exist_ok=True)

                    sanitized = re.sub(
                        "[^a-zA-Z_]+",
                        "_",
                        f"_{step_name}",
                    ).rstrip("_")
                    extension = extracted["extension"]
                    filename = f"{i:0{index_chars}}{sanitized}{extension}"
                    (job_dir / filename).write_text(script)

    if gha_expressions_found:
        sys.exit(
            "Each of the above scripts contains a GitHub Actions "
            "${{ <expression> }} which must be replaced with an `env` variable"
            " for security reasons."
        )


if __name__ == "__main__":
    main()
