# [MODULE DOC]
# Builds the markdown messages the merge bot posts as PR comments: the "Merge started"
# explanation (varies by force / ignore-current flags) and the revert-started message.
# Defines link constants and CIFLOW label regexes. No import-time side effects besides
# compiling regexes.
from __future__ import annotations

import os
import re
from re import Pattern


BOT_COMMANDS_WIKI = "https://github.com/pytorch/pytorch/wiki/Bot-commands"

CIFLOW_LABEL = re.compile(r"^ciflow/.+")
CIFLOW_TRUNK_LABEL = re.compile(r"^ciflow/trunk")

OFFICE_HOURS_LINK = "https://github.com/pytorch/pytorch/wiki/Dev-Infra-Office-Hours"
CONTACT_US = f"Questions? Feedback? Please reach out to the [PyTorch DevX Team]({OFFICE_HOURS_LINK})"
ALTERNATIVES = f"Learn more about merging in the [wiki]({BOT_COMMANDS_WIKI})."


def has_label(labels: list[str], pattern: Pattern[str] = CIFLOW_LABEL) -> bool:
    # [FUNCTION DOC]
    # Returns True if any label in the list matches the regex pattern at its start
    # (default: any ciflow label). Pure function, no side effects.
    return len(list(filter(pattern.match, labels))) > 0


class TryMergeExplainer:
    # [CLASS DOC]
    # Holds the merge request parameters (force flag, ignore-current flag, PR labels and
    # identity) and renders the markdown comment explaining what the merge command will do.
    # Key methods: _get_flag_msg (flag-dependent sentence) and get_merge_message (full text).
    force: bool
    labels: list[str]
    pr_num: int
    org: str
    project: str
    ignore_current: bool

    has_trunk_label: bool
    has_ciflow_label: bool

    def __init__(
        self,
        force: bool,
        labels: list[str],
        pr_num: int,
        org: str,
        project: str,
        ignore_current: bool,
    ):
        # [FUNCTION DOC]
        # Stores all arguments as attributes. No validation and no side effects; the
        # has_trunk_label / has_ciflow_label annotations are never assigned here.
        self.force = force
        self.labels = labels
        self.pr_num = pr_num
        self.org = org
        self.project = project
        self.ignore_current = ignore_current

    def _get_flag_msg(
        self,
        ignore_current_checks: list[tuple[str, str | None, int | None]] | None = None,
    ) -> str:
        # [FUNCTION DOC] HELPER: Picks the main sentence of the merge message.
        # Force: warns the merge bypasses CI. Ignore-current with a provided check list:
        # lists those checks as markdown links (name, url). Otherwise: merge once checks pass.
        # Args: ignore_current_checks is a list of (name, url, id) tuples or None.
        if self.force:
            return (
                "Your change will be merged immediately since you used the force (-f) flag, "
                + "**bypassing any CI checks** (ETA: 1-5 minutes).  "
                + "Please use `-f` as last resort and instead consider `-i/--ignore-current` "
                + "to continue the merge ignoring current failures.  This will allow "
                + "currently pending tests to finish and report signal before the merge."
            )
        elif self.ignore_current and ignore_current_checks is not None:
            msg = f"Your change will be merged while ignoring the following {len(ignore_current_checks)} checks: "
            msg += ", ".join(f"[{x[0]}]({x[1]})" for x in ignore_current_checks)
            return msg
        else:
            return "Your change will be merged once all checks pass (ETA 0-4 Hours)."

    def get_merge_message(
        self,
        ignore_current_checks: list[tuple[str, str | None, int | None]] | None = None,
    ) -> str:
        # [FUNCTION DOC]
        # Assembles the full "Merge started" markdown: title, flag message, wiki link,
        # contact line, and a collapsed "Advanced Debugging" section linking the workflow
        # run (env var GH_RUN_URL, which renders as None text if unset).
        # KNOWN ISSUE: the contact line and the details block are concatenated with no
        # separating newline.
        title = "### Merge started"
        main_message = self._get_flag_msg(ignore_current_checks)

        advanced_debugging = "\n".join(
            (
                "<details><summary>Advanced Debugging</summary>",
                "Check the merge workflow status ",
                f'<a href="{os.getenv("GH_RUN_URL")}">here</a>',
                "</details>",
            )
        )

        msg = title + "\n"
        msg += main_message + "\n\n"
        msg += ALTERNATIVES + "\n\n"
        msg += CONTACT_US
        msg += advanced_debugging
        return msg


def get_revert_message(org: str, project: str, pr_num: int) -> str:
    # [FUNCTION DOC]
    # Returns the comment text announcing a revert job started, linking the run from env
    # var GH_RUN_URL, followed by the contact line.
    # KNOWN ISSUE: org, project and pr_num are unused.
    msg = (
        "@pytorchbot successfully started a revert job."
        + f" Check the current status [here]({os.getenv('GH_RUN_URL')}).\n"
    )
    msg += CONTACT_US
    return msg
