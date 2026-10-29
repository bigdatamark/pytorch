# [MODULE DOC] Helpers for writing Markdown to the GitHub Actions step summary file
# (named by the GITHUB_STEP_SUMMARY environment variable): Jinja templates for content
# blocks, lists and tables, small Markdown builders, and summarizers for files and
# wheel listings. Import compiles the module level templates.
from __future__ import annotations

import logging
import os
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

from cli.lib.common.utils import get_wheels
from jinja2 import Template


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


logger = logging.getLogger(__name__)

_TPL_CONTENT = Template(
    textwrap.dedent("""\
    ## {{ title }}

    ```{{ lang }}
    {{ content }}
    ```
""")
)

_TPL_LIST_ITEMS = Template(
    textwrap.dedent("""\
    ## {{ title }}
    {% for it in items %}
    - {{ it.pkg }}: {{ it.relpath }}
    {% else %}
    _(no item found)_
    {% endfor %}
    """)
)

_TPL_TABLE = Template(
    textwrap.dedent("""\
    {%- if rows %}
    | {{ cols | join(' | ') }} |
    |{%- for _ in cols %} --- |{%- endfor %}
    {%- for r in rows %}
    | {%- for c in cols %} {{ r.get(c, "") }} |{%- endfor %}
    {%- endfor %}
    {%- else %}
    _(no data)_
    {%- endif %}
""")
)


def gh_summary_path() -> Path | None:
    # [FUNCTION DOC] Read GITHUB_STEP_SUMMARY and return it as a Path, or None when it
    # is unset or empty.
    """Return the Path to the GitHub step summary file, or None if not set."""
    p = os.environ.get("GITHUB_STEP_SUMMARY")
    return Path(p) if p else None


def write_gh_step_summary(md: str, *, append_content: bool = True) -> bool:
    # [FUNCTION DOC] Write dedented, stripped Markdown plus a trailing newline to the
    # step summary file, appending by default or truncating when append_content is
    # false (UTF-8). Returns False after logging when the variable is unset, else True.
    # I/O errors propagate.
    """
    Write Markdown content to the GitHub Step Summary file if GITHUB_STEP_SUMMARY is set.
    append_content: default true, if True, append to the end of the file, else overwrite the whole file

    Returns:
        True if written successfully (in GitHub Actions environment),
        False if skipped (e.g., running locally where the variable is not set).
    """
    sp = gh_summary_path()
    if not sp:
        logger.info("[gh-summary] GITHUB_STEP_SUMMARY not set, skipping write.")
        return False

    md_clean = textwrap.dedent(md).strip() + "\n"

    mode = "a" if append_content else "w"
    with sp.open(mode, encoding="utf-8") as f:
        f.write(md_clean)
    return True


def md_heading(text: str, level: int = 2) -> str:
    # [FUNCTION DOC] Return a Markdown heading line; level is clamped to 1 through 6.
    """Generate a Markdown heading string with the given level (1-6)."""
    return f"{'#' * max(1, min(level, 6))} {text}\n"


def md_details(summary: str, content: str) -> str:
    # [FUNCTION DOC] Return a collapsible details block with the given summary line and
    # body text (no escaping is applied).
    """Generate a collapsible <details> block with a summary and inner content."""
    return f"<details>\n<summary>{summary}</summary>\n\n{content}\n\n</details>\n"


def summarize_content_from_file(
    output_dir: Path,
    freeze_file: str,
    title: str = "Content from file",
    code_lang: str = "",  # e.g. "text" or "ini"
) -> bool:
    # [FUNCTION DOC] Append the stripped text of a file in output_dir to the step
    # summary as a titled fenced block with the given language tag. Returns False if
    # the file does not exist; otherwise the result of the summary write (False when
    # not running in GitHub Actions).
    f = Path(output_dir) / freeze_file
    if not f.exists():
        return False
    content = f.read_text(encoding="utf-8").strip()
    md = render_content(content, title=title, lang=code_lang)
    return write_gh_step_summary(md)


def summarize_wheels(path: Path, title: str = "Wheels", max_depth: int = 3):
    # [FUNCTION DOC] Find wheels under path (up to max_depth levels, via get_wheels)
    # and append a titled bullet list of them to the step summary. Returns False when
    # no wheels are found, else the summary write result.
    items = get_wheels(path, max_depth=max_depth)
    if not items:
        return False
    md = render_list(items, title=title)
    return write_gh_step_summary(md)


def md_kv_table(rows: Iterable[Mapping[str, str | int | float]]) -> str:
    # [FUNCTION DOC] Render mappings as a Markdown table whose columns are the union of
    # all keys; missing cells are empty. Returns a placeholder line when there are no
    # rows. KNOWN ISSUE: column order comes from a set, so it is not deterministic.
    """
    Render a list of dicts as a Markdown table using Jinja template.
    """
    rows = list(rows)
    cols = list({k for r in rows for k in r})
    md = _TPL_TABLE.render(cols=cols, rows=rows).strip() + "\n"
    return md


def render_list(
    items: Iterable[str],
    *,
    title: str = "List",
) -> str:
    # [FUNCTION DOC] Render a titled bullet list where each item shows its pkg and
    # relpath attributes (a placeholder line when empty). KNOWN ISSUE: the annotation
    # says strings but the template needs objects with those two attributes.
    tpl = _TPL_LIST_ITEMS
    md = tpl.render(title=title, items=items)
    return md


def render_content(
    content: str,
    *,
    title: str = "Content",
    lang: str = "text",
) -> str:
    # [FUNCTION DOC] Render a titled heading followed by the content in a fenced code
    # block tagged with lang.
    tpl = _TPL_CONTENT
    md = tpl.render(title=title, content=content, lang=lang)
    return md
