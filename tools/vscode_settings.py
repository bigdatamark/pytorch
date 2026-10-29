#!/usr/bin/env python3

# [MODULE DOC] Script that merges the recommended VS Code settings file into the user's
# settings file in the .vscode folder. Uses the json5 package when available, which allows
# comments and trailing commas. No import-time side effects other than the optional import.

from pathlib import Path


try:
    # VS Code settings allow comments and trailing commas, which are not valid JSON.
    import json5 as json  # type: ignore[import]

    HAS_JSON5 = True
except ImportError:
    import json  # type: ignore[no-redef]

    HAS_JSON5 = False


ROOT_FOLDER = Path(__file__).absolute().parent.parent
VSCODE_FOLDER = ROOT_FOLDER / ".vscode"
RECOMMENDED_SETTINGS = VSCODE_FOLDER / "settings_recommended.json"
SETTINGS = VSCODE_FOLDER / "settings.json"


# settings can be nested, so we need to recursively update the settings.
def deep_update(d: dict, u: dict) -> dict:  # type: ignore[type-arg]
    # [FUNCTION DOC] Recursively merges the update mapping into the first mapping, in place:
    # nested mappings are merged, lists are concatenated, other values replaced. Returns the
    # first mapping.
    # KNOWN ISSUE: lists are appended without removing duplicates, so running the script
    # repeatedly duplicates list entries.
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = deep_update(d.get(k, {}), v)
        elif isinstance(v, list):
            d[k] = d.get(k, []) + v
        else:
            d[k] = v
    return d


def main() -> None:
    # [FUNCTION DOC] Loads the recommended and current settings (an absent current file counts
    # as empty), merges them and writes the result back with four-space indentation. Raises
    # SystemExit with a message if the current settings cannot be parsed.
    # Existing comments in the user's settings file are lost.
    recommended_settings = json.loads(RECOMMENDED_SETTINGS.read_text())
    try:
        current_settings_text = SETTINGS.read_text()
    except FileNotFoundError:
        current_settings_text = "{}"

    try:
        current_settings = json.loads(current_settings_text)
    except ValueError as ex:  # json.JSONDecodeError is a subclass of ValueError
        if HAS_JSON5:
            raise SystemExit("Failed to parse .vscode/settings.json.") from ex
        raise SystemExit(
            "Failed to parse .vscode/settings.json. "
            "Maybe it contains comments or trailing commas. "
            "Try `pip install json5` to install an extended JSON parser."
        ) from ex

    settings = deep_update(current_settings, recommended_settings)

    SETTINGS.write_text(
        json.dumps(
            settings,
            indent=4,
        )
        + "\n",  # add a trailing newline
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
