# [MODULE DOC] Helpers for reading benchmark configuration: parsing a head-count pair string,
# loading a configuration file written as JSON or a simple YAML subset (with no external YAML
# dependency), and printing the default configuration template. Defines the default
# configuration mapping at import time.
"""Configuration utilities for parsing JSON and YAML config files."""

import json
import re


def heads_input_type(s: str) -> tuple[int, int]:
    # [FUNCTION DOC] Parses a string holding two comma separated integers (query heads and
    # key-value heads) and returns them as a pair. Raises ValueError with a fixed message if
    # the string cannot be split into exactly two integers.
    """Convert string format 'Hq,Hkv' to tuple (Hq, Hkv)."""
    try:
        hq, hkv = map(int, s.split(","))
        return hq, hkv
    except Exception as e:
        raise ValueError("Heads must be Hq,Hkv") from e


default_config = {
    "dynamic": False,
    "calculate_bwd": False,
    "dtype": "bfloat16",
    "b": [2, 8, 16],
    "nh": ["16,16", "16,2"],
    "s": [512, 1024, 4096],
    "d": [64, 128],
    "mods": ["noop", "causal", "alibi", "sliding_window"],
    "backend": ["efficient"],
    "max_autotune": False,
    "decoding": False,
    "kv_size": None,
    "throughput": True,
    "save_path": None,
    "output_json_for_dashboard": None,
    "benchmark_name": "PyTorch operator microbenchmark",
}


def load_config_file(config_path: str) -> dict:
    # [FUNCTION DOC] Reads the file at the path, tries to parse it as JSON and falls back to
    # the simple YAML parser on a decode error. If the heads entry exists and is a list,
    # converts every string item to a head-count pair and leaves other items unchanged.
    # Returns the configuration mapping. Raises FileNotFoundError if the file is missing.
    # KNOWN ISSUE: the docstring says ValueError is raised for an invalid format, but the
    # fallback parser never raises for malformed content; only a bad head-count string raises
    # ValueError.
    """Load configuration from JSON or YAML file.

    Automatically converts 'nh' field from strings to tuples.

    Args:
        config_path: Path to the configuration file

    Returns:
        Dictionary containing the configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file format is invalid
    """
    with open(config_path) as f:
        config_str = f.read()

    # Try to load as JSON first
    try:
        config = json.loads(config_str)
    except json.JSONDecodeError:
        # Fall back to YAML parsing
        config = _parse_simple_yaml(config_str)

    # Apply automatic conversions for 'nh' field
    if "nh" in config and isinstance(config["nh"], list):
        config["nh"] = [
            heads_input_type(h) if isinstance(h, str) else h for h in config["nh"]
        ]

    return config


def _parse_simple_yaml(yaml_str: str) -> dict:
    # [FUNCTION DOC] HELPER: Parses a minimal YAML subset line by line: strips text after a
    # hash sign, skips lines without a colon, and splits at the first colon. Values become
    # booleans, None (for null, none or empty), a list (bracketed items, quoted or bare,
    # converted to integers when possible), a quoted string with quotes stripped, or an
    # integer, float or plain string in that order of preference. Returns the mapping.
    # KNOWN ISSUE: comment stripping cuts at the first hash sign even inside quoted strings,
    # and bare list items are only converted to integers, never floats.
    """Simple YAML parser for basic configs (without external dependencies).

    Supports:
    - key: value pairs
    - booleans (true/false)
    - null values
    - integers and floats
    - strings (quoted and unquoted)
    - lists in JSON format [item1, item2, ...]
    - comments (lines starting with # or after #)

    Args:
        yaml_str: YAML content as string

    Returns:
        Dictionary containing parsed YAML content
    """
    config = {}

    for line in yaml_str.split("\n"):
        # Remove comments
        line = line.split("#")[0].strip()

        if not line or ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        # Parse value based on type
        if value.lower() == "true":
            config[key] = True
        elif value.lower() == "false":
            config[key] = False
        elif value.lower() in ("null", "none", ""):
            config[key] = None
        elif value.startswith("[") and value.endswith("]"):
            # Parse list - handle quoted strings properly
            pattern = r'"([^"]+)"|\'([^\']+)\'|([^,\[\]\s]+)'
            matches = re.findall(pattern, value[1:-1])  # Remove [ ]
            parsed_items = []
            for match in matches:
                # match is a tuple of (double_quoted, single_quoted, unquoted)
                item = match[0] or match[1] or match[2]
                item = item.strip()
                if item:
                    try:
                        parsed_items.append(int(item))
                    except ValueError:
                        parsed_items.append(item)
            config[key] = parsed_items
        elif value.startswith(('"', "'")):
            config[key] = value.strip("\"'")
        else:
            # Try to parse as number
            try:
                config[key] = int(value)
            except ValueError:
                try:
                    config[key] = float(value)
                except ValueError:
                    config[key] = value

    return config


def print_default_config(output_format: str) -> None:
    # [FUNCTION DOC] Prints the default configuration to standard output: as indented JSON
    # when the format is json, otherwise (any other value) as simple YAML-like key lines with
    # null for missing values, lowercase booleans, quoted strings and JSON lists.
    """Print a default configuration template in JSON or YAML format.

    Args:
        output_format: Either "json" or "yaml"
    """
    if output_format == "json":
        print(json.dumps(default_config, indent=2))
    else:  # yaml
        for key, value in default_config.items():
            if value is None:
                print(f"{key}: null")
            elif isinstance(value, bool):
                print(f"{key}: {str(value).lower()}")
            elif isinstance(value, str):
                print(f'{key}: "{value}"')
            elif isinstance(value, list):
                print(f"{key}: {json.dumps(value)}")
            else:
                print(f"{key}: {value}")
