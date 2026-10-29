# [MODULE DOC] Thin ClickHouse access helpers: a cached client built from environment
# variables and a query function returning rows as dictionaries. Import time: imports the
# clickhouse_connect package (an ImportError if it is not installed).
import json
import os
from functools import lru_cache
from typing import Any

import clickhouse_connect  # type: ignore[import]


# [FUNCTION DOC] Returns a process-wide cached secure HTTPS ClickHouse client (port 8443).
# Reads the endpoint, username and password from environment variables and raises a key error
# if any is unset. The endpoint has any https prefix and :8443 suffix removed first.
@lru_cache(maxsize=1)
def get_clickhouse_client() -> Any:
    endpoint = os.environ["CLICKHOUSE_ENDPOINT"]
    # I cannot figure out why these values aren't being handled automatically
    # when it is fine in the lambda
    endpoint = endpoint.removeprefix("https://")
    endpoint = endpoint.removesuffix(":8443")
    return clickhouse_connect.get_client(
        host=endpoint,
        user=os.environ["CLICKHOUSE_USERNAME"],
        password=os.environ["CLICKHOUSE_PASSWORD"],
        secure=True,
        interface="https",
        port=8443,
    )


def query_clickhouse(query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    # [FUNCTION DOC] Runs a parameterized query on the cached client, asking for newline
    # delimited JSON, and returns one dictionary per non-empty output line. Raises whatever
    # the client raises on connection or query failure (I/O over the network).
    """
    Queries ClickHouse.  Returns datetime in YYYY-MM-DD HH:MM:SS format.
    """

    def convert_to_json_list(res: bytes) -> list[dict[str, Any]]:
        # [FUNCTION DOC] HELPER: Decodes the raw bytes as text, splits on newlines, and parses
        # each non-empty line as JSON; raises a JSON decode error on a malformed line.
        rows = []
        for row in res.decode().split("\n"):
            if row:
                rows.append(json.loads(row))
        return rows

    res = get_clickhouse_client().raw_query(query, params, fmt="JSONEachRow")
    return convert_to_json_list(res)
