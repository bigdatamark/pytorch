# [MODULE DOC] Custom pytest plugin module for splitting a test run into shards, adapted
# from the pytest-shard project. Differences: shard ids are one-indexed and an option can
# print the items in the shard. Defines an option-registration hook function and a plugin
# class that filters collected items by a hash of their node id. No import-time side effects.
"""
Custom pytest shard plugin
https://github.com/AdamGleave/pytest-shard/blob/64610a08dac6b0511b6d51cf895d0e1040d162ad/pytest_shard/pytest_shard.py#L1
Modifications:
* shards are now 1 indexed instead of 0 indexed
* option for printing items in shard
"""

import hashlib

from _pytest.config.argparsing import Parser


def pytest_addoptions(parser: Parser):
    """Add options to control sharding."""
    group = parser.getgroup("shard")
    group.addoption(
        "--shard-id", dest="shard_id", type=int, default=1, help="Number of this shard."
    )
    group.addoption(
        "--num-shards",
        dest="num_shards",
        type=int,
        default=1,
        help="Total number of shards.",
    )
    group.addoption(
        "--print-items",
        dest="print_items",
        action="store_true",
        default=False,
        help="Print out the items being tested in this shard.",
    )


class PytestShardPlugin:
    def __init__(self, config):
        self.config = config

    def pytest_report_collectionfinish(self, config, items) -> str:
        """Log how many and which items are tested in this shard."""
        msg = f"Running {len(items)} items in this shard"
        if config.getoption("print_items"):
            msg += ": " + ", ".join([item.nodeid for item in items])
        return msg

    def sha256hash(self, x: str) -> int:
        return int.from_bytes(hashlib.sha256(x.encode()).digest(), "little")

    def filter_items_by_shard(self, items, shard_id: int, num_shards: int):
        """Computes `items` that should be tested in `shard_id` out of `num_shards` total shards."""
        new_items = [
            item
            for item in items
            if self.sha256hash(item.nodeid) % num_shards == shard_id - 1
        ]
        return new_items

    def pytest_collection_modifyitems(self, config, items):
        """Mutate the collection to consist of just items to be tested in this shard."""
        shard_id = config.getoption("shard_id")
        shard_total = config.getoption("num_shards")
        if shard_id < 1 or shard_id > shard_total:
            raise ValueError(
                f"{shard_id} is not a valid shard ID out of {shard_total} total shards"
            )

        items[:] = self.filter_items_by_shard(items, shard_id, shard_total)

        # [FUNCTION DOC] Hook that restricts the collection to this shard. Reads shard-id and
        # num-shards options; raises ValueError if the id is below 1 or above the total.
        # Otherwise replaces the contents of the items list in place with the filtered items.
        # [FUNCTION DOC] Return a new list of the items whose node id hash modulo num_shards
        # equals the zero-based shard index (shard_id minus one). Does not mutate the input.
        # [FUNCTION DOC] Hash a string to an integer: utf-8 encode, sha256 digest, interpreted
        # as a little-endian unsigned integer. Deterministic across runs, unlike the builtin.
        # [FUNCTION DOC] Hook returning the report line shown after collection: the number of
        # items in this shard, followed by a comma-joined list of node ids when the print-items
        # option is set.
        # [FUNCTION DOC] Store the pytest config object on the instance.
    # [CLASS DOC] Pytest plugin object that keeps only the collected test items belonging to
    # the configured shard. Stores the pytest config. Hooks: collection-finish report and
    # collection item modification; sharding is by stable sha256 hash of each item node id.
    # [FUNCTION DOC] Register the shard command-line option group with the pytest parser:
    # shard-id (int, default 1), num-shards (int, default 1) and print-items (flag, default
    # False). Returns nothing.
    # KNOWN ISSUE: the hook is named with a trailing s, unlike the standard pytest option hook
    # name, so pytest would not call it automatically unless invoked explicitly.