# mypy: allow-untyped-defs
# [MODULE DOC] Helpers that turn the TorchScript upgrader graphs (from the C++ side) into
# a YAML-friendly list of bytecode tables. Imports two private torch._C bindings at import
# time; no other side effects. Not meant to be run directly (raises RuntimeError as main).

from torch._C import _compile_graph_to_code_table, _generate_upgraders_graph


def format_bytecode(table):
    # [FUNCTION DOC] Convert a sequence of (identifier, content) pairs into a mapping from
    # identifier to content, where every nested tuple in content is turned into a nested
    # list. Args: table, iterable of pairs. Returns: a new dict. No side effects.
    # given a nested tuple, convert it to nested list
    def listify(content):
        # [FUNCTION DOC] HELPER: Recursively replace tuples by lists; non-tuple values are
        # returned unchanged.
        if not isinstance(content, tuple):
            return content
        return [listify(i) for i in content]

    formatted_table = {}
    for entry in table:
        identifier = entry[0]
        content = entry[1]
        content = listify(content)
        formatted_table[identifier] = content
    return formatted_table


def generate_upgraders_bytecode() -> list:
    # [FUNCTION DOC] Ask the C++ side for the map of upgrader name to upgrader graph, compile
    # each graph into a code table, and return a list with one single-key mapping per
    # upgrader (name -> formatted bytecode table). Takes no arguments; no side effects
    # beyond the native calls.
    yaml_content = []
    upgraders_graph_map = _generate_upgraders_graph()
    for upgrader_name, upgrader_graph in upgraders_graph_map.items():
        bytecode_table = _compile_graph_to_code_table(upgrader_name, upgrader_graph)
        entry = {upgrader_name: format_bytecode(bytecode_table)}
        yaml_content.append(entry)
    return yaml_content


if __name__ == "__main__":
    raise RuntimeError("This file is not meant to be run directly")
