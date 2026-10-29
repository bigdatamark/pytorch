# [MODULE DOC] YAML loading/dumping helpers for the codegen.
# Picks the C-accelerated safe loader and dumper when libyaml bindings exist, else the pure
# Python safe ones, and exposes the dumper as YamlDumper. Defines YamlLoader, a safe loader
# that rejects duplicate mapping keys. No side effects at import beyond these imports.

# Safely load fast C Yaml loader/dumper if they are available
try:
    from yaml import CSafeLoader as Loader
except ImportError:
    from yaml import SafeLoader as Loader  # type: ignore[assignment, misc]

try:
    from yaml import CSafeDumper as Dumper
except ImportError:
    from yaml import SafeDumper as Dumper  # type: ignore[assignment, misc]
YamlDumper = Dumper


# A custom loader for YAML that errors on duplicate keys.
# This doesn't happen by default: see https://github.com/yaml/pyyaml/issues/165
class YamlLoader(Loader):
    # [CLASS DOC] Safe YAML loader (C or pure Python base, whichever was importable) whose only
    # change is that building any mapping fails when the same key appears twice in it.
    def construct_mapping(self, node, deep=False):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] Builds a mapping from a YAML mapping node. First constructs every key
        # and checks it against the keys seen so far in this node; a repeat raises an
        # AssertionError naming the key and the node's start line. Then delegates to the base
        # class to build and return the real mapping. Args: node, deep (build nested values
        # eagerly). Quadratic in the number of keys because seen keys are kept in a list.
        mapping = []
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)  # type: ignore[no-untyped-call]
            if key in mapping:
                raise AssertionError(
                    f"Found a duplicate key in the yaml. key={key}, line={node.start_mark.line}"
                )
            mapping.append(key)
        mapping = super().construct_mapping(node, deep=deep)  # type: ignore[no-untyped-call]
        return mapping
