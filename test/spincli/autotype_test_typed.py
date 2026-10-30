# [MODULE DOC] Fixture for the spin autotype test: the expected result of type inference. It holds
# simple module constants of several builtin kinds and a greeter function whose parameter
# annotation is the union of int, str and None with a None default, returning str, followed by two
# calls with a string and a number. Compared by syntax tree against the inferred version of the
# untyped sibling file.
NUMERIC_CONSTANT = 42
STRING_CONSTANT = "Alice"
LIST_CONSTANT = [1, 2, 3, 4, 5]
DICT_CONSTANT = {"key1": "value1", "key2": "value2"}
TUPLE_CONSTANT = (1, 2, 3)
SET_CONSTANT = {1, 2, 3, 4, 5}
BOOLEAN_CONSTANT = True
NONE_CONSTANT = None


def greeter(name: int | str | None = None) -> str:
    # [FUNCTION DOC] Returns a greeting string that embeds the given name (default None); the
    # annotations match what pyrefly infer is expected to produce for the untyped sibling.
    return f"Hello, {name}!"


greeter(STRING_CONSTANT)
greeter(NUMERIC_CONSTANT)
