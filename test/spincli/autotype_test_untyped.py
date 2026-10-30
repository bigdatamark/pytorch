# [MODULE DOC] Fixture input for the spin autotype test: same constants and calls as the typed
# sibling file but with an unannotated greeter. The test copies this file to a temporary directory
# and runs the pyrefly inference command on the copy. Never modified in place.
NUMERIC_CONSTANT = 42
STRING_CONSTANT = "Alice"
LIST_CONSTANT = [1, 2, 3, 4, 5]
DICT_CONSTANT = {"key1": "value1", "key2": "value2"}
TUPLE_CONSTANT = (1, 2, 3)
SET_CONSTANT = {1, 2, 3, 4, 5}
BOOLEAN_CONSTANT = True
NONE_CONSTANT = None


def greeter(name=None):
    # [FUNCTION DOC] Returns a greeting string that embeds the given name (default None).
    # Intentionally has no annotations so inference can add them.
    return f"Hello, {name}!"


greeter(STRING_CONSTANT)
greeter(NUMERIC_CONSTANT)
