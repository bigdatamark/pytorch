# [MODULE DOC] Negative typing test. Type-checker input expecting an unsupported-operand
# error when a Size object is concatenated with a tuple holding a string, since Size
# addition only accepts integer sequences.
from torch import Size


s1 = Size([1, 2, 3])
s1 + ("foo",)  # E: Unsupported operand types
