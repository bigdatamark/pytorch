# mypy: ignore-errors
# [MODULE DOC] Test fixture module used to check how division behaves in a module that
# is compiled with the future-division setting. Defines two constant-returning functions.



def div_int_future():
    # [FUNCTION DOC] HELPER: Returns the result of dividing integer 1 by integer 2.
    return 1 / 2


def div_float_future():
    # [FUNCTION DOC] HELPER: Returns the float result of dividing 3.14 by 0.125.
    return 3.14 / 0.125
