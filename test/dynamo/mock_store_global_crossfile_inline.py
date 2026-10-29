# [MODULE DOC] HELPER: Holds a module global boolean flag, global_flag (initially False), with two
# setter functions, used to test dynamo handling of stores to globals of another file when inlined.
global_flag = False


def set_flag_true():
    # [FUNCTION DOC] HELPER: Set the module global flag global_flag to True.
    global global_flag
    global_flag = True


def set_flag_false():
    # [FUNCTION DOC] HELPER: Set the module global flag global_flag to False.
    global global_flag
    global_flag = False
