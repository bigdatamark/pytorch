# [MODULE DOC] Operator symbols used by the gradual-typing constraint system.
# Defines module-level string constants naming the arithmetic operators (add, sub, mul, div, mod),
# comparison operators (eq, neq, lt, gt, leq) and the gradual-typing relations (implication,
# matching, consistency, precision) used as the op field of binary constraints.
# Matching, precision and leq use non-ASCII unicode symbols as their values. No import-time side effects.
op_add = "+"
op_sub = "-"
op_mul = "*"
op_div = "/"
op_eq = "="
op_neq = "!="
op_imp = "=>"
op_matching = "\u22b3"  # (contains)
op_consistency = "~"
op_precision = "\u2291"  # (square image of or equal to)
op_leq = "\u2264"  # less-than or equal to
op_lt = "<"
op_gt = ">"
op_mod = "%"
