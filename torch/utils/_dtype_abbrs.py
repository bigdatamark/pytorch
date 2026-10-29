# [MODULE DOC] Defines dtype_abbrs, a mapping from every torch dtype (as reported by the C
# extension) to its short abbreviation string. Built at import time; used for testing/logging.
import torch


# Used for testing and logging
dtype_abbrs = {dt: dt.abbr for dt in torch._C._get_all_dtypes()}
