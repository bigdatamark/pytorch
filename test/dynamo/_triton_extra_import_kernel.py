# Owner(s): ["module: dynamo"]
# [MODULE DOC] HELPER: Defines a module-level user Triton kernel that uses libdevice, imported
# from triton.language.extra, so tests can check dynamo handles extra Triton imports. Importing
# requires triton to be installed.

# A user-defined Triton kernel at module scope, using an imported Triton
# helper name other than the usual `triton` or `tl` globals.
import triton
import triton.language as tl
from triton.language.extra import libdevice


# [FUNCTION DOC] HELPER: Triton kernel; each program handles BLOCK elements starting at its
# program id times BLOCK, loads them from x_ptr, applies libdevice exp2 and stores the results
# back in place. No bounds mask is used, so the buffer length must be a multiple of BLOCK.
@triton.jit
def triton_kernel_with_extra_import(x_ptr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = libdevice.exp2(tl.load(x_ptr + offsets))
    tl.store(x_ptr + offsets, values)
