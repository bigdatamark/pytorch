# [PACKAGE DOC] Inductor kernel package: importing it registers the flex attention/decoding, mm, mm_common, mm_plus_mm and flex_gemm lowerings/templates.
from . import flex, mm, mm_common, mm_plus_mm
from .flex_gemm import lowering as flex_gemm_lowering
