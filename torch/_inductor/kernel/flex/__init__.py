# [PACKAGE DOC] Flex kernel package: importing it registers the flex_attention and flex_decoding lowerings/templates.
# mypy: allow-untyped-defs
# Import so here and then reimport above so that register_lowering gets triggered
from . import flex_attention, flex_decoding
