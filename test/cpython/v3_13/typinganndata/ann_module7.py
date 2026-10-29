# [MODULE DOC] HELPER: fixture with postponed annotations defining the constant
# DEFAULT_BUFFER_SIZE and class BufferedReader, whose docstring starts with a text signature.
# Tests class have ``__text_signature__``

from __future__ import annotations

DEFAULT_BUFFER_SIZE = 8192

class BufferedReader(object):
    # [CLASS DOC] HELPER: empty class whose docstring embeds a text-signature header so that
    # signature inspection can derive parameters from it.
    """BufferedReader(raw, buffer_size=DEFAULT_BUFFER_SIZE)\n--\n\n
    Create a new buffered reader using the given readable raw IO object.
    """
    pass
