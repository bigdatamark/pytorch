# [MODULE DOC] ExportDB example: a null context manager around computation.
# Defines example_args, tags and model.
# mypy: allow-untyped-defs
import contextlib

import torch

class NullContextManager(torch.nn.Module):
    # [CLASS DOC] Stateless example module computing inside a no-op context.
    """
    Null context manager in Python will be traced out.
    """

    def forward(self, x):
        # [FUNCTION DOC] Enters a null context and returns sine of x plus cosine of x.
        """
        Null context manager in Python will be traced out.
        """
        ctx = contextlib.nullcontext()
        with ctx:
            return x.sin() + x.cos()

example_args = (torch.randn(3, 2),)
tags = {"python.context-manager"}
model = NullContextManager()
