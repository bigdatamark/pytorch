# mypy: ignore-errors
# [MODULE DOC]
# Small nn.Module models shared by distributed tests (FSDP, composable APIs, wrapping policies).
# Provides UnitModule, CompositeModel, UnitParamModule, CompositeParamModel, FakeSequential and
# NestedSequentialModel; each takes a device and builds its layers directly on that device.
# No import-time side effects beyond importing torch.

# Owner(s): ["oncall: distributed"]


import torch
import torch.nn as nn


class UnitModule(nn.Module):
    # [CLASS DOC] Basic building-block model: a linear layer, a ReLU/Linear/ReLU sequential,
    # then a second linear layer, all 100x100 and created on the given device.
    # Attributes: l1, seq, l2. Used as a wrappable unit inside larger test models.
    def __init__(self, device: torch.device):
        # [FUNCTION DOC] Builds the layers l1, seq and l2 (all 100 wide) on the given device.
        # Args: device is the torch device on which parameters are allocated.
        super().__init__()
        self.l1 = nn.Linear(100, 100, device=device)
        self.seq = nn.Sequential(
            nn.ReLU(),
            nn.Linear(100, 100, device=device),
            nn.ReLU(),
        )
        self.l2 = nn.Linear(100, 100, device=device)

    def forward(self, x):
        # [FUNCTION DOC] Applies l1, then seq, then l2 to the input and returns the result.
        return self.l2(self.seq(self.l1(x)))


class CompositeModel(nn.Module):
    # [CLASS DOC] Model with two nested UnitModule children between two linear layers.
    # Attributes: l1, u1, u2, l2. Has no direct parameters other than inside submodules.
    def __init__(self, device: torch.device):
        # [FUNCTION DOC] Builds l1, two UnitModules u1 and u2, and l2 on the given device.
        super().__init__()
        self.l1 = nn.Linear(100, 100, device=device)
        self.u1 = UnitModule(device)
        self.u2 = UnitModule(device)
        self.l2 = nn.Linear(100, 100, device=device)

    def forward(self, x):
        # [FUNCTION DOC] Runs the input through l1, u1, u2 and l2 in order; returns the output.
        return self.l2(self.u2(self.u1(self.l1(x))))


class UnitParamModule(nn.Module):
    # [CLASS DOC] Unit model like UnitModule but with a directly owned parameter p (100x100).
    # Attributes: l, seq, p. The final step is a matrix multiply with p.
    def __init__(self, device: torch.device):
        # [FUNCTION DOC] Builds linear layer l, sequential seq, and a randomly initialized
        # 100x100 parameter p, all on the given device.
        super().__init__()
        self.l = nn.Linear(100, 100, device=device)
        self.seq = nn.Sequential(
            nn.ReLU(),
            nn.Linear(100, 100, device=device),
            nn.ReLU(),
        )
        self.p = nn.Parameter(torch.randn((100, 100), device=device))

    def forward(self, x):
        # [FUNCTION DOC] Applies l then seq to the input and matrix-multiplies by p.
        return torch.mm(self.seq(self.l(x)), self.p)


class CompositeParamModel(nn.Module):
    # [CLASS DOC] Model mixing a linear layer, two UnitModule children, a direct parameter p
    # and a persistent buffer named "buffer". Used to test parameter/buffer handling when
    # wrapping modules.
    def __init__(self, device: torch.device):
        # [FUNCTION DOC] Builds l, u1, u2, parameter p and a persistent random buffer, all
        # 100 wide and on the given device.
        super().__init__()
        self.l = nn.Linear(100, 100, device=device)
        self.u1 = UnitModule(device)
        self.u2 = UnitModule(device)
        self.p = nn.Parameter(torch.randn((100, 100), device=device))
        self.register_buffer(
            "buffer", torch.randn((100, 100), device=device), persistent=True
        )

    def forward(self, x):
        # [FUNCTION DOC] Runs the input through l, u1 and u2, then matrix-multiplies by p.
        # The registered buffer is not used in the computation.
        a = self.u2(self.u1(self.l(x)))
        b = self.p
        return torch.mm(a, b)


class FakeSequential(nn.Module):
    # [CLASS DOC] Stand-in for nn.Sequential that is a distinct module type, so a wrap policy
    # targeting nn.Sequential does not match it. Stores its children in a plain list
    # (_module_sequence), which means they are NOT registered as submodules.
    # KNOWN ISSUE: because children sit in a plain list, their parameters are not visible
    # to parameters(), state_dict() or device moves of this module.
    # Define this class to achieve a desired nested wrapping using the module
    # wrap policy with `nn.Sequential`
    def __init__(self, *modules: tuple[nn.Module, ...]) -> None:
        # [FUNCTION DOC] Stores the given modules, in order, in a plain list attribute.
        # KNOWN ISSUE: the type annotation says each argument is a tuple, but callers pass
        # individual modules.
        super().__init__()
        self._module_sequence = list(modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] Applies each stored module to the running value in order; returns it.
        for module in self._module_sequence:
            x = module(x)
        return x


class NestedSequentialModel(nn.Module):
    # [CLASS DOC] Model with two nn.Sequential blocks (seq1, seq2) holding nested
    # FakeSequential modules and a linear layer lin between them; small widths (1 to 4).
    # Designed to exercise module traversal order in wrapping tests.
    # KNOWN ISSUE: it defines no usable forward method (see the note inside __init__).
    def __init__(self, device: torch.device) -> None:
        # [FUNCTION DOC] Builds seq1, lin and seq2 on the given device. Also defines a local
        # function named forward that is never used or attached to the class.
        super().__init__()
        # This nested structure exercises traversal order to catch differences
        # between valid traversals (e.g. BFS and DFS variations).
        self.seq1 = nn.Sequential(
            nn.Linear(1, 1, device=device),
            FakeSequential(
                nn.Linear(1, 1, device=device),
                nn.ReLU(),
                FakeSequential(
                    nn.Linear(1, 1, device=device),
                ),
                nn.ReLU(),
            ),
            nn.Linear(1, 2, device=device),
        )
        self.lin = nn.Linear(2, 2, device=device)
        self.seq2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2, 3, device=device),
            FakeSequential(
                nn.Linear(3, 2, bias=False, device=device),
                nn.Linear(2, 4, bias=False, device=device),
            ),
        )

        # FIXME(rec): forward() is not a method, it's a local function inside __init__
        # that is never used. It should probably be outdented by four spaces, or removed.
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Intended forward (seq1, then lin, then seq2) but dead code.
            # KNOWN ISSUE: nested inside __init__, never called, so the model has no forward.
            return self.seq2(self.lin(self.seq1(x)))
