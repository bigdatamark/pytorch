# [MODULE DOC] Reveal-type sample checked by the typing test driver: it builds a ModuleList from
# two distinct Module subclasses, calls children on its first element (a bare comparison whose
# value is discarded, with a noqa for the useless-comparison lint), and reveals the type of the
# list, which must be reported as ModuleList per the expected-output marker on that line. Runs
# at import time if executed.
import torch


# ModuleList with elements of type Module
class FooModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Empty nn.Module subclass used as one of the element types stored in
    # the ModuleList.
    pass


class BarModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Second empty nn.Module subclass, distinct from the first one, so the
    # list holds heterogeneous module types.
    pass


ml: torch.nn.ModuleList = torch.nn.ModuleList([FooModule(), BarModule()])
ml[0].children() == []  # noqa: B015
reveal_type(ml)  # E: {ModuleList}
