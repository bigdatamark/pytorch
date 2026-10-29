# [MODULE DOC] Defines the BackwardState marker class used by Dynamo, AOTAutograd and compiled
# autograd to carry forward-pass hooks into the backward pass. Only imports torch.fx.
import torch.fx


class BackwardState:
    # [CLASS DOC] Marker/container type for passing Python hooks from forward to backward. It is
    # created by Dynamo as an empty object that is an input to the forward graph, populated
    # with attributes only after the forward graph is finished, and becomes an input of the
    # backward graph. At most one may exist per graph. Its only declared attribute is the
    # proxy that represents it while tracing; no methods are defined.
    """
    BackwardState is used to pass Python hooks from the forwards pass
    into the backwards pass in Dynamo+Compiled Autograd.

    It is created by TorchDynamo and has special handling there.
    Dynamo will pass an empty BackwardState to the forwards, then populate
    members on it (via setattr) only after the forwards graph is finished.
    Later on, in CompileAutograd we will inline and add the needed guards
    on the BackwardState.

    BackwardState is identified and has special handling in AOTAutograd.
    During AOTAutograd:
        1) BackwardState is an input to the forwards graph
        2) It must only be used in the backwards
        3) It will be empty in the forwards
        4) In the forwards we add a wrapper to save it
        5) In the backwards it becomes an input
        6) There can only be one per graph

    BackwardState requires CompiledAutograd.
    """

    proxy: torch.fx.Proxy
