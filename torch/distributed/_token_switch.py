# mypy: allow-untyped-defs
# [MODULE DOC] Token routing switch abstraction for expert-parallel dispatch and
# combine, with an NCCL EP backed implementation. Contains the Routing record,
# autograd functions for dispatch and combine, the abstract TokenSwitch,
# TokenSwitchNCCL, and helpers that locate and prepare the optional nccl4py package
# and import the torch._nccl_ep extension. Import-time side effect: only creates a
# module-level keepalive list; the extension itself is imported lazily when a
# TokenSwitchNCCL is built. Helpers may mutate environment variables (NCCL_EP_HOME,
# NCCL_HOME) and preload a shared library.
from __future__ import annotations

import abc
import ctypes
import importlib.util
import os
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup


# Keeps preloaded shared libraries resident for the process lifetime.
_NCCL_EP_KEEPALIVE: list[ctypes.CDLL] = []


def _find_pkg_dir(name: str) -> str | None:
    # [FUNCTION DOC] HELPER: Returns the first search location directory of an installed
    # package found by name without importing it, or None if the package (or its parent
    # namespace) is absent or is not a package.
    # Locate an installed package's directory without importing it. find_spec
    # raises (rather than returning None) when a dotted name's parent namespace
    # is absent, so guard that.
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return spec.submodule_search_locations[0]


def _prepare_nccl4py() -> None:
    # [FUNCTION DOC] HELPER: Prepares the environment for a system-NCCL build whose EP
    # extension links against the nccl4py library. Raises ImportError if nccl4py is not
    # installed or its EP JIT headers are missing (unless NCCL_EP_HOME is already set).
    # Side effects: sets NCCL_EP_HOME, sets NCCL_HOME by default when the nvidia.nccl
    # wheel has headers, and loads libnccl_ep by path (kept alive in a module-level list)
    # so its SONAME resolves.
    # Dynamic (USE_SYSTEM_NCCL=ON / wheel) build only: the extension NEEDED-links
    # libnccl_ep.so, which the nccl4py wheel provides at runtime. Point nccl-ep's
    # JIT at nccl4py's EP headers (NCCL_EP_HOME) and the nvidia.nccl wheel's NCCL
    # headers (NCCL_HOME), and make libnccl_ep resolvable, before importing the
    # extension (its libnccl dependency is already loaded by torch). The static
    # build bakes all of this in and never reaches here.
    nccl_pkg = _find_pkg_dir("nccl")
    if nccl_pkg is None:
        raise ImportError(
            "TokenSwitchNCCL needs the 'nccl4py' package for this build's "
            "libnccl_ep.so and runtime JIT headers. Install it with "
            "`pip install nccl4py==0.3.1`."
        )
    ep_dir = os.path.join(nccl_pkg, "ep")
    if "NCCL_EP_HOME" not in os.environ:
        ep_headers = os.path.join(ep_dir, "include", "nccl_ep")
        if not os.path.isdir(ep_headers):
            raise ImportError(
                f"nccl4py at {nccl_pkg} is missing the EP JIT headers expected "
                f"at {ep_headers}; reinstall nccl4py or set NCCL_EP_HOME."
            )
        os.environ["NCCL_EP_HOME"] = ep_dir

    # Point the JIT at NCCL headers. libnccl.so itself needs no preload here: a
    # USE_SYSTEM_NCCL=ON torch (the only build that reaches this path) already
    # dynamically loaded the system libnccl, so libnccl_ep.so's NEEDED
    # libnccl.so.2 resolves to that already-loaded copy.
    nvidia_nccl = _find_pkg_dir("nvidia.nccl")
    if nvidia_nccl is not None and os.path.isdir(os.path.join(nvidia_nccl, "include")):
        os.environ.setdefault("NCCL_HOME", nvidia_nccl)

    # nccl4py ships libnccl_ep unversioned (libnccl_ep.so) while the extension's
    # NEEDED entry is its SONAME libnccl_ep.so.0; load it by path to register
    # that SONAME for the import below.
    lib = os.path.join(ep_dir, "lib", "libnccl_ep.so")
    if os.path.isfile(lib):
        _NCCL_EP_KEEPALIVE.append(ctypes.CDLL(lib, mode=ctypes.RTLD_LOCAL))


def _import_nccl_ep() -> Any:
    # [FUNCTION DOC] HELPER: Returns the torch._nccl_ep extension module. Tries a direct
    # import first (static build); on ImportError calls the nccl4py preparation and
    # retries. Raises ImportError with a descriptive message if the extension is
    # unavailable, chaining the original error. May propagate the ImportError from the
    # preparation step.
    # The EP bindings live in the optional torch._nccl_ep extension (USE_NCCL_EP).
    # A USE_SYSTEM_NCCL=OFF build links libnccl_ep statically and bakes its JIT
    # header paths, so the extension imports directly -- self-contained, no
    # nccl4py. A USE_SYSTEM_NCCL=ON build NEEDED-links libnccl_ep from the nccl4py
    # wheel, so the first import fails until nccl4py is set up; fall back to that
    # and retry.
    try:
        # pyrefly: ignore [missing-import]  # built only with USE_NCCL_EP
        import torch._nccl_ep as _ep

        return _ep
    except ImportError:
        pass

    _prepare_nccl4py()
    try:
        # pyrefly: ignore [missing-import]  # built only with USE_NCCL_EP
        import torch._nccl_ep as _ep
    except ImportError as e:
        raise ImportError(
            "torch._nccl_ep is unavailable; this PyTorch was not built with "
            "USE_NCCL_EP (or, for a USE_SYSTEM_NCCL=ON build, the nccl4py wheel "
            "is missing)."
        ) from e

    return _ep


@dataclass(frozen=True, slots=True)
class Routing:
    # [CLASS DOC] Immutable record of a routing plan: an opaque backend handle and the
    # top-k expert index tensor it was created from.
    handle: object
    topk_idx: torch.Tensor


class _DispatchAutograd(torch.autograd.Function):
    # [CLASS DOC] HELPER: Autograd function for TokenSwitch dispatch. Forward allocates
    # zeroed output buffers sized for the maximum received tokens and runs the switch
    # dispatch. Backward treats the incoming token gradient as expert-side data and runs
    # the switch combine to send it back; weights and indexes get no gradient.
    @staticmethod
    def forward(
        ctx: Any,
        ts: TokenSwitch,
        routing: Routing,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
        max_recv_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # [FUNCTION DOC] HELPER: Forward pass. Takes the switch, routing, tokens (N by H),
        # top-k weights (N by K) and the maximum receive count. Allocates zero-filled output
        # tokens, weights and indices with that many rows, calls the switch dispatch, and
        # saves the switch, routing and token shape on ctx. Returns the three output tensors.
        _N, H = tokens.shape
        K = topk_weights.shape[1]
        out_tokens = tokens.new_zeros(max_recv_tokens, H)
        out_topk_weights = topk_weights.new_zeros(max_recv_tokens, K)
        out_topk_idx = routing.topk_idx.new_zeros(max_recv_tokens, K)
        ts._dispatch(
            routing, tokens, topk_weights, out_tokens, out_topk_weights, out_topk_idx
        )
        ctx.ts = ts
        ctx.routing = routing
        ctx.tokens_shape = tokens.shape
        return out_tokens, out_topk_weights, out_topk_idx

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(
        ctx: Any,
        grad_out_tokens: torch.Tensor,
        grad_out_topk_weights: torch.Tensor,
        grad_out_topk_idx: torch.Tensor,
    ) -> tuple[None, None, torch.Tensor, None, None]:
        # [FUNCTION DOC] HELPER: Backward pass. Combines the (contiguous) token gradient
        # through the switch into a zero tensor of the original token shape and returns it as
        # the gradient for tokens only; None for the switch, routing, weights and max count.
        # Gradients of the weight and index outputs are ignored.
        grad_tokens = grad_out_tokens.new_zeros(ctx.tokens_shape)
        ctx.ts._combine(ctx.routing, grad_out_tokens.contiguous(), grad_tokens)
        return None, None, grad_tokens, None, None


class _CombineAutograd(torch.autograd.Function):
    # [CLASS DOC] HELPER: Autograd function for TokenSwitch combine. Forward gathers
    # expert outputs to token order; backward runs a dispatch of the output gradient to
    # the experts.
    @staticmethod
    def forward(
        ctx: Any,
        ts: TokenSwitch,
        routing: Routing,
        expert_tokens: torch.Tensor,
    ) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Forward pass. Allocates a zeroed (num tokens by hidden)
        # output from the routing index shape, runs the switch combine from the expert tokens,
        # saves the switch, routing, expert shape and dtype and top-k on ctx, and returns the
        # output.
        N = routing.topk_idx.shape[0]
        H = expert_tokens.shape[1]
        out_tokens = expert_tokens.new_zeros(N, H)
        ts._combine(routing, expert_tokens, out_tokens)
        ctx.ts = ts
        ctx.routing = routing
        ctx.expert_shape = expert_tokens.shape
        ctx.expert_dtype = expert_tokens.dtype
        ctx.top_k = routing.topk_idx.shape[1]
        return out_tokens

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(
        ctx: Any, grad_out_tokens: torch.Tensor
    ) -> tuple[None, None, torch.Tensor]:
        # [FUNCTION DOC] HELPER: Backward pass. Runs a switch dispatch of the output gradient
        # (converted to the expert dtype) into a full-size buffer of max receive tokens rows,
        # using dummy weight buffers, since the backend needs full-size output; then slices to
        # the original expert row count and returns it as the gradient for expert tokens (None
        # for the other inputs).
        M, H = ctx.expert_shape
        N = grad_out_tokens.shape[0]
        K = ctx.top_k
        dtype = ctx.expert_dtype
        # ncclEpDispatch requires the output buffer sized to the group's
        # max_recv_tokens_per_rank, regardless of what shape expert_tokens had
        # in forward (often a slice like out_tokens[:M]). Allocate full-size,
        # run dispatch, then slice to ctx.expert_shape so the returned grad
        # matches the input that produced it.
        max_recv = ctx.ts._max_recv_tokens_per_rank
        grad_expert_full = grad_out_tokens.new_zeros(max_recv, H).to(dtype)
        dummy_weights = grad_out_tokens.new_zeros(N, K, dtype=torch.float32)
        dummy_out_weights = grad_out_tokens.new_zeros(max_recv, K, dtype=torch.float32)
        dummy_out_idx = ctx.routing.topk_idx.new_zeros(max_recv, K)
        ctx.ts._dispatch(
            ctx.routing,
            grad_out_tokens.to(dtype).contiguous(),
            dummy_weights,
            grad_expert_full,
            dummy_out_weights,
            dummy_out_idx,
        )
        return None, None, grad_expert_full[:M].contiguous()


class TokenSwitch(abc.ABC):
    # [CLASS DOC] Abstract base for a token routing switch used for expert-parallel
    # dispatch and combine. Subclasses implement create_routing and the private _dispatch
    # and _combine; the public dispatch and combine methods add optional output buffers
    # and autograd support. Typical flow: create routing, dispatch tokens to experts,
    # combine expert outputs.
    """Abstract token routing switch (e.g. expert-parallel dispatch / combine).

    Typical usage: :meth:`create_routing`, then :meth:`dispatch` / :meth:`combine`.
    """

    @abc.abstractmethod
    def create_routing(
        self,
        topk_idx: torch.Tensor,
        per_expert_token_counts: torch.Tensor | None = None,
    ) -> Routing:
        # [FUNCTION DOC] Abstract: creates a Routing for the current phase from top-k expert
        # indices and an optional int32 buffer receiving per-expert token counts. Base
        # implementation raises NotImplementedError.
        """Create expert routing for the current phase (e.g. top-k indices).

        ``per_expert_token_counts`` is optional 1D int32, length >= local experts:
        output buffer for per-expert receive counts (NCCL EP ``RECV_EXPERT_COUNTER``).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _dispatch(
        self,
        routing: Routing,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
        out_tokens: torch.Tensor,
        out_topk_weights: torch.Tensor,
        out_topk_idx: torch.Tensor,
    ) -> None:
        # [FUNCTION DOC] HELPER: Abstract: writes dispatched tokens, weights and indices into
        # the given output buffers according to the routing. Base raises NotImplementedError.
        raise NotImplementedError

    @abc.abstractmethod
    def _combine(
        self,
        routing: Routing,
        expert_tokens: torch.Tensor,
        out_tokens: torch.Tensor,
    ) -> None:
        # [FUNCTION DOC] HELPER: Abstract: writes expert outputs gathered back in token order
        # into the output buffer. Base raises NotImplementedError.
        raise NotImplementedError

    def dispatch(
        self,
        routing: Routing,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
        max_recv_tokens: int | None = None,
        *,
        out: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # [FUNCTION DOC] Routes tokens to experts. If out (a triple of output buffers) is
        # given, dispatches directly into them and returns them, without autograd support.
        # Otherwise requires max_recv_tokens (raises ValueError if missing), allocates the
        # buffers and returns a triple (tokens, weights, indices) through an autograd
        # function; the weights receive no gradient.
        """Route tokens to experts.

        With ``out=(out_tokens, out_topk_weights, out_topk_idx)``: writes to the provided
        buffers and returns them; no autograd support.
        Without ``out``: allocates output buffers and returns
        ``(out_tokens, out_topk_weights, out_topk_idx)`` with autograd support.
        ``max_recv_tokens`` is required when ``out`` is not provided.
        ``topk_weights`` receives no gradient (routing metadata).
        """
        if out is not None:
            self._dispatch(routing, tokens, topk_weights, *out)
            return out
        if max_recv_tokens is None:
            raise ValueError("max_recv_tokens is required when out= is not provided")
        return _DispatchAutograd.apply(
            self, routing, tokens, topk_weights, max_recv_tokens
        )  # type: ignore[return-value]

    def combine(
        self,
        routing: Routing,
        expert_tokens: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # [FUNCTION DOC] Gathers expert outputs back to token order. If out is given, writes
        # into it and returns it, without autograd support; otherwise allocates the output and
        # returns it through an autograd function so gradients flow.
        """Gather expert outputs back to token order.

        With ``out=out_tokens``: writes to the provided buffer and returns it;
        no autograd support.
        Without ``out``: allocates an output buffer and returns it with autograd support.
        """
        if out is not None:
            self._combine(routing, expert_tokens, out)
            return out
        return _CombineAutograd.apply(self, routing, expert_tokens)  # type: ignore[return-value]


class TokenSwitchNCCL(TokenSwitch):
    # [CLASS DOC] TokenSwitch backed by NCCL EP. Holds the loaded extension module, the
    # maximum receive count per rank and the created EP group; routing handles are created
    # per phase. Construction can raise ImportError if the extension is unavailable.
    """Token switch backed by NCCL EP (:func:`ncclEpCreateGroup` / dispatch / combine)."""

    def __init__(
        self,
        process_group: ProcessGroup,
        num_experts: int,
        max_dispatch_tokens_per_rank: int,
        max_recv_tokens_per_rank: int,
        max_token_bytes: int,
    ) -> None:
        # [FUNCTION DOC] Imports the NCCL EP extension (see the import helper) and creates the
        # EP group from the process group, number of experts, per-rank dispatch and receive
        # token limits and the maximum token size in bytes. Stores the extension, the receive
        # limit and the group. Raises ImportError if unavailable.
        self._ep = _import_nccl_ep()
        self._max_recv_tokens_per_rank = max_recv_tokens_per_rank
        self._group = self._ep._NcclEpGroup.create(
            process_group,
            num_experts,
            max_dispatch_tokens_per_rank,
            max_recv_tokens_per_rank,
            max_token_bytes,
        )

    def create_routing(
        self,
        topk_idx: torch.Tensor,
        per_expert_token_counts: torch.Tensor | None = None,
    ) -> Routing:
        # [FUNCTION DOC] Creates a backend routing handle from the group, top-k indices and
        # optional per-expert count buffer, and returns a Routing holding the handle and
        # indices.
        """Create expert routing for this phase; pass to :meth:`dispatch` / :meth:`combine`."""
        handle = self._ep._NcclEpHandle.create(
            self._group,
            topk_idx,
            per_expert_token_counts,
        )
        return Routing(handle=handle, topk_idx=topk_idx)

    def _dispatch(
        self,
        routing: Routing,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
        out_tokens: torch.Tensor,
        out_topk_weights: torch.Tensor,
        out_topk_idx: torch.Tensor,
    ) -> None:
        # [FUNCTION DOC] HELPER: Calls the extension dispatch with the routing handle, inputs
        # and the three output buffers (written in place).
        self._ep._nccl_ep_dispatch(
            routing.handle,
            tokens,
            topk_weights,
            out_tokens,
            out_topk_weights,
            out_topk_idx,
        )

    def _combine(
        self,
        routing: Routing,
        expert_tokens: torch.Tensor,
        out_tokens: torch.Tensor,
    ) -> None:
        # [FUNCTION DOC] HELPER: Calls the extension combine with the routing handle, expert
        # tokens and output buffer (written in place).
        self._ep._nccl_ep_combine(
            routing.handle,
            expert_tokens,
            out_tokens,
        )
