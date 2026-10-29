# [MODULE DOC] Standalone compilation API for Inductor: compile a GraphModule with example inputs into
# a CompiledArtifact that can be called, saved (binary file or unpacked directory) and loaded later
# without recompiling. Defines the CompiledArtifact base class, its cache-based and AOT-bundled
# implementations, helpers to resolve fake mode and dynamic shapes, and the standalone_compile entry
# point. Import-time side effects: creates the module logger.
from __future__ import annotations

import contextlib
import copy
import itertools
import logging
import os
import pickle
import shutil
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Literal, TYPE_CHECKING


DynamicShapesType = Literal["from_example_inputs", "from_tracing_context", "from_graph"]

import torch.fx
from torch._dynamo.aot_compile_types import BundledAOTAutogradSerializableCallable
from torch._dynamo.utils import dynamo_timed
from torch._inductor.cpp_builder import normalize_path_separator
from torch._inductor.cudagraph_utils import BoxedDeviceIndex
from torch._inductor.runtime.cache_dir_utils import temporary_cache_dir
from torch._inductor.utils import BoxedBool, InputType
from torch._subclasses import FakeTensorMode
from torch._subclasses.fake_tensor import maybe_get_fake_mode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.fx.graph_module import _share_torchbind_and_process_group_on_deepcopy

from . import config
from ._functionalize_collectives import (
    _functionalize_inplace_collectives,
    _unbox_process_group_torchbinds,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from torch.compiler._cache import CacheInfo
    from torch.fx import GraphModule

    from .output_code import OutputCode


log = logging.getLogger(__name__)


class CompiledArtifact(ABC):
    # [CLASS DOC] Abstract base for a compiled, invokable, serializable artifact. Holds the compiled
    # callable and optional (bytes, CacheInfo) artifacts. Subclasses implement __call__ and save
    # (abstract, one-line stubs); load dispatches on format and file header.
    """
    CompiledArtifact class represents the inductor cache artifacts that
    can be invoked in order to avoid repeated compilation.

    CompiledArtifact can be obtained by calling standalone_compile(gm, example_inputs)
    to create a fresh CompiledArtifact from a GraphModule and example inputs.

    Later this CompiledArtifact can be saved to disk, either as a binary or unpacked
    into the provided folder via the CompiledArtifact.save function.

    CompiledArtifact.load provides a way to create a CompiledArtifact from the
    binary or unpacked data.

    Finally, the CompiledArtifact can be invoked via the __call__ method
    to execute the cached artifact.
    """

    def __init__(
        self,
        compiled_fn: Callable[..., Any],
        artifacts: tuple[bytes, CacheInfo] | None,
    ):
        # [FUNCTION DOC] Store the compiled callable and the optional cache artifacts tuple.
        self._compiled_fn = compiled_fn
        self._artifacts = artifacts

    @abstractmethod
    def __call__(self, *args: Any) -> Any: ...

    @abstractmethod
    def save(
        self, *, path: str, format: Literal["binary", "unpacked"] = "binary"
    ) -> None: ...

    @staticmethod
    def load(
        *, path: str, format: Literal["binary", "unpacked"] = "binary"
    ) -> CompiledArtifact:
        # [FUNCTION DOC] Load an artifact from path. Format unpacked goes to the cache implementation. Binary
        # format reads the file, checks the header: AOT header (verifies torch key, deserializes the AOT
        # artifact) or cache header (verifies torch key, reads key and bytes, loads cache artifacts and
        # builds the artifact from the key). Raises AssertionError on torch key mismatch, trailing data or
        # bad format, RuntimeError on an unknown header. Reads the file and populates caches.
        if format == "unpacked":
            # If format is unpacked, it must be a CacheCompiledArtifact
            return CacheCompiledArtifact.load(path=path, format=format)

        if format != "binary":
            raise AssertionError(f"expected format == 'binary', got {format}")
        with open(path, "rb") as file:
            from torch.utils._appending_byte_serializer import BytesReader

            from .codecache import torch_key

            result_bytes = file.read()
            reader = BytesReader(result_bytes)
            header = reader.read_bytes()
            if header == AOTCompiledArtifact.AOT_HEADER:
                if reader.read_bytes() != torch_key():
                    raise AssertionError("torch_key mismatch in serialized artifact")
                artifact = reader.read_bytes()
                if not reader.is_finished():
                    raise AssertionError("expected reader to be finished")
                return AOTCompiledArtifact.deserialize(artifact)
            # Otherwise, it's in the CacheCompiledArtifact format
            elif header == CacheCompiledArtifact.CACHE_HEADER:
                if reader.read_bytes() != torch_key():
                    raise AssertionError("torch_key mismatch in serialized artifact")
                key = reader.read_str()
                artifact_bytes = reader.read_bytes()
                if not reader.is_finished():
                    raise AssertionError("expected reader to be finished")
                torch.compiler.load_cache_artifacts(artifact_bytes)
                return CacheCompiledArtifact._load_impl(nullcontext(), key)
            else:
                raise RuntimeError(
                    "Invalid header, expected CacheCompiledArtifact or AOTCompiledArtifact, got: "
                    + header.decode("utf-8")
                )


class CacheCompiledArtifact(CompiledArtifact):
    # [CLASS DOC] Artifact backed by torch.compiler cache artifacts (the AOT autograd cache entry). Wraps the
    # compiled callable plus (bytes, CacheInfo). Can be saved as a binary blob or unpacked into a cache
    # directory, and reloaded by looking up the autograd cache key. CACHE_HEADER marks the binary format.
    """
    CompiledArtifact that depends on torch.compiler.save_cache_artifacts
    """

    CACHE_HEADER = bytes("CacheCompiledArtifact", "utf-8")

    def __init__(
        self,
        compiled_fn: Callable[..., Any],
        artifacts: tuple[bytes, CacheInfo] | None,
    ):
        # [FUNCTION DOC] Store the compiled callable and the optional cache artifacts tuple.
        self._compiled_fn = compiled_fn
        self._artifacts = artifacts

    def __call__(self, *args: Any) -> Any:
        # [FUNCTION DOC] Invoke the compiled callable with the given positional args and return its result.
        return self._compiled_fn(*args)

    def is_saveable(self) -> bool:
        # [FUNCTION DOC] True only if artifacts exist and exactly one aot_autograd artifact was recorded.
        if self._artifacts is None:
            return False
        _, cache_info = self._artifacts
        # 0 means nothing was saved
        # >1 means multiple artifacts were saved, which is concerning
        # (we only expect one)
        return len(cache_info.aot_autograd_artifacts) == 1

    def _validate_and_unpack(self) -> tuple[bytes, str]:
        # [FUNCTION DOC] HELPER: Validate artifacts and return (artifact bytes, autograd cache key). Raises
        # RuntimeError if there are no artifacts or no aot_autograd artifacts, AssertionError if more than
        # one.
        """Validate the cached artifact, returning ``(artifact_bytes, key)``.

        Single source of the None / empty / multiple aot_autograd_artifacts checks,
        shared by ``_to_binary_bytes`` and ``save``'s unpacked branch. Messages are
        neutral (not tied to ``save``) because ``_to_binary_bytes`` callers obtain the
        bytes without going through ``save``.
        """
        if self._artifacts is None:
            raise RuntimeError("CompiledArtifact has no artifact to serialize")
        artifact_bytes, cache_info = self._artifacts
        if len(cache_info.aot_autograd_artifacts) == 0:
            raise RuntimeError(
                f"CompiledArtifact has no aot_autograd artifacts to serialize. This "
                f"likely means there was something that was not serializable in the graph "
                f"passed to standalone_compile. This can generally be fixed by ensuring "
                f"that your model only uses constructs that are serializable. {cache_info}"
            )
        if len(cache_info.aot_autograd_artifacts) > 1:
            raise AssertionError(
                f"CompiledArtifact has more than one aot_autograd artifact but we only "
                f"expected one. {cache_info}"
            )
        return artifact_bytes, cache_info.aot_autograd_artifacts[0]

    def _to_binary_bytes(self) -> bytes:
        # [FUNCTION DOC] HELPER: Serialize to the binary format: cache header, torch key, cache key string, then
        # artifact bytes. Validates first (may raise). Returns the bytes without writing to disk.
        """Serialize this artifact to the in-memory ``binary`` byte format.

        Produces exactly the bytes that ``save(format="binary")`` writes to disk, so
        callers that only need the bytes (rather than a file on disk) can reuse this
        without going through ``save``. The byte layout is the one
        ``load(format="binary")`` reads back: header, ``torch_key``, the autograd-cache
        ``key`` string, then the opaque ``artifact_bytes``.
        """
        artifact_bytes, key = self._validate_and_unpack()

        from torch.utils._appending_byte_serializer import BytesWriter

        from .codecache import torch_key

        writer = BytesWriter()
        writer.write_bytes(CacheCompiledArtifact.CACHE_HEADER)
        writer.write_bytes(torch_key())
        writer.write_str(key)
        writer.write_bytes(artifact_bytes)
        return writer.to_bytes()

    def save(
        self, *, path: str, format: Literal["binary", "unpacked"] = "binary"
    ) -> None:
        # [FUNCTION DOC] Save the artifact (timed). Binary: asserts path is not a directory and atomically writes
        # the serialized bytes. Unpacked: validates, removes an existing directory at path (asserts it is a
        # directory), loads cache artifacts into a temporary cache dir at path, then writes each inductor
        # output code to disk and logs its location. Raises AssertionError on bad format or path.
        with dynamo_timed("CompiledArtifact.save"):
            if format == "binary":
                # can't assert that it is a file since it might not exist yet
                if os.path.isdir(path):
                    raise AssertionError(f"expected path to not be a dir: {path}")

                from torch._inductor.codecache import write_atomic

                # ``_to_binary_bytes`` is the single source of the None / empty /
                # multiple aot_autograd_artifacts validation, so the binary branch
                # does not repeat those checks here.
                write_atomic(path, self._to_binary_bytes())
            else:
                if format != "unpacked":
                    raise AssertionError(f"expected format == 'unpacked', got {format}")
                # Same None / empty / multiple validation as the binary branch, shared via
                # _validate_and_unpack; the unpacked branch needs only artifact_bytes.
                artifact_bytes, _key = self._validate_and_unpack()
                if os.path.exists(path):
                    if not os.path.isdir(path):
                        raise AssertionError(f"expected path to be a dir: {path}")
                    shutil.rmtree(path, ignore_errors=True)

                from .codecache import FxGraphCache

                with temporary_cache_dir(path):
                    # This function unpacks the cache artifacts to disk
                    loaded_cache_info = torch.compiler.load_cache_artifacts(
                        artifact_bytes
                    )
                    if loaded_cache_info is None:
                        raise AssertionError(
                            "expected loaded_cache_info to not be None"
                        )
                    # Now write all the output_code artifacts to disk so that
                    # they can be inspected and modified
                    for key in loaded_cache_info.inductor_artifacts:
                        subdir = FxGraphCache._get_tmp_dir_for_key(key)
                        if not os.path.exists(subdir):
                            raise AssertionError(f"expected subdir to exist: {subdir}")
                        for path in sorted(os.listdir(subdir)):
                            with open(os.path.join(subdir, path), "rb") as f:
                                graph = pickle.load(f)
                            output_file = graph.write_to_disk()
                            log.info("Output code written to: %s", output_file)

    @staticmethod
    def _load_impl(
        cache_dir_ctx: AbstractContextManager[Any], key: str
    ) -> CompiledArtifact:
        # [FUNCTION DOC] HELPER: Build an artifact from an autograd cache key. Within the given cache-dir context,
        # with dynamic shape guards skipping and strict autograd cache enabled, looks up the key locally
        # (asserts a hit), then post-compiles the entry under a fresh tracing context and fake mode. Returns
        # a CacheCompiledArtifact wrapping a callable that boxes its args into a list.
        with (
            cache_dir_ctx,
            config.patch(unsafe_skip_cache_dynamic_shape_guards=True),
        ):
            with torch._functorch.config.patch(strict_autograd_cache=True):
                from torch._functorch._aot_autograd.autograd_cache import (
                    AOTAutogradCache,
                )

                result = AOTAutogradCache._lookup(
                    key,
                    local=True,
                    remote=False,
                    args=[],
                    cache_info={},
                    aot_config=None,
                )

            if result is None:
                raise AssertionError(
                    "expected AOTAutogradCache lookup result to not be None"
                )
            (entry, _) = result

            from .compile_fx import _CompileFxKwargs

            fx_config = _CompileFxKwargs(
                cudagraphs=BoxedBool(False),
                boxed_forward_device_index=BoxedDeviceIndex(0),
            )

            context = torch._guards.TracingContext(FakeTensorMode(shape_env=ShapeEnv()))
            with torch._guards.tracing(context):
                compiled_fn = entry.wrap_post_compile(
                    [], entry.sanitized_aot_config, fx_config
                )
        return CacheCompiledArtifact(lambda *args: compiled_fn(list(args)), None)

    @staticmethod
    def _prepare_load(
        *, path: str, format: Literal["binary", "unpacked"] = "binary"
    ) -> tuple[str, AbstractContextManager[Any]]:
        # [FUNCTION DOC] HELPER: Format-specific preparation for load; returns (key, cache-dir context manager).
        # Binary: reads the file, verifies torch key, reads key and bytes, loads the cache artifacts and
        # returns a null context. Unpacked: path must be a directory containing an aotautograd folder with
        # exactly one entry (the key) and returns a temporary cache dir context. Raises AssertionError on
        # invalid layout.
        """
        Do format specific prep and loads, return a context manager and key
        """
        path = normalize_path_separator(path)
        with dynamo_timed("CompiledArtifact.load"):
            if format == "binary":
                # can't assert that it is a file since it might not exist yet
                if os.path.isdir(path):
                    raise AssertionError(f"expected path to not be a dir: {path}")
                with open(path, "rb") as file:
                    artifacts = file.read()
                from torch.utils._appending_byte_serializer import BytesReader

                from .codecache import torch_key

                reader = BytesReader(artifacts)
                if reader.read_bytes() != torch_key():
                    raise AssertionError("torch_key mismatch in serialized artifact")
                key = reader.read_str()
                artifact_bytes = reader.read_bytes()
                if not reader.is_finished():
                    raise AssertionError("expected reader to be finished")

                torch.compiler.load_cache_artifacts(artifact_bytes)
                return key, nullcontext()
            else:
                if format != "unpacked":
                    raise AssertionError(f"expected format == 'unpacked', got {format}")
                if not os.path.isdir(path):
                    raise AssertionError(f"expected path to be a dir: {path}")
                autograd_cache_dir = os.path.join(path, "aotautograd")
                if not os.path.isdir(autograd_cache_dir):
                    raise AssertionError(
                        f"expected autograd_cache_dir to be a dir: {autograd_cache_dir}"
                    )
                files = list(os.listdir(autograd_cache_dir))
                if len(files) != 1:
                    raise AssertionError(f"expected exactly 1 file, got {len(files)}")
                key = files[0]
                cache_dir_ctx = temporary_cache_dir(path)
                return key, cache_dir_ctx

    @staticmethod
    def load(
        *, path: str, format: Literal["binary", "unpacked"] = "binary"
    ) -> CompiledArtifact:
        # [FUNCTION DOC] Load a cache-backed artifact from path in the given format via prepare-load and load-impl.
        key, cache_dir_ctx = CacheCompiledArtifact._prepare_load(
            path=path, format=format
        )
        return CacheCompiledArtifact._load_impl(cache_dir_ctx, key)


class AOTCompiledArtifact(CompiledArtifact):
    # [CLASS DOC] Artifact wrapping a single bundled AOT-autograd serializable callable (inner_fn); always
    # serializable. Supports only the binary format (unpacked raises RuntimeError). AOT_HEADER marks the
    # binary format. No separate cache artifacts are kept.
    """
    Similar to CompiledArtifact, but the object is a single, bundled precompiled function.
    This object is always a serializable callable function.

    This object is essentially a wrapper for BundledAOTAutogradSerializableCallable, which
    is used by torch._dynamo.aot_compile for AOT Precompilation.
    """

    AOT_HEADER = bytes("AOTCompiledArtifact", "utf-8")

    def __init__(
        self,
        compiled_fn: Callable[..., Any],
    ):
        # [FUNCTION DOC] Wrap the compiled callable in a bundled AOT autograd serializable callable and set the
        # artifacts to None.
        self.inner_fn = BundledAOTAutogradSerializableCallable(compiled_fn)
        self._artifacts = (
            None  # We don't need artifacts, the inner object handles everything
        )

    @staticmethod
    def from_bundled_callable(
        bundled_fn: BundledAOTAutogradSerializableCallable,
    ) -> AOTCompiledArtifact:
        # [FUNCTION DOC] Build an AOTCompiledArtifact from an already bundled callable by re-wrapping its
        # compiled function.
        return AOTCompiledArtifact(bundled_fn.compiled_fn)

    def __call__(self, *args: Any) -> Any:
        # [FUNCTION DOC] Invoke the inner bundled callable with the given args.
        return self.inner_fn(*args)

    def save(
        self, *, path: str, format: Literal["binary", "unpacked"] = "binary"
    ) -> None:
        # [FUNCTION DOC] Save as a binary file: AOT header, torch key, serialized bytes, written atomically to
        # path. Raises RuntimeError for the unpacked format.
        if format == "unpacked":
            raise RuntimeError(
                "AOTCompiledArtifact does not support unpacked format yet"
            )
        result_bytes = self.serialize()
        from torch.utils._appending_byte_serializer import BytesWriter

        from .codecache import torch_key

        writer = BytesWriter()
        writer.write_bytes(AOTCompiledArtifact.AOT_HEADER)
        writer.write_bytes(torch_key())
        writer.write_bytes(result_bytes)

        from torch._inductor.codecache import write_atomic

        # Save a sentinel file to indicate that this is AOT
        write_atomic(path, writer.to_bytes())

    def serialize(self) -> bytes:
        # [FUNCTION DOC] Serialize the inner bundled callable to bytes.
        return BundledAOTAutogradSerializableCallable.serialize_compile_artifacts(
            self.inner_fn
        )

    @staticmethod
    def deserialize(result_bytes: bytes) -> AOTCompiledArtifact:
        # [FUNCTION DOC] Deserialize bytes into an AOTCompiledArtifact. Raises AssertionError if the result is
        # not a bundled AOT autograd callable.
        deserialized = (
            BundledAOTAutogradSerializableCallable.deserialize_compile_artifacts(
                result_bytes
            )
        )
        if not isinstance(deserialized, BundledAOTAutogradSerializableCallable):
            raise AssertionError(
                f"expected BundledAOTAutogradSerializableCallable, got {type(deserialized)}"
            )
        return AOTCompiledArtifact.from_bundled_callable(deserialized)

    @staticmethod
    def load(
        *, path: str, format: Literal["binary", "unpacked"] = "binary"
    ) -> CompiledArtifact:
        # [FUNCTION DOC] Load an AOT artifact from a binary file: verifies the AOT header, torch key and no
        # trailing data, then deserializes. Raises RuntimeError for unpacked, AssertionError on mismatch.
        if format == "unpacked":
            raise RuntimeError(
                "AOTCompiledArtifact does not support unpacked format yet"
            )
        with open(path, "rb") as file:
            from torch.utils._appending_byte_serializer import BytesReader

            from .codecache import torch_key

            result_bytes = file.read()
            reader = BytesReader(result_bytes)
            header = reader.read_bytes()
            if header != AOTCompiledArtifact.AOT_HEADER:
                raise AssertionError("expected AOTCompiledArtifact header")
            if reader.read_bytes() != torch_key():
                raise AssertionError("torch_key mismatch in serialized artifact")
            artifact = reader.read_bytes()
            if not reader.is_finished():
                raise AssertionError("expected reader to be finished")
            return AOTCompiledArtifact.deserialize(artifact)


def _resolve_ignore_shape_env(dynamic_shapes: DynamicShapesType):
    # [FUNCTION DOC] HELPER: True when dynamic_shapes is from_example_inputs, telling compile_fx to ignore the
    # shape envs of the ambient context and the graph module.
    # tells compile_fx to ignore the shape_envs on the ambient context
    # and the graph_module.
    return dynamic_shapes == "from_example_inputs"


def _resolve_fake_mode(
    gm: GraphModule,
    dynamic_shapes: DynamicShapesType,
    fake_mode: FakeTensorMode | None = None,
) -> FakeTensorMode:
    # [FUNCTION DOC] HELPER: Pick the FakeTensorMode for a standalone compile. from_example_inputs: use the
    # given fake_mode (must have a ShapeEnv, else ValueError) or make a fresh one. Other modes reject an
    # explicit fake_mode (ValueError). from_tracing_context: take the mode from the current tracing
    # context (asserts it is set). from_graph: search output nodes then all nodes for fake values in
    # example_value or val metadata, skipping non-Node outputs, falling back to a fresh mode. Unknown
    # dynamic_shapes raises ValueError.
    if dynamic_shapes == "from_example_inputs":
        if fake_mode is not None:
            if fake_mode.shape_env is None:
                raise ValueError(
                    "standalone_compile requires `fake_mode` to have a ShapeEnv "
                    'when `dynamic_shapes="from_example_inputs"`.'
                )
            return fake_mode
        return FakeTensorMode(shape_env=ShapeEnv())
    elif fake_mode is not None:
        raise ValueError(
            "standalone_compile only supports passing `fake_mode` when "
            '`dynamic_shapes="from_example_inputs"`.'
        )
    elif dynamic_shapes == "from_tracing_context":
        # Reuse fake_mode from the TracingContext.
        # NB: The TracingContext only exists if we're currently in a torch.compile backend.
        context = torch._guards.TracingContext.get()
        if context.fake_mode is None:
            raise AssertionError("expected TracingContext.fake_mode to not be None")
        return context.fake_mode
    elif dynamic_shapes == "from_graph":
        # Strategy: find a FakeTensor in the graph output, grab its FakeTensorMode.
        # The graph passed to standalone_compile must be an Inductor-approved graph,
        # which means that there is at least one Tensor output and the output node
        # contains a flat list of Tensors.
        last_node = next(iter(reversed(gm.graph.nodes)))
        if last_node.op != "output":
            raise AssertionError(
                f"expected last node op == 'output', got {last_node.op}"
            )
        if len(last_node.args) != 1:
            raise AssertionError(
                f"expected last node to have 1 arg, got {len(last_node.args)}"
            )

        # If gm came from Dynamo, then last_node.args[0] is always a list,
        # even in single-Tensor returns.
        #
        # It's possible to get into a situation where last_node.args[0]
        # is a Node (and not a list!). This happens if you call split_module
        # on the graph. We allow for this case since it is common.
        nodes = (
            [last_node.args[0]]
            if isinstance(last_node.args[0], torch.fx.Node)
            else last_node.args[0]
        )
        # Find a FakeTensor to recover its FakeTensorMode. This resolver was written when
        # standalone_compile was only ever fed Dynamo/AOT graphs, which return a flat list
        # of real Tensor outputs and stash the fake under "example_value". precompile now
        # feeds make_fx graphs too, and those break both assumptions: make_fx stashes the
        # fake under "val" (not "example_value"), and the traced fn's own return value is
        # kept as an output leaf -- so when fn calls .backward() (which returns None) that
        # None sits in the output list as a plain non-Node entry. So skip non-Node output
        # entries (the old code called .meta on the None and crashed), check both metadata
        # keys, and fall back to scanning every node since a make_fx fake may live only on
        # an interior node. maybe_get_fake_mode also unwraps a traceable wrapper subclass.
        output_nodes = [n for n in nodes if isinstance(n, torch.fx.Node)]
        for node in itertools.chain(output_nodes, gm.graph.nodes):
            for key in ("example_value", "val"):
                maybe_fake_mode = maybe_get_fake_mode(node.meta.get(key))
                if maybe_fake_mode is not None:
                    return maybe_fake_mode

        return FakeTensorMode(shape_env=ShapeEnv())
    else:
        raise ValueError(
            f"standalone_compile got unsupported `dynamic_shapes` value: dynamic_shapes={dynamic_shapes}."
        )


@contextlib.contextmanager
def _standalone_context(
    gm: GraphModule,
    dynamic_shapes: DynamicShapesType,
    aot: bool,
    fake_mode: FakeTensorMode | None = None,
):
    # [FUNCTION DOC] HELPER: Context manager for standalone compilation. Resolves the fake mode, then enters a
    # tracing context, a fresh cache artifact manager, autotune-at-compile-time config, and functorch
    # config (bundled autograd cache per aot, forced non-lazy backward lowering). Yields None; state is
    # restored on exit.
    from torch.compiler._cache import CacheArtifactManager

    resolved_fake_mode = _resolve_fake_mode(gm, dynamic_shapes, fake_mode)
    tracing_context = torch._guards.TracingContext(resolved_fake_mode)
    with (
        torch._guards.tracing(tracing_context),
        CacheArtifactManager.with_fresh_cache(),
        config.patch("triton.autotune_at_compile_time", True),
        torch._functorch.config.patch(
            {
                "bundled_autograd_cache": aot,
                # Standalone artifacts are saved immediately after compile_fx
                # returns. Training graphs normally lower the backward lazily on
                # first backward(), so force it while the artifact recorder is
                # still active.
                "force_non_lazy_backward_lowering": True,
            }
        ),
    ):
        yield


def standalone_compile(
    gm: GraphModule,
    example_inputs: Sequence[InputType],
    *,
    dynamic_shapes: DynamicShapesType,
    options: Any,
    aot: bool = False,  # AOT mode, which uses BundledAOTAutogradCache
    donate_graph_module: bool = False,
    fake_mode: FakeTensorMode | None = None,
) -> CompiledArtifact:
    # [FUNCTION DOC] Implementation of the standalone compile entry. Within the standalone context, copies the
    # graph module unless donated (sharing torchbind/process group objects), functionalizes in-place
    # collectives and unboxes process group torchbinds, then runs compile_fx with the given options.
    # aot=True returns an AOTCompiledArtifact (RuntimeError if the compiled fn lacks serialize);
    # otherwise saves cache artifacts (warns if unavailable) and returns a CacheCompiledArtifact.
    # Raises AssertionError if the compiled result is not callable.
    """
    Implementation of torch.inductor.standalone_compile
    """
    from .compile_fx import compile_fx

    ignore_shape_env = _resolve_ignore_shape_env(dynamic_shapes)
    with _standalone_context(gm, dynamic_shapes, aot, fake_mode):
        # compile_fx takes ownership of gm and may mutate it on cache miss.
        # Deepcopy first so the rewrites below land on the owned copy rather
        # than the caller's gm. The gm may carry a non-pickleable torchbind
        # ProcessGroup (or, after a previous unbox, a Python
        # ``dist.ProcessGroup``); smuggle it through deepcopy as a shared
        # reference instead of crashing.
        if not donate_graph_module:
            with _share_torchbind_and_process_group_on_deepcopy():
                gm = copy.deepcopy(gm)
        # ``make_fx`` traces ``dist.*`` collectives as opaque ``c10d.{op}_``
        # calls. Inductor's collective machinery only recognizes the
        # ``_c10d_functional.{op}`` + ``wait_tensor`` form, so rewrite here
        # before compile_fx runs. Also unbox any torchbind ProcessGroup
        # attrs into Python ``dist.ProcessGroup`` so the runtime collective
        # op accepts them (raw torchbind is rejected).
        gm = _functionalize_inplace_collectives(gm)
        gm = _unbox_process_group_torchbinds(gm)
        compiled_fn = compile_fx(
            gm, example_inputs, ignore_shape_env=ignore_shape_env, **options
        )
        if not callable(compiled_fn):
            raise AssertionError("expected compiled_fn to be callable")
        if aot:
            if not hasattr(compiled_fn, "serialize"):
                raise RuntimeError(
                    "Compiled function should have serialize method when aot=True"
                )
            return AOTCompiledArtifact(compiled_fn)
        artifacts = torch.compiler.save_cache_artifacts()
        if artifacts is None:
            log.warning(
                "standalone_compile artifact generation failed, cannot save. "
                "Run with TORCH_LOGS=+torch._inductor.codecache to identify the problem"
            )

    return CacheCompiledArtifact(compiled_fn, artifacts)


class NoRunnableInductorModuleError(RuntimeError):
    # [CLASS DOC] Error raised by compile_to_python when the graph has no compute to lower and so yields no
    # runnable Inductor output module source.
    """Raised by ``compile_to_python`` when the graph yields no runnable Inductor
    output module -- it has no compute to lower (returns inputs/constants unchanged),
    so the compiled artifact carries no output-module source. Callers (e.g.
    torch.compiler.precompile) convert this to a clear user-facing error suggesting an
    alternative.
    """


def _runnable_source(compiled_graph: OutputCode) -> str:
    # [FUNCTION DOC] HELPER: Return the wrapper module source_code of a compiled graph; raises
    # NoRunnableInductorModuleError if it is missing or empty.
    """Return the Inductor output-module source for a compiled inner graph.

    ``compile_fx_inner`` returns a ``CompiledFxGraph`` that carries the wrapper-module
    source as ``source_code``. A graph with no compute (returns inputs/constants
    unchanged) short-circuits to a boxed passthrough with no such source, which we
    surface as ``NoRunnableInductorModuleError``.
    """
    source = getattr(compiled_graph, "source_code", None)
    if not source:
        raise NoRunnableInductorModuleError(
            "the compiled graph produced no runnable Inductor output module: it has no "
            "compute to lower (returns inputs/constants unchanged)."
        )
    return source


def _placeholder_fake_inputs(gm: GraphModule) -> list[Any]:
    # [FUNCTION DOC] HELPER: Collect the fake val metadata of all placeholder nodes of gm, in order. Raises
    # RuntimeError if a placeholder lacks val (graph not traced under a FakeTensorMode).
    """Return the fake ``val`` metadata of ``gm``'s placeholders -- the compile-time input
    contract for a post-AOTAutograd graph. These fake tensors already carry the
    AOTAutograd-decided static/symbolic shapes under one consistent ``FakeTensorMode``, so
    lowering against them (rather than re-fakifying real ``example_inputs``) preserves
    symbolic dims. A placeholder without ``val`` means ``gm`` was not traced under a
    ``FakeTensorMode``, violating the post-AOTAutograd precondition."""
    fake_inputs = []
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        if "val" not in node.meta:
            raise RuntimeError(
                "compile_to_python placeholder has no fake ``val`` metadata; expected a "
                "post-AOTAutograd graph traced under a FakeTensorMode."
            )
        fake_inputs.append(node.meta["val"])
    return fake_inputs


def _acceleration_cache_bytes(
    artifacts: tuple[bytes, CacheInfo] | None,
) -> bytes | None:
    # [FUNCTION DOC] HELPER: Return the raw cache artifact bytes (an accelerator bundle) from the saved
    # artifacts tuple, or None (with a debug log) when none were captured.
    """Return the opaque cache-artifacts bundle that accelerates a later ``load_from_python``,
    or ``None`` when nothing cacheable was produced (no compute) or caches are disabled.

    This is a PURE ACCELERATOR, not a serialized callable. It is the raw
    ``save_cache_artifacts()`` bundle (FxGraph entry + any triton/autotune cache);
    ``load_from_python`` feeds it to ``load_cache_artifacts`` so that exec'ing the emitted
    ``python_code`` loads the precompiled kernels instead of JIT-compiling them. The
    ``python_code`` runs correctly without it, so it carries no key/header framing -- it is
    handed straight back to ``load_cache_artifacts``.
    """
    if artifacts is None:
        log.debug("no cache artifacts captured; no acceleration cache")
        return None
    artifact_bytes, _ = artifacts
    return artifact_bytes


def load_from_python(
    python_code: str, cache: bytes | None = None
) -> Callable[..., Any]:
    # [FUNCTION DOC] Exec the module source from compile_to_python in a fresh namespace and return its call
    # function. If a cache bundle is given it is loaded first (global cache side effect) so kernels use
    # precompiled binaries. Raises RuntimeError if no callable call is defined.
    """Load the module emitted by ``compile_to_python`` into a runnable ``call``.

    ``python_code`` is self-contained: exec'ing it defines ``call`` and JIT-compiles the
    inlined kernels on first use, so it runs with ``cache=None``. When ``cache`` (the
    accelerator bundle from ``compile_to_python``) is provided, it is loaded FIRST so the
    kernels load their precompiled binaries instead of recompiling -- a pure speedup, never
    a correctness requirement. Mirrors ``compile_to_python``: (python_code, cache) in,
    runnable ``call`` out.
    """
    if cache is not None:
        torch.compiler.load_cache_artifacts(cache)
    namespace: dict[str, Any] = {"__name__": "__compile_to_python__"}
    exec(compile(python_code, "<compile_to_python>", "exec"), namespace)
    call = namespace.get("call")
    if not callable(call):
        raise RuntimeError(
            "compile_to_python module did not define a callable ``call``."
        )
    return call


def compile_to_python(
    gm: GraphModule,
    example_inputs: Sequence[InputType],
    *,
    options: dict[str, Any] | None = None,
) -> tuple[str, bytes | None]:
    # [FUNCTION DOC] Internal inductor-layer compile of a post-AOTAutograd, decomposed, functionalized graph
    # module (placeholders carry fake val metadata). Returns (python source of the output module, cache
    # bytes or None). Raises TypeError if gm is not a GraphModule, RuntimeError for async/progressive
    # FX compile modes or when no fake mode is found on placeholders, and NoRunnableInductorModuleError
    # for graphs without compute. Copies gm, functionalizes collectives, then runs compile_fx_inner
    # under no_grad with config patches: user options merged over keep_static_cubin_raw=True, with
    # benchmark_harness and cpp_wrapper forced False, autotune at compile time, fresh cache manager.
    """Compile ``gm`` and return ``(inner_python, cache)`` -- the INNER half of the
    backend contract behind ``torch.compiler.precompile``.

    This is an INTERNAL layered-contract entry point, not an end-user API. End users
    should call ``torch.compiler.precompile``; this function only emits the inductor
    piece of the artifact and assumes its caller (the AOT layer) wraps it. It lives in
    ``torch._inductor`` (a private, leading-underscore module), so it is exposed for
    the AOT layer to import, not as a stable public surface.

    ``inner_python`` is the Inductor output module exposing ``call(args) -> outs``
    for the post-AOTAutograd inner graph (dense, functionalized). It is the inductor
    piece only: it carries NO prelude/epilogue (subclass flatten/unflatten, input-
    mutation copy-back, output-alias regen, grad disabling). Those belong to the AOT
    layer -- a companion change in ``torch._functorch.aot_autograd`` wraps this and
    composes AOTAutograd's codegen'd runtime wrappers around the result.
    Callers must run ``call`` under ``torch.no_grad()`` (the kernels use out= ops).

    Caller preconditions (this layer does not re-derive them):

    - ``gm`` is a post-AOTAutograd dense, functionalized inner graph (the dense
      forward/backward AOTAutograd hands to its inductor backend), NOT a raw Dynamo
      or pre-dispatch graph still carrying subclasses or autograd. Crucially it must be
      DECOMPOSED against the inductor decomposition table: this function drives the
      inductor codegen entry point directly, and inductor lowering asserts "both a
      fallback and a decomp for same op" on an undecomposed op (e.g. ``aten._softmax``).
    - Placeholders carry fake ``val`` metadata (the graph was traced under a
      ``FakeTensorMode``); those fake tensors are the compile-time input contract.
    - ``gm`` lowers to a runnable inductor output module. A graph with no compute
      raises ``NoRunnableInductorModuleError``.

    The compile lowers against the placeholders' fake ``val`` metadata, whose
    static/symbolic shapes AOTAutograd already resolved, so the graph is the source of
    truth for shapes -- there is no ``dynamic_shapes`` knob. ``example_inputs`` is accepted
    for the backend-contract signature but is not re-fakified for the compile itself.

    ``options`` is an optional inductor config-override dict (``None`` means no
    overrides), the same shape as ``torch._inductor.compile``'s ``options``. The
    keys are inductor config names: they are merged into the ``config.patch``
    block this function already wraps the compile in, so they take effect as
    config rather than being forwarded as ``compile_fx`` kwargs. The two
    capture-critical pins below (``benchmark_harness``, ``cpp_wrapper``) always win
    over a conflicting user option, since the source-capture contract depends on
    them and a caller cannot be allowed to break the emitted module.

    ``inner_python`` is self-contained: exec'ing it JIT-compiles the inlined kernels on
    first use, so it runs with no cache. ``cache`` is a PURE ACCELERATOR -- the opaque
    ``save_cache_artifacts()`` bundle; passing it to ``load_from_python`` warms the kernel
    caches so exec loads the precompiled binaries instead of recompiling. It is ``None``
    when the graph produced no cacheable module (no compute) or caches are disabled
    (``force_disable_caches`` or ``fx_graph_cache=False``); ``inner_python`` still runs.

    ``inner_python`` is read off the ``CompiledFxGraph`` that ``compile_fx_inner``
    returns -- Inductor stashes the wrapper-module source on it as ``source_code`` -- so
    it reflects the FINAL selected module on a cold compile, a warm cache restore, or with
    caches disabled. No process-global codegen hook is involved, and the throwaway
    max_autotune benchmark lowerings (which compile to their own modules during
    autotuning) never become the returned graph, so nothing needs to be filtered out.
    """
    if not isinstance(gm, torch.fx.GraphModule):
        raise TypeError(
            f"compile_to_python expects a post-AOTAutograd torch.fx.GraphModule, "
            f"got {type(gm)}. This is an internal entry point wrapped by a higher AOT "
            f"layer and is not meant to be called directly."
        )
    # The experimental TORCHINDUCTOR_FX_COMPILE_MODE=async+/progressive+ schemes make
    # compile_fx return an _AsyncOutputCode/_ProgressiveOutputCode that carries only
    # _boxed_call and never surfaces the wrapper ``source_code`` this contract reads off.
    # These are module-level globals resolved from the env at import time, not
    # config-patchable like the pins below, so detect them up front and fail with a
    # distinct error instead of the no-compute NoRunnableInductorModuleError -- the graph
    # is runnable, the async scheme just did not surface its source.
    from torch._inductor import compile_fx as _compile_fx

    if _compile_fx.fx_compile_async or _compile_fx.fx_compile_progressive:
        raise RuntimeError(
            "compile_to_python needs synchronous source capture and does not yet "
            "support TORCHINDUCTOR_FX_COMPILE_MODE=async+/progressive+ (the async "
            "output code does not surface the wrapper source)."
        )
    # Treat ``options`` as inductor config overrides and fold them into the same
    # ``config.patch`` we wrap the compile in. ``keep_static_cubin_raw`` is defaulted True
    # (BEFORE user options, so a caller can still trade it off for a smaller cache): its
    # default False nulls the static-launcher raw cubin and relies on the cubin FILE being
    # on disk, which holds for a same-machine warm cache but NOT the cold/cross-container
    # load this cache targets -- keeping it lets ``load_from_python`` rehydrate the cubin
    # from the bundle so the static CUDA launcher survives a fresh-dir load instead of
    # silently falling back to the slower dynamic launch. The two output pins are applied
    # AFTER the user options so they override any conflicting key: benchmark_harness=False
    # keeps the emitted module runnable (no get_args()/benchmark_compiled_module()/__main__),
    # and cpp_wrapper=False keeps it a python wrapper (the C++ backend emits a C++ ``call``
    # we cannot inline). A caller must not be able to break either.
    config_patches: dict[str, Any] = {"keep_static_cubin_raw": True}
    if options is not None:
        config_patches.update(options)
    config_patches.update(
        {
            "benchmark_harness": False,
            "cpp_wrapper": False,
        }
    )
    # ``gm`` is a POST-AOTAutograd inner graph: already functionalized and decomposed
    # against the inductor decomposition table. That precondition lets us drive the
    # inductor codegen entry point (``compile_fx_inner``) DIRECTLY on the dense graph
    # rather than re-entering AOTAutograd via ``standalone_compile``. Re-entry would only
    # re-run decomposition -- a no-op on an already-decomposed graph -- and hand back an
    # AOTAutograd-level cache artifact that belongs to the layer above; driving the inner
    # compile keeps this at the inductor layer and yields the accelerator cache bundle that
    # ``_acceleration_cache_bytes`` returns. (If ``gm`` is NOT decomposed against the
    # inductor table, inductor lowering asserts "both a fallback and a decomp for same op"
    # -- the precondition is load-bearing, not defensive.)
    from torch._guards import detect_fake_mode, tracing, TracingContext
    from torch.compiler._cache import CacheArtifactManager

    from .compile_fx import compile_fx_inner
    from .virtualized import V

    # Own a copy: the collective rewrites and inductor may mutate the graph, and ``gm`` may
    # carry a non-pickleable torchbind ProcessGroup smuggled through deepcopy as a shared
    # reference (mirrors ``standalone_compile``). ``make_fx`` traces ``dist.*`` collectives
    # as opaque ``c10d.{op}_`` calls; rewrite them to the ``_c10d_functional.{op}`` +
    # ``wait_tensor`` form inductor recognizes and unbox torchbind ProcessGroups.
    with _share_torchbind_and_process_group_on_deepcopy():
        gm = copy.deepcopy(gm)
    gm = _functionalize_inplace_collectives(gm)
    gm = _unbox_process_group_torchbinds(gm)

    # Lower against the placeholders' fake ``val`` metadata (the compile-time input
    # contract, carrying the graph's static/symbolic shapes under one FakeTensorMode)
    # rather than re-fakifying ``example_inputs``, which are real and would drop symbolic
    # dims. A post-AOTAutograd graph's shapes are already baked into this metadata, so
    # there is no separate dynamic-shapes knob.
    fake_inputs = _placeholder_fake_inputs(gm)
    fake_mode = detect_fake_mode(fake_inputs)
    if fake_mode is None:
        raise RuntimeError(
            "compile_to_python could not detect a FakeTensorMode on the graph's "
            "placeholders; expected a post-AOTAutograd graph traced under one."
        )
    # no_grad pins the inference path; autotune_at_compile_time keeps the emitted source
    # self-contained (autotuning resolved at compile, not deferred to runtime). The fresh
    # CacheArtifactManager isolates the cache bundle ``_acceleration_cache_bytes`` returns.
    with (
        torch.no_grad(),
        config.patch(config_patches),
        config.patch("triton.autotune_at_compile_time", True),
        tracing(TracingContext(fake_mode)),
        V.set_fake_mode(fake_mode),
        CacheArtifactManager.with_fresh_cache(),
    ):
        compiled_graph = compile_fx_inner(
            gm,
            fake_inputs,
            static_input_idxs=(),
            cudagraphs=BoxedBool(False),
            is_inference=True,
            boxed_forward_device_index=BoxedDeviceIndex(None),
        )
        artifacts = torch.compiler.save_cache_artifacts()
    inner_python = _runnable_source(compiled_graph)
    cache = _acceleration_cache_bytes(artifacts)
    return inner_python, cache


def autograd_cache_key(
    graph,
    example_inputs,
    dynamic_shapes: DynamicShapesType,
    aot: bool = False,  # AOT mode, which uses BundledAOTAutogradCache
    fake_mode: FakeTensorMode | None = None,
):
    # [FUNCTION DOC] Compute the autograd cache key for a graph: resolves the ignore-shape-env flag from
    # dynamic_shapes and, in the standalone context, delegates to compile_fx.autograd_cache_key with
    # the graph and example inputs. Returns whatever that function returns.
    from . import compile_fx

    ignore_shape_env = _resolve_ignore_shape_env(dynamic_shapes)
    with _standalone_context(graph, dynamic_shapes, aot, fake_mode):
        return compile_fx.autograd_cache_key(
            graph,
            example_inputs,
            ignore_shape_env=ignore_shape_env,
        )
