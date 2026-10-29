# [MODULE DOC]
# Reading and writing of the PT2 archive format (a zip-based container) that
# bundles exported programs, their weights and constants, AOTInductor compiled
# artifacts, extra files and executorch files. Main contents: PT2ArchiveWriter
# and PT2ArchiveReader (record-level I/O), package_pt2 (save), load_pt2 (load),
# is_pt2_package, AOTICompiledModel (callable wrapper over a compiled package
# loader), PT2ArchiveContents (load result), plus private helpers that split
# tensors into pickled versus raw-byte payloads with storage deduplication.
# Import-time side effects: creates the module logger and overrides the
# __module__ attribute of the public functions/classes to torch.export.pt2_archive.
import glob
import io
import json
import logging
import os
import pickle
import tempfile
import zipfile
from dataclasses import dataclass
from typing import Any, IO, TYPE_CHECKING, TypeAlias
from typing_extensions import TypeIs

import torch
import torch.utils._pytree as pytree
from torch._export.serde import schema
from torch._export.serde.serialize import (
    _dataclass_to_dict,
    _dict_to_dataclass,
    deserialize_device,
    deserialize_scalar_type,
    deserialize_size,
    deserialize_storage_offset,
    deserialize_stride,
    ExportedProgramDeserializer,
    serialize,
    serialize_tensor_meta,
    SerializedArtifact,
)
from torch._inductor.cpp_builder import normalize_path_separator
from torch._library.opaque_object import is_custom_class_obj
from torch._subclasses.fake_tensor import FakeTensor, is_fake_tensor
from torch.export import ExportedProgram
from torch.export._tree_utils import reorder_kwargs
from torch.export.pt2_archive._package_weights import (
    get_complete_tensor,
    group_weights,
    TensorProperties,
    Weights,
)
from torch.export.pt2_archive.constants import (
    AOTINDUCTOR_DIR,
    ARCHIVE_FORMAT_PATH,
    ARCHIVE_FORMAT_VALUE,
    ARCHIVE_VERSION_PATH,
    ARCHIVE_VERSION_VALUE,
    CONSTANTS_CONFIG_FILENAME_FORMAT,
    CONSTANTS_DIR,
    CUSTOM_OBJ_FILENAME_PREFIX,
    EXECUTORCH_DIR,
    EXTRA_DIR,
    MODELS_DIR,
    MODELS_FILENAME_FORMAT,
    OPAQUE_OBJ_FILENAME_PREFIX,
    SAMPLE_INPUTS_FILENAME_FORMAT,
    TENSOR_CONSTANT_FILENAME_PREFIX,
    WEIGHT_FILENAME_PREFIX,
    WEIGHTS_CONFIG_FILENAME_FORMAT,
    WEIGHTS_DIR,
)
from torch.types import FileLike


if TYPE_CHECKING:
    from torch.utils._ordered_set import OrderedSet


DEFAULT_PICKLE_PROTOCOL = 2
AOTI_FILES: TypeAlias = list[str | Weights] | dict[str, list[str | Weights]]


logger: logging.Logger = logging.getLogger(__name__)


def is_pt2_package(serialized_model: bytes | str) -> bool:
    # [FUNCTION DOC] Returns True if the given bytes or file path is a zip whose
    # first entry's root folder contains an archive-format record equal to the
    # marker "pt2". Any failure to open or parse the zip is caught, logged at info
    # level, and results in False. Never raises for bad input; reads the input only.
    # KNOWN ISSUE: the root folder is split on the OS path separator although zip
    # entry names always use forward slashes, so this misdetects on Windows.
    """
    Check if the serialized model is a PT2 Archive package.
    """
    try:
        with zipfile.ZipFile(
            io.BytesIO(serialized_model)
            if isinstance(serialized_model, bytes)
            else serialized_model
        ) as zip_reader:
            root_folder = zip_reader.namelist()[0].split(os.path.sep)[0]
            archive_format_path = f"{root_folder}/{ARCHIVE_FORMAT_PATH}"
            if archive_format_path in zip_reader.namelist():
                return zip_reader.read(archive_format_path) == b"pt2"
    except Exception:
        logger.info("Model is not a PT2 package")
    return False


class PT2ArchiveWriter:
    # [CLASS DOC] Context manager that writes records into a PT2 archive through
    # a native PyTorchFileWriter. State: archive_file, the native writer. On
    # normal or exceptional exit it adds the archive format and archive version
    # records if missing, then finalizes the file. Methods write bytes, strings,
    # files and whole folders, and query already written records.
    """
    Context manager for writing a PT2 archive.
    """

    def __init__(self, archive_path_or_buffer: FileLike):
        # [FUNCTION DOC] Opens a native file writer on a path (separators
        # normalized) or a writable buffer and sets the zip format minimum
        # version to 6. Side effect: creates/truncates the target file.
        if isinstance(archive_path_or_buffer, str):
            archive_path_or_buffer = normalize_path_separator(archive_path_or_buffer)
        self.archive_file = torch._C.PyTorchFileWriter(archive_path_or_buffer)  # type: ignore[arg-type]
        # NOTICE: version here is different from the archive_version
        # this is the version of zip file format, which is used by PyTorchFileWriter, which write to /.data/version
        # archive_version is the version of the PT2 archive spec, which write to /archive_version
        self.archive_file.set_min_version(6)

    def __enter__(self) -> "PT2ArchiveWriter":
        # [FUNCTION DOC] Returns the writer itself for use in a with statement.
        return self

    def __exit__(self, *args: Any) -> None:
        # [FUNCTION DOC] Writes the archive format and version records if they are
        # not yet present, then closes (finalizes) the archive. Runs even when the
        # with body raised, so a partial archive is still finalized; never
        # suppresses the exception.
        if not self.has_record(ARCHIVE_FORMAT_PATH):
            self.write_string(ARCHIVE_FORMAT_PATH, ARCHIVE_FORMAT_VALUE)

        if not self.has_record(ARCHIVE_VERSION_PATH):
            self.write_string(ARCHIVE_VERSION_PATH, ARCHIVE_VERSION_VALUE)

        self.close()

    def has_record(self, name: str) -> bool:
        # [FUNCTION DOC] Returns True if a record with exactly this name has
        # already been written to the archive.
        """
        Check if a record exists in the archive.
        """
        return name in self.archive_file.get_all_written_records()

    def count_prefix(self, prefix: str) -> int:
        # [FUNCTION DOC] Returns how many already written records have a name
        # starting with the prefix; used to pick the next free file index.
        """
        Count the number of records that start with a given prefix.
        """
        return sum(
            1
            for record in self.archive_file.get_all_written_records()
            if record.startswith(prefix)
        )

    def write_bytes(self, name: str, data: bytes) -> None:
        # [FUNCTION DOC] Writes a bytes object as a record with the given name.
        # Raises AssertionError if data is not a bytes object. Side effect: appends
        # a record to the archive.
        """
        Write a bytes object to the archive.
        name: The destination file inside the archive.
        data: The bytes object to write.
        """
        if not isinstance(data, bytes):
            raise AssertionError(f"Expected bytes but got {type(data)}")
        self.archive_file.write_record(name, data, len(data))

    def write_string(self, name: str, data: str) -> None:
        # [FUNCTION DOC] Encodes the string (default encoding, UTF-8) and writes it
        # as a record. Raises AssertionError if data is not a string.
        """
        Write a string object to the archive.
        name: The destination file inside the archive.
        data: The string object to write.
        """
        if not isinstance(data, str):
            raise AssertionError(f"Expected string but got {type(data)}")
        data_bytes = data.encode()
        self.write_bytes(name, data_bytes)

    def write_file(self, name: str, file_path: str) -> None:
        # [FUNCTION DOC] Reads the whole file from disk into memory and writes it
        # as a record named name. Raises AssertionError if file_path is not an
        # existing regular file.
        """
        Copy a file into the archive.
        name: The destination file inside the archive.
        file_path: The source file on disk.
        """
        if not os.path.isfile(file_path):
            raise AssertionError(f"{file_path} is not a valid file path")

        with open(file_path, "rb") as f:
            file_bytes = f.read()
            self.write_bytes(name, file_bytes)

    def write_folder(self, archive_dir: str, folder_dir: str) -> None:
        # [FUNCTION DOC] Recursively copies every regular file under folder_dir
        # into the archive beneath archive_dir, keeping relative paths. Raises
        # AssertionError if folder_dir is not a directory. Side effect: writes
        # one record per file.
        # KNOWN ISSUE: the recursive glob skips hidden (dot) files and folders.
        """
        Copy a folder into the archive.
        archive_dir: The destination folder inside the archive.
        folder_dir: The source folder on disk.
        """
        if not os.path.isdir(folder_dir):
            raise AssertionError(f"{folder_dir} is not a valid directory path")

        file_paths = filter(
            os.path.isfile, glob.glob(f"{folder_dir}/**", recursive=True)
        )
        for file_path in file_paths:
            # pyrefly: ignore [no-matching-overload]
            filename = os.path.relpath(file_path, folder_dir)
            archive_path = os.path.join(archive_dir, filename)
            # pyrefly: ignore [bad-argument-type]
            self.write_file(archive_path, file_path)

    def close(self) -> None:
        # [FUNCTION DOC] Finalizes the archive by writing the end-of-file
        # structures; no further writes are valid afterwards.
        """
        Close the archive.
        """
        self.archive_file.write_end_of_file()


class PT2ArchiveReader:
    # [CLASS DOC] Context manager that reads records from a PT2 archive through
    # a native PyTorchFileReader. State: archive_file, the native reader.
    # Invariant: the archive format record matched the expected value at
    # construction. Methods read bytes/strings, report the archive version and
    # list record names. Nothing needs closing.
    """
    Context manager for reading a PT2 archive.
    """

    def __init__(self, archive_path_or_buffer: FileLike):
        # [FUNCTION DOC] Opens a native reader on a path (separators normalized)
        # or a buffer and validates the archive format record. Raises
        # AssertionError if the format record differs; the native reader may raise
        # if the record is missing or the file is not a valid archive.
        if isinstance(archive_path_or_buffer, str):
            archive_path_or_buffer = normalize_path_separator(archive_path_or_buffer)
        self.archive_file = torch._C.PyTorchFileReader(archive_path_or_buffer)  # type: ignore[arg-type]
        if self.read_string(ARCHIVE_FORMAT_PATH) != ARCHIVE_FORMAT_VALUE:
            raise AssertionError("Invalid archive format")

    def __enter__(self) -> "PT2ArchiveReader":
        # [FUNCTION DOC] Returns the reader itself for use in a with statement.
        return self

    def __exit__(self, *args: Any) -> None:
        # [FUNCTION DOC] Does nothing; the native reader has no close operation.
        # torch._C.PyTorchFileReader doesn't have a close method
        pass

    def read_bytes(self, name: str) -> bytes:
        # [FUNCTION DOC] Returns the raw contents of the named record. The native
        # reader raises an error if the record does not exist.
        """
        Read a bytes object from the archive.
        name: The source file inside the archive.
        """
        return self.archive_file.get_record(name)

    def read_string(self, name: str) -> str:
        # [FUNCTION DOC] Returns the named record decoded as a string (default
        # encoding, UTF-8). Errors from missing records or decoding propagate.
        """
        Read a string object from the archive.
        name: The source file inside the archive.
        """
        data = self.read_bytes(name)
        return data.decode()

    def archive_version(self) -> int:
        # [FUNCTION DOC] Returns the PT2 archive spec version as an integer, read
        # from the version record. If the record cannot be read the archive is
        # treated as the oldest version, 0. Raises ValueError if the record text
        # is not an integer.
        """
        Get the archive version.
        """
        try:
            archive_version = self.read_string(ARCHIVE_VERSION_PATH)
        except Exception:
            # if archive_version is not found, it means the archive is older than version 0.
            # In this case, we assume the archive is version 0.
            archive_version = "0"

        return int(archive_version)

    def get_file_names(self) -> list[str]:
        # [FUNCTION DOC] Returns the names of all records in the archive.
        """
        Get the file names in the archive.
        """
        return self.archive_file.get_all_records()


is_pt2_package.__module__ = "torch.export.pt2_archive"
PT2ArchiveWriter.__module__ = "torch.export.pt2_archive"
PT2ArchiveReader.__module__ = "torch.export.pt2_archive"


def _package_aoti_files(
    archive_writer: PT2ArchiveWriter,
    aoti_files: AOTI_FILES | None,
    pickle_protocol: int = DEFAULT_PICKLE_PROTOCOL,
) -> None:
    # [FUNCTION DOC] HELPER: Writes AOTInductor artifacts into the archive. A
    # plain list is treated as one model named "model"; otherwise a mapping of
    # model name to files. Ordinary files go under the AOTInductor directory per
    # model (custom object files go to the constants directory); empty names are
    # skipped. Weights objects are deduplicated by shared storage, saved with
    # torch.save into the weights directory, and a per-model weights_config.json
    # maps weight name to file, shape, stride and offset. Does nothing for None.
    # Raises AssertionError if the input is neither list nor mapping or a file is missing.
    if aoti_files is None:
        return

    if isinstance(aoti_files, list):
        aoti_files = {"model": aoti_files}

    if not isinstance(aoti_files, dict):
        raise AssertionError(
            f"Expected aoti_files to be a dict, but got {type(aoti_files)}"
        )

    all_weights: dict[str, Weights] = {}  # model_name -> weight
    weights_configs: dict[
        str, dict[str, Any]
    ] = {}  # model_name -> (weight_name -> (filename, shape, stride, offset))

    for model_name, files in aoti_files.items():
        weights_configs[model_name] = {}

        for file in files:
            if file == "":
                continue

            if isinstance(file, Weights):
                all_weights[model_name] = file
                continue

            # Note: Previously we rejected multiple .so cases. But Triton CPU AOTI
            # has multiple .so files per model (wrapper, launcher, kernel).

            filename = os.path.basename(file)
            if filename.startswith(CUSTOM_OBJ_FILENAME_PREFIX):
                new_filepath = os.path.join(CONSTANTS_DIR, filename)
            else:
                new_filepath = os.path.join(AOTINDUCTOR_DIR, model_name, filename)
            logger.debug(
                "Saving AOTI generated file %s to archive in %s", file, new_filepath
            )
            archive_writer.write_file(
                str(new_filepath),
                file,
            )

    if len(all_weights) > 0:
        # Dedup weights
        grouped_tensors: list[OrderedSet[tuple[str, str]]] = group_weights(all_weights)
        for idx, group in enumerate(grouped_tensors):
            filename = f"{WEIGHT_FILENAME_PREFIX}{idx}"
            complete_tensor = get_complete_tensor(group, all_weights)
            buffer = io.BytesIO()
            torch.save(complete_tensor, buffer, pickle_protocol=pickle_protocol)
            archive_writer.write_bytes(
                os.path.join(WEIGHTS_DIR, filename), buffer.getvalue()
            )
            for model_name, weight_name in group:
                _, w_property = all_weights[model_name].get_weight(weight_name)
                weights_configs[model_name][weight_name] = (
                    filename,
                    w_property.shape,
                    w_property.stride,
                    w_property.offset,
                )

        for model_name, weights_config in weights_configs.items():
            archive_writer.write_string(
                os.path.join(AOTINDUCTOR_DIR, model_name, "weights_config.json"),
                json.dumps(weights_config),
            )
            logger.debug("packaging weights_config for model %s", model_name)
            logger.debug(weights_config)


def _is_fake_tensor(t: torch.Tensor) -> TypeIs[FakeTensor]:
    # [FUNCTION DOC] HELPER: Type-narrowing wrapper returning whether t is a fake tensor.
    return is_fake_tensor(t)


def _is_tensor_subclass(t: torch.Tensor) -> bool:
    # [FUNCTION DOC] HELPER: Returns True for a tensor whose underlying data
    # tensor is of a type other than the plain tensor class (a tensor subclass).
    return isinstance(t, torch.Tensor) and type(t.data) is not torch.Tensor


def _get_raw_tensor_bytes(value: torch.Tensor) -> bytes:
    # [FUNCTION DOC] HELPER: Returns the raw bytes of the tensor's whole
    # untyped storage (metadata is saved elsewhere). Fake and empty tensors
    # yield empty bytes. Non-CPU tensors are copied to CPU first and the copy is
    # kept alive while its memory is read.
    """
    Get the raw bytes of a tensor. This is used to save the tensor in pt2 archive.
    """
    # NOTE: don't chain .cpu() with .data_ptr(). If an HtoD copy needs to be
    # performed, the CPU copy needs to be kept alive when its underlying
    # memory is accessed.
    import ctypes

    if _is_fake_tensor(value):
        value_bytes = b""
    elif value.data_ptr():
        cpu_tensor = value.cpu()
        value_untyped_storage = cpu_tensor.untyped_storage()
        # we store the raw bytes the untyped storage. Tensor metadata is stored separately
        value_bytes = bytes(
            ctypes.cast(
                value_untyped_storage.data_ptr(),
                ctypes.POINTER(ctypes.c_ubyte * value_untyped_storage.size()),
            ).contents
        )
    else:
        # for empty tensor
        value_bytes = b""
    return value_bytes


def _should_use_pickle(t: torch.Tensor) -> bool:
    # [FUNCTION DOC] HELPER: True for real (non-fake) tensor subclasses, which
    # must be pickled; plain and fake tensors use the raw-byte path.
    return _is_tensor_subclass(t) and not _is_fake_tensor(t)


def _save_pickled_tensors(
    pickled_items: list[tuple[str, torch.Tensor]],
    archive_writer: PT2ArchiveWriter,
    config: dict[str, schema.PayloadMeta],
    directory: str,
    filename_prefix: str,
    idx: int,
    pickle_protocol: int = DEFAULT_PICKLE_PROTOCOL,
) -> int:
    # [FUNCTION DOC] HELPER: Saves each (name, tensor) with torch.save to a new
    # record named prefix plus a running index under the directory, and records
    # a PayloadMeta (pickle flag set, parameter flag, tensor meta) in the config
    # mapping keyed by name. Mutates config; writes archive records. Returns the next free index.
    """Save pickled tensors and update config. Returns updated index."""
    for item_fqn, tensor in pickled_items:
        path_name = f"{filename_prefix}{idx}"
        archive_path = os.path.join(directory, path_name)
        buffer = io.BytesIO()
        torch.save(tensor, buffer, pickle_protocol=pickle_protocol)
        archive_writer.write_bytes(archive_path, buffer.getvalue())

        config[item_fqn] = schema.PayloadMeta(
            path_name=path_name,
            is_param=isinstance(tensor, torch.nn.Parameter),
            use_pickle=True,
            tensor_meta=serialize_tensor_meta(tensor),
        )
        idx += 1
    return idx


def _save_raw_tensors(
    raw_items: dict[str, tuple[torch.Tensor, TensorProperties]],
    model_name: str,
    archive_writer: PT2ArchiveWriter,
    config: dict[str, schema.PayloadMeta],
    directory: str,
    filename_prefix: str,
    idx: int,
) -> int:
    # [FUNCTION DOC] HELPER: Groups the tensors by shared storage, and for each
    # group writes the bytes of the complete covering tensor as one record
    # (prefix plus running index) under the directory. Each member gets a
    # PayloadMeta (no pickle) in the config mapping so aliasing can be restored
    # at load time. Mutates config; returns the next free index (unchanged for empty input).
    """Save deduplicated raw tensor bytes and update config. Returns updated index."""
    if not raw_items:
        return idx

    weights_dict = {model_name: Weights(raw_items)}
    storage_groups = group_weights(weights_dict)

    for group in storage_groups:
        # Find the complete tensor that covers all others in this storage group
        complete_tensor = get_complete_tensor(group, weights_dict)

        path_name = f"{filename_prefix}{idx}"
        archive_path = os.path.join(directory, path_name)
        tensor_bytes = _get_raw_tensor_bytes(complete_tensor)
        archive_writer.write_bytes(archive_path, tensor_bytes)
        idx += 1

        for _, item_fqn in group:
            tensor, _ = weights_dict[model_name].get_weight(item_fqn)
            config[item_fqn] = schema.PayloadMeta(
                path_name=path_name,
                is_param=isinstance(tensor, torch.nn.Parameter),
                use_pickle=False,
                tensor_meta=serialize_tensor_meta(tensor),
            )

    return idx


def _package_state_dict(
    model_name: str,
    exported_program: ExportedProgram,
    archive_writer: PT2ArchiveWriter,
    pickle_protocol: int = DEFAULT_PICKLE_PROTOCOL,
) -> schema.PayloadConfig:
    # [FUNCTION DOC] HELPER: Writes the exported program's state dict into the
    # weights directory. Tensor subclasses are pickled; other tensors are saved
    # as deduplicated raw bytes. Returns a PayloadConfig describing every weight.
    # Raises AssertionError if a state dict value is not a tensor.
    weights_config: dict[str, schema.PayloadMeta] = {}

    pickled_weights: list[tuple[str, torch.Tensor]] = []
    raw_weights: dict[str, tuple[torch.Tensor, TensorProperties]] = {}

    # Categorize weights
    for weight_fqn, weight_tensor in exported_program.state_dict.items():
        if not isinstance(weight_tensor, torch.Tensor):
            raise AssertionError("only torch.Tensor is allowed in state_dict")
        if _should_use_pickle(weight_tensor):
            pickled_weights.append((weight_fqn, weight_tensor))
        else:
            raw_weights[weight_fqn] = (weight_tensor, TensorProperties(weight_tensor))

    idx = archive_writer.count_prefix(os.path.join(WEIGHTS_DIR, WEIGHT_FILENAME_PREFIX))

    # Save weights in pickle format
    idx = _save_pickled_tensors(
        pickled_weights,
        archive_writer,
        weights_config,
        WEIGHTS_DIR,
        WEIGHT_FILENAME_PREFIX,
        idx,
        pickle_protocol,
    )

    # Save weights in raw bytes format
    _save_raw_tensors(
        raw_weights,
        model_name,
        archive_writer,
        weights_config,
        WEIGHTS_DIR,
        WEIGHT_FILENAME_PREFIX,
        idx,
    )

    return schema.PayloadConfig(config=weights_config)


def _package_constants(
    model_name: str,
    exported_program: ExportedProgram,
    archive_writer: PT2ArchiveWriter,
    pickle_protocol: int = DEFAULT_PICKLE_PROTOCOL,
) -> schema.PayloadConfig:
    # [FUNCTION DOC] HELPER: Writes the exported program's constants into the
    # constants directory and returns a PayloadConfig. Tensor constants follow
    # the pickled/raw split like weights; torchbind script objects are saved with
    # the native pickler under the custom-object prefix, and other custom or
    # opaque objects with standard pickle under the opaque prefix.
    # Raises RuntimeError for any other constant type.
    constants_config: dict[str, schema.PayloadMeta] = {}

    pickled_constants: list[tuple[str, torch.Tensor]] = []
    raw_constants: dict[str, tuple[torch.Tensor, TensorProperties]] = {}
    custom_objects: list[tuple[str, Any]] = []

    # Categorize constants
    for constant_fqn, constant in exported_program.constants.items():
        if isinstance(constant, torch.Tensor):
            if _should_use_pickle(constant):
                pickled_constants.append((constant_fqn, constant))
            else:
                raw_constants[constant_fqn] = (constant, TensorProperties(constant))

        elif isinstance(constant, torch._C.ScriptObject) or is_custom_class_obj(
            constant
        ):
            custom_objects.append((constant_fqn, constant))

        else:
            raise RuntimeError(f"Unsupported constant type: {type(constant)}")

    tensor_idx = archive_writer.count_prefix(
        os.path.join(CONSTANTS_DIR, TENSOR_CONSTANT_FILENAME_PREFIX)
    )
    custom_obj_idx = archive_writer.count_prefix(
        os.path.join(CONSTANTS_DIR, CUSTOM_OBJ_FILENAME_PREFIX)
    )
    opaque_obj_idx = archive_writer.count_prefix(
        os.path.join(CONSTANTS_DIR, OPAQUE_OBJ_FILENAME_PREFIX)
    )

    # Save constants in pickle format
    tensor_idx = _save_pickled_tensors(
        pickled_constants,
        archive_writer,
        constants_config,
        CONSTANTS_DIR,
        TENSOR_CONSTANT_FILENAME_PREFIX,
        tensor_idx,
        pickle_protocol,
    )

    # Save constants in raw bytes format
    _save_raw_tensors(
        raw_constants,
        model_name,
        archive_writer,
        constants_config,
        CONSTANTS_DIR,
        TENSOR_CONSTANT_FILENAME_PREFIX,
        tensor_idx,
    )

    # ScriptObjects use the torchbind-aware pickler (zip-archive format,
    # readable from C++ at load time). Opaque values use standard Python
    # pickle. Distinct filename prefixes act as the format discriminator
    # for the reader.
    for constant_fqn, constant in custom_objects:
        if isinstance(constant, torch._C.ScriptObject):
            path_name = f"{CUSTOM_OBJ_FILENAME_PREFIX}{custom_obj_idx}"
            obj_bytes = torch._C._pickle_save(constant)
            custom_obj_idx += 1
        else:
            path_name = f"{OPAQUE_OBJ_FILENAME_PREFIX}{opaque_obj_idx}"
            obj_bytes = pickle.dumps(constant, protocol=pickle_protocol)
            opaque_obj_idx += 1
        archive_path = os.path.join(CONSTANTS_DIR, path_name)
        archive_writer.write_bytes(archive_path, obj_bytes)

        constants_config[constant_fqn] = schema.PayloadMeta(
            path_name=path_name,
            is_param=False,
            use_pickle=True,
            tensor_meta=None,
        )

    return schema.PayloadConfig(config=constants_config)


def _package_payload_config(
    archive_writer: PT2ArchiveWriter,
    payload_config: schema.PayloadConfig,
    config_file: str,
) -> None:
    # [FUNCTION DOC] HELPER: Converts the payload config to plain data, dumps it
    # as JSON text, and writes it to the archive record config_file.
    """
    Save the payload config as json file in the archive.
    """
    archive_writer.write_string(
        config_file, json.dumps(_dataclass_to_dict(payload_config))
    )


def _package_exported_programs(
    archive_writer: PT2ArchiveWriter,
    exported_programs: ExportedProgram | dict[str, ExportedProgram] | None,
    opset_version: dict[str, int] | None = None,
    pickle_protocol: int = DEFAULT_PICKLE_PROTOCOL,
) -> None:
    # [FUNCTION DOC] HELPER: Saves one or several exported programs (a single
    # program is named "model"). Per model it writes the weights, the weights
    # config, the constants, the constants config, then the serialized program
    # (without state dict or constants) and its sample inputs. Does nothing for
    # None. Raises AssertionError if the input is neither a program nor a mapping.
    if exported_programs is None:
        return

    if isinstance(exported_programs, ExportedProgram):
        exported_programs = {"model": exported_programs}

    if not isinstance(exported_programs, dict):
        raise AssertionError(
            f"Expected exported_programs to be a dict, but got {type(exported_programs)}"
        )

    for model_name, ep in exported_programs.items():
        weights_config = _package_state_dict(
            model_name, ep, archive_writer, pickle_protocol
        )
        weights_config_file = WEIGHTS_CONFIG_FILENAME_FORMAT.format(model_name)
        _package_payload_config(archive_writer, weights_config, weights_config_file)

        constants_config = _package_constants(
            model_name, ep, archive_writer, pickle_protocol
        )
        constants_config_file = CONSTANTS_CONFIG_FILENAME_FORMAT.format(model_name)
        _package_payload_config(archive_writer, constants_config, constants_config_file)

        artifact: SerializedArtifact = serialize(
            ep,
            opset_version,
            pickle_protocol,
            serialize_state_dict=False,
            serialize_constants=False,
        )

        archive_writer.write_bytes(
            MODELS_FILENAME_FORMAT.format(model_name), artifact.exported_program
        )
        archive_writer.write_bytes(
            SAMPLE_INPUTS_FILENAME_FORMAT.format(model_name),
            artifact.example_inputs,
        )


def _package_extra_files(
    archive_writer: PT2ArchiveWriter, extra_files: dict[str, Any] | None
) -> None:
    # [FUNCTION DOC] HELPER: Writes each extra file as a string record under the
    # extra directory prefix; no-op for None. Contents must be strings.
    if extra_files is None:
        return

    for extra_file_name, content in extra_files.items():
        archive_writer.write_string(f"{EXTRA_DIR}{extra_file_name}", content)


def _package_executorch_files(
    archive_writer: PT2ArchiveWriter, executorch_files: dict[str, bytes] | None
) -> None:
    # [FUNCTION DOC] HELPER: Writes each executorch file as a bytes record under
    # the executorch directory prefix; no-op for None.
    if executorch_files is None:
        return

    for file_name, content in executorch_files.items():
        archive_writer.write_bytes(f"{EXECUTORCH_DIR}{file_name}", content)


def package_pt2(
    f: FileLike,
    *,
    exported_programs: ExportedProgram | dict[str, ExportedProgram] | None = None,
    aoti_files: AOTI_FILES | None = None,
    extra_files: dict[str, Any] | None = None,
    opset_version: dict[str, int] | None = None,
    pickle_protocol: int = DEFAULT_PICKLE_PROTOCOL,
    executorch_files: dict[str, bytes] | None = None,
) -> FileLike:
    # [FUNCTION DOC] Public save entry point. Writes exported programs, AOTInductor
    # files, extra files and executorch files into a PT2 archive at path or buffer f.
    # Raises AssertionError if exported_programs, aoti_files and extra_files are all
    # None. Logs a warning (not an error) if f is not a .pt2 path or writable seekable
    # buffer. Returns f (path converted to a string; buffers are rewound to the start).
    # Side effect: creates the archive file.
    r"""
    Saves the artifacts to a PT2Archive format. The artifact can then be loaded
    using ``load_pt2``.

    Args:
        f (str | os.PathLike[str] | IO[bytes]): A file-like object (has to
         implement write and flush) or a string containing a file name.

        exported_programs (Union[ExportedProgram, dict[str, ExportedProgram]]):
         The exported program to save, or a dictionary mapping model name to an
         exported program to save. The exported program will be saved under
         models/\*.json. If only one ExportedProgram is specified, this will
         automatically be named "model".

        aoti_files (Union[list[str], dict[str, list[str]]]): A list of files
         generated by AOTInductor via
         ``torch._inductor.aot_compile(..., {"aot_inductor.package": True})``,
         or a dictionary mapping model name to its AOTInductor generated files.
         If only one set of files is specified, this will automatically be named
         "model".

        extra_files (Optional[Dict[str, Any]]): Map from filename to contents
         which will be stored as part of the pt2.

        opset_version (Optional[Dict[str, int]]): A map of opset names
         to the version of this opset

        pickle_protocol: can be specified to override the default protocol

        executorch_files (Optional[dict[str, bytes]]): Optional executorch
         artifacts to save.

    """
    if exported_programs is None and aoti_files is None and extra_files is None:
        raise AssertionError(
            "No value passed in for `exported_programs`, `aoti_files`, and "
            "`extra_files`, implying that you do not plan on saving anything."
        )

    if not (
        (isinstance(f, (io.IOBase, IO)) and f.writable() and f.seekable())
        or (isinstance(f, (str, os.PathLike)) and os.fspath(f).endswith(".pt2"))
        or (isinstance(f, tempfile._TemporaryFileWrapper) and f.name.endswith(".pt2"))
    ):
        # TODO: turn this into an error
        logger.warning(
            "Expect archive file to be a file ending in .pt2, or is a buffer. "
            "Instead got {%s}",
            f,
        )

    if isinstance(f, (str, os.PathLike)):
        f = os.fspath(f)

    # pyrefly: ignore [bad-argument-type]
    with PT2ArchiveWriter(f) as archive_writer:
        _package_exported_programs(
            archive_writer, exported_programs, pickle_protocol=pickle_protocol
        )
        _package_aoti_files(
            archive_writer,
            aoti_files,
            pickle_protocol=pickle_protocol,
        )
        _package_extra_files(archive_writer, extra_files)
        _package_executorch_files(archive_writer, executorch_files)

    if isinstance(f, (io.IOBase, IO)):
        f.seek(0)
    # pyrefly: ignore [bad-return]
    return f


class AOTICompiledModel:
    # [CLASS DOC] Callable wrapper around a native AOTInductor package loader.
    # State: loader, the native object that owns the compiled model and its
    # constants. Calling it flattens inputs per the stored call spec and runs the
    # model; other methods expose metadata and constant loading.
    """
    Callable AOT Inductor loaded model from a .pt2
    """

    def __init__(self, loader: torch._C._aoti.AOTIModelPackageLoader) -> None:
        # [FUNCTION DOC] Stores the native loader; no other work.
        self.loader = loader

    def __call__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] Runs the compiled model. Reads the input/output tree specs
        # from the loader, reorders keyword arguments to the input spec, flattens
        # args and kwargs, keeps only tensor leaves (non-tensor inputs are silently
        # dropped), runs the loader, and rebuilds the output structure. Returns the
        # structured output.
        call_spec = self.loader.get_call_spec()
        in_spec = pytree.treespec_loads(call_spec[0])
        out_spec = pytree.treespec_loads(call_spec[1])
        flat_inputs = pytree.tree_flatten((args, reorder_kwargs(kwargs, in_spec)))[0]
        flat_inputs = [x for x in flat_inputs if isinstance(x, torch.Tensor)]
        flat_outputs = self.loader.boxed_run(flat_inputs)
        return pytree.tree_unflatten(flat_outputs, out_spec)

    def get_metadata(self) -> dict[str, str]:
        # [FUNCTION DOC] Returns the string metadata mapping of the compiled package.
        return self.loader.get_metadata()

    def load_constants(
        self,
        constants_map: dict[str, torch.Tensor],
        *,
        check_full_update: bool,
        user_managed: bool = False,
        allow_h2d_copy: bool = False,
    ) -> None:
        # [FUNCTION DOC] Loads the given constant tensors (by fully qualified name)
        # into the compiled model via the native loader. check_full_update requires
        # every constant to be provided; user_managed makes the loader keep the
        # caller's tensor pointers (caller keeps them alive); allow_h2d_copy permits
        # copying CPU tensors to the model device. Errors come from the native loader.
        """
        Given a mapping of constant fqns to tensors, load the constants into the model.
        You can use ``get_constant_fqns`` to get the list of constant fqns that
        are needed in the compiled model.

        Args:
            constants_map: A mapping of constant fqns to tensors.
            check_full_update: Whether to add check to see if all the constants
            are updated and have values.
            user_managed: If True, the loader stores the tensor pointers
            directly; the caller must keep them alive.
            allow_h2d_copy: If True, CPU tensors are silently copied to the
            model's device. Useful for loading a CPU ``state_dict()`` into a
            non-CPU model. Incompatible with ``user_managed``.
        """
        self.loader.load_constants(
            constants_map, False, check_full_update, user_managed, allow_h2d_copy
        )

    def get_constant_fqns(self) -> list[str]:
        # [FUNCTION DOC] Returns the fully qualified names of constants the compiled model needs.
        return self.loader.get_constant_fqns()

    def __deepcopy__(self, memo: dict[Any, Any] | None) -> "AOTICompiledModel":
        # [FUNCTION DOC] Logs a warning and returns a new wrapper that shares the
        # same native loader.
        # KNOWN ISSUE: this is not a real deep copy; the copy aliases the loader's state.
        logger.warning(
            "AOTICompiledModel deepcopy warning: AOTICompiledModel.loader is not deepcopied."
        )
        return AOTICompiledModel(self.loader)


@dataclass
class PT2ArchiveContents:
    # [CLASS DOC] Plain result record of load_pt2. Fields: exported_programs
    # (model name to program), aoti_runners (model name to compiled model) and
    # extra_files (file name to string contents).
    exported_programs: dict[str, ExportedProgram]
    aoti_runners: dict[str, AOTICompiledModel]
    extra_files: dict[str, Any]


def _build_file_map(
    archive_reader: PT2ArchiveReader,
    config: schema.PayloadConfig,
    base_dir: str,
) -> dict[str, torch.Tensor]:
    # [FUNCTION DOC] HELPER: For each non-pickled payload in the config, loads its
    # record (once per distinct path name) as a flat one-dimensional tensor of the
    # recorded dtype, moved to the recorded device, with requires_grad restored.
    # Zero-byte records become zero-filled tensors of the recorded size. Returns
    # a mapping of path name to tensor. Raises AssertionError if tensor meta is
    # missing and ValueError if the byte size is not a multiple of the element size.
    """
    Build a map from file path to the payload in flat tensor format.
    """
    file_map: dict[str, torch.Tensor] = {}
    for payload_meta in config.config.values():
        # skip pickled objects
        if payload_meta.use_pickle:
            continue
        # skip files that already exist in the map
        if payload_meta.path_name in file_map:
            continue

        if payload_meta.tensor_meta is None:
            raise AssertionError("payload_meta.tensor_meta cannot be None")

        tensor_meta = payload_meta.tensor_meta
        dtype = deserialize_scalar_type(tensor_meta.dtype)
        device = deserialize_device(tensor_meta.device)
        record_name = os.path.join(base_dir, payload_meta.path_name)
        nbytes = archive_reader.archive_file.get_record_size(record_name)

        if nbytes == 0:
            size = deserialize_size(tensor_meta.sizes)
            tensor = torch.zeros(size, dtype=dtype, device=device)
            if tensor_meta.requires_grad:
                tensor.requires_grad_(True)
        else:
            element_size = torch._utils._element_size(dtype)
            if nbytes % element_size != 0:
                raise ValueError(
                    f"Record {record_name}: size {nbytes} is not a multiple of "
                    f"element size {element_size} for dtype {dtype}"
                )
            numel = nbytes // element_size
            raw = archive_reader.archive_file.get_storage_from_record(
                record_name, numel, dtype
            )
            # get_storage_from_record returns a tensor with empty DispatchKeySet,
            # so we wrap the storage in a proper CPU tensor via set_().
            tensor = torch.empty(0, dtype=dtype)
            tensor.set_(raw.untyped_storage(), 0, (numel,), (1,))
            if tensor_meta.requires_grad:
                tensor.requires_grad_(True)
            if device != torch.device("cpu"):
                tensor = tensor.to(device)

        file_map[payload_meta.path_name] = tensor

    return file_map


def _load_payload_config(
    archive_reader: PT2ArchiveReader,
    config_file: str,
) -> schema.PayloadConfig:
    # [FUNCTION DOC] HELPER: Reads the JSON record config_file and converts it to a PayloadConfig.
    """
    Load and parse a payload config from the archive.
    """
    return _dict_to_dataclass(
        schema.PayloadConfig,
        json.loads(archive_reader.read_string(config_file)),
    )


def _load_state_dict(
    archive_reader: PT2ArchiveReader,
    model_name: str,
) -> dict[str, torch.Tensor] | bytes:
    # [FUNCTION DOC] HELPER: Loads the state dict of a model. If a legacy single
    # weights file exists, warns and returns its raw bytes. Otherwise reads the
    # weights config, builds flat payload tensors, and returns a mapping of weight
    # name to tensor: pickled entries via torch.load (unrestricted unpickling),
    # others as strided views of shared payloads (aliasing preserved), wrapped as
    # Parameter when flagged. Raises AssertionError if the config or tensor meta is missing.
    # Make it BC compatible with legacy weight files
    legacy_weights_file = f"{WEIGHTS_DIR}{model_name}.pt"
    if legacy_weights_file in archive_reader.get_file_names():
        logger.warning(
            "You are loading weight from the legacy format. "
            "Please generate a new pt2 file using torch.export.save()."
        )
        return archive_reader.read_bytes(legacy_weights_file)
    else:
        weights_config_file = WEIGHTS_CONFIG_FILENAME_FORMAT.format(model_name)
        if weights_config_file not in archive_reader.get_file_names():
            raise AssertionError(f"{weights_config_file} not found in PT2 archive")
        weights_config = _load_payload_config(archive_reader, weights_config_file)
        # construct the mapping from file name (e.g. weight_0) to flat weight payload
        state_dict_file_map = _build_file_map(
            archive_reader, weights_config, WEIGHTS_DIR
        )
        # chain the mapping weight FQN -> weight file name -> strided weight payload
        # so that the aliasing of weights is preserved
        state_dict: dict[str, torch.Tensor] = {}
        for weight_fqn, payload_meta in weights_config.config.items():
            if payload_meta.use_pickle:
                weight_bytes = archive_reader.read_bytes(
                    os.path.join(WEIGHTS_DIR, payload_meta.path_name)
                )
                state_dict[weight_fqn] = torch.load(
                    io.BytesIO(weight_bytes), weights_only=False
                )
            else:
                tensor_meta = payload_meta.tensor_meta
                if tensor_meta is None:
                    raise AssertionError(
                        "tensor_meta cannot be None for non-pickled weight"
                    )
                weight_tensor = torch.as_strided(
                    input=state_dict_file_map[payload_meta.path_name],
                    size=deserialize_size(tensor_meta.sizes),
                    stride=deserialize_stride(tensor_meta.strides),
                    storage_offset=deserialize_storage_offset(
                        tensor_meta.storage_offset
                    ),
                )
                if payload_meta.is_param:
                    state_dict[weight_fqn] = torch.nn.Parameter(
                        weight_tensor, requires_grad=tensor_meta.requires_grad
                    )
                else:
                    state_dict[weight_fqn] = weight_tensor

        return state_dict


def _load_constants(
    archive_reader: PT2ArchiveReader,
    model_name: str,
) -> dict[str, torch.Tensor] | bytes:
    # [FUNCTION DOC] HELPER: Loads the constants of a model. A legacy single
    # constants file is returned as raw bytes with a warning. Otherwise reads the
    # constants config and, by file name prefix, rebuilds tensor constants
    # (unpickled or strided views of shared payloads), torchbind objects (native
    # unpickler) and opaque objects (standard unpickle). Returns name to value mapping.
    # Raises AssertionError for a missing config or tensor meta, RuntimeError for
    # an unknown prefix.
    # Make it BC compatible with legacy constant files
    legacy_constants_file = f"{CONSTANTS_DIR}{model_name}.pt"
    if legacy_constants_file in archive_reader.get_file_names():
        logger.warning(
            "You are loading constant from the legacy format. "
            "Please generate a new pt2 file using torch.export.save()."
        )
        return archive_reader.read_bytes(legacy_constants_file)
    else:
        constants_config_file = CONSTANTS_CONFIG_FILENAME_FORMAT.format(model_name)
        if constants_config_file not in archive_reader.get_file_names():
            raise AssertionError(f"{constants_config_file} not found in PT2 archive")
        constants_config = _load_payload_config(archive_reader, constants_config_file)
        # construct the mapping from file name (e.g. constant_0) to constant payload
        constant_file_map = _build_file_map(
            archive_reader, constants_config, CONSTANTS_DIR
        )
        # chain the mapping constant FQN -> constant file name -> strided constant payload
        # so that the aliasing of constants is preserved
        constants: dict[str, torch.Tensor] = {}
        for constant_fqn, payload_meta in constants_config.config.items():
            path_name = payload_meta.path_name
            if path_name.startswith(TENSOR_CONSTANT_FILENAME_PREFIX):
                if payload_meta.use_pickle:
                    constant_bytes = archive_reader.read_bytes(
                        os.path.join(CONSTANTS_DIR, path_name)
                    )
                    constants[constant_fqn] = torch.load(
                        io.BytesIO(constant_bytes), weights_only=False
                    )
                else:
                    tensor_meta = payload_meta.tensor_meta
                    if tensor_meta is None:
                        raise AssertionError(
                            "tensor_meta cannot be None for non-pickled constant"
                        )
                    constant_tensor = torch.as_strided(
                        input=constant_file_map[path_name],
                        size=deserialize_size(tensor_meta.sizes),
                        stride=deserialize_stride(tensor_meta.strides),
                        storage_offset=deserialize_storage_offset(
                            tensor_meta.storage_offset
                        ),
                    )
                    constants[constant_fqn] = constant_tensor

            elif path_name.startswith(CUSTOM_OBJ_FILENAME_PREFIX):
                constant_bytes = archive_reader.read_bytes(
                    os.path.join(CONSTANTS_DIR, path_name)
                )
                constants[constant_fqn] = torch._C._pickle_load_obj(constant_bytes)

            elif path_name.startswith(OPAQUE_OBJ_FILENAME_PREFIX):
                constant_bytes = archive_reader.read_bytes(
                    os.path.join(CONSTANTS_DIR, path_name)
                )
                constants[constant_fqn] = pickle.loads(constant_bytes)

            else:
                raise RuntimeError(f"Unsupported constant type: {path_name}")

        return constants


def _load_exported_programs(
    archive_reader: PT2ArchiveReader,
    file_names: list[str],
    expected_opset_version: dict[str, int] | None,
) -> dict[str, ExportedProgram]:
    # [FUNCTION DOC] HELPER: For every record under the models directory, derives
    # the model name from the file name, loads its sample inputs, deserialized
    # program description, state dict and constants, and deserializes an
    # ExportedProgram checking the expected opset versions. Returns a mapping of
    # model name to program.
    exported_program_files = [
        file for file in file_names if file.startswith(MODELS_DIR)
    ]
    exported_programs = {}
    for file in exported_program_files:
        prefix, suffix = MODELS_FILENAME_FORMAT.split(
            "{}"
        )  # split "models/{}.json" into "models/" and "json"
        model_name = file[
            len(prefix) : -len(suffix)
        ]  # given "models/foo.json" we can now get "foo"

        sample_inputs_file = SAMPLE_INPUTS_FILENAME_FORMAT.format(model_name)
        serialized_sample_inputs = archive_reader.read_bytes(sample_inputs_file)

        from torch._export.serde.serialize import _bytes_to_dataclass

        exported_program_bytes = archive_reader.read_bytes(file)
        serialized_exported_program = _bytes_to_dataclass(
            schema.ExportedProgram, exported_program_bytes
        )
        state_dict = _load_state_dict(archive_reader, model_name)
        constants = _load_constants(archive_reader, model_name)

        ep = ExportedProgramDeserializer(expected_opset_version).deserialize(
            serialized_exported_program,
            state_dict,
            constants,
            serialized_sample_inputs,
        )

        exported_programs[model_name] = ep

    return exported_programs


def _load_extra_files(
    archive_reader: PT2ArchiveReader, file_names: list[str]
) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Reads every record under the extra directory as a
    # string and returns a mapping of name (prefix removed) to contents.
    extra_files = [file for file in file_names if file.startswith(EXTRA_DIR)]

    extra_file_contents: dict[str, Any] = {}
    for file in extra_files:
        contents = archive_reader.read_string(file)
        extra_file_contents[file[len(EXTRA_DIR) :]] = contents

    return extra_file_contents


def _load_aoti(
    file: str,
    model_name: str,
    run_single_threaded: bool,
    num_runners: int,
    device_idx: int,
) -> AOTICompiledModel:
    # [FUNCTION DOC] HELPER: Creates a native package loader for one model in the
    # package file and wraps it. Compares the device information stored in the
    # package metadata with the current device and logs a warning for each
    # mismatching key. Returns the AOTICompiledModel.
    loaded_metadata = torch._C._aoti.AOTIModelPackageLoader.load_metadata_from_package(  # type: ignore[attr-defined]
        file, model_name
    )

    aoti_compiled_model = AOTICompiledModel(
        torch._C._aoti.AOTIModelPackageLoader(
            file,
            model_name,
            run_single_threaded,
            num_runners,
            device_idx,
        )
    )

    device = loaded_metadata["AOTI_DEVICE_KEY"]
    from torch._inductor.codecache import get_device_information

    current_device_info = get_device_information(device)

    for k, v in current_device_info.items():
        if k in loaded_metadata:
            if v != loaded_metadata[k]:
                logger.warning(
                    "Device information mismatch for %s: %s vs %s. "
                    "This could cause some issues when loading the AOTInductor compiled artifacts.",
                    k,
                    v,
                    loaded_metadata[k],
                )

    return aoti_compiled_model


def load_pt2(
    f: FileLike,
    *,
    expected_opset_version: dict[str, int] | None = None,
    run_single_threaded: bool = False,
    num_runners: int = 1,
    device_index: int = -1,
    load_weights_from_disk: bool = False,
) -> PT2ArchiveContents:  # type: ignore[type-arg]
    # [FUNCTION DOC] Public load entry point. Opens the archive, checks its version
    # equals the current one, loads exported programs and extra files, finds AOTI
    # model names, and creates compiled runners (buffers are first copied to a
    # temporary .pt2 file because the native loader cannot read buffers). With
    # load_weights_from_disk it also loads weight files and pushes the weights
    # into each runner as user-managed constants. Returns PT2ArchiveContents.
    # Raises ValueError on version mismatch; warns if f is not a .pt2 path or buffer.
    # KNOWN ISSUE: the docstring omits load_weights_from_disk, and the weight files
    # are read with torch.load using default arguments.
    """
    Loads all the artifacts previously saved with ``package_pt2``.

    Args:
        f (str | os.PathLike[str] | IO[bytes]): A file-like object (has to
         implement write and flush) or a string containing a file name.

        expected_opset_version (Optional[Dict[str, int]]): A map of opset names
         to expected opset versions

        num_runners (int): Number of runners to load AOTInductor artifacts

        run_single_threaded (bool): Whether the model should be run without
            thread synchronization logic. This is useful to avoid conflicts with
            CUDAGraphs.

        device_index (int): The index of the device to which the PT2 package is
            to be loaded. By default, `device_index=-1` is used, which corresponds
            to the device `cuda` when using CUDA. Passing `device_index=1` would
            load the package to `cuda:1`, for example.

    Returns:
        A ``PT2ArchiveContents`` object which contains all the objects in the PT2.
    """

    from torch._inductor.cpp_builder import normalize_path_separator

    if not (
        (isinstance(f, (io.IOBase, IO)) and f.readable() and f.seekable())
        or (isinstance(f, (str, os.PathLike)) and os.fspath(f).endswith(".pt2"))
    ):
        # TODO: turn this into an error in 2.9
        logger.warning(
            "Unable to load package. f must be a buffer or a file ending in "
            ".pt2. Instead got {%s}",
            f,
        )

    if isinstance(f, (str, os.PathLike)):
        f = os.fspath(f)

    weights = {}
    weight_maps = {}
    # pyrefly: ignore [bad-argument-type]
    with PT2ArchiveReader(f) as archive_reader:
        version = archive_reader.read_string(ARCHIVE_VERSION_PATH)
        if version != ARCHIVE_VERSION_VALUE:
            raise ValueError(
                f"Saved archive version {version} does not match our current "
                f"archive version {ARCHIVE_VERSION_VALUE}."
            )

        file_names = archive_reader.get_file_names()

        exported_programs = _load_exported_programs(
            archive_reader, file_names, expected_opset_version
        )
        extra_files = _load_extra_files(archive_reader, file_names)

        # Get a list of AOTI model names
        aoti_model_names: set[str] = set()
        for file in file_names:
            if file.startswith(AOTINDUCTOR_DIR):
                file_end = file[
                    len(AOTINDUCTOR_DIR) :
                ]  # remove data/aotinductor/ prefix
                file_end = normalize_path_separator(
                    file_end
                )  # Win32 need normalize path before split.
                model_name = file_end.split("/")[
                    0
                ]  # split "model_name/...cpp" into "model_name"
                aoti_model_names.add(model_name)
                if load_weights_from_disk and file.endswith("weights_config.json"):
                    weight_map = json.loads(archive_reader.read_string(file))
                    weight_maps[model_name] = weight_map
            elif load_weights_from_disk and file.startswith(WEIGHTS_DIR):
                weight_file_name = file[
                    len(WEIGHTS_DIR) :
                ]  # remove data/weights/ prefix
                weight_bytes = archive_reader.read_bytes(file)
                loaded_weight = torch.load(io.BytesIO(weight_bytes))
                weights[weight_file_name] = loaded_weight

    if isinstance(f, (io.IOBase, IO)):
        if len(aoti_model_names) > 0:
            # Workaround for AOTIModelPackageLoader not reading buffers
            with tempfile.NamedTemporaryFile(suffix=".pt2") as tf:
                f.seek(0)
                tf.write(f.read())
                f.seek(0)
                logger.debug("Writing buffer to tmp file located at %s.", tf.name)

                aoti_runners = {
                    model_name: _load_aoti(
                        tf.name,
                        model_name,
                        run_single_threaded,
                        num_runners,
                        device_index,
                    )
                    for model_name in aoti_model_names
                }
        else:
            aoti_runners = {}
    else:
        aoti_runners = {
            model_name: _load_aoti(
                f,
                model_name,
                run_single_threaded,
                num_runners,
                device_index,
            )
            for model_name in aoti_model_names
        }

    if weight_maps:
        for model_name in aoti_model_names:
            model_weights = {}
            for weight_name, (file, shape, stride, storage_offset) in weight_maps[
                model_name
            ].items():
                weight = weights[file]
                model_weights[weight_name] = weight.as_strided(
                    shape, stride, storage_offset
                )

            # user_managed=True ensures the weights updates are shared by all runners.
            aoti_runners[model_name].load_constants(
                model_weights, check_full_update=True, user_managed=True
            )

    return PT2ArchiveContents(exported_programs, aoti_runners, extra_files)


def load_weights_to_pt2_contents(
    pt2_contents: PT2ArchiveContents, weights_map: dict[str, Any]
) -> None:
    # [FUNCTION DOC] For each model name in weights_map, loads the given weights
    # into that model's AOTI runner as user-managed constants with a full-update
    # check. Raises RuntimeError if a model name has no runner in the contents.
    # KNOWN ISSUE: the docstring does not document the weights_map argument.
    """
    Load weights into the models in PT2 archive contents

    Args:
        pt2_contents (PT2ArchiveContents): The contents of the PT2 archive.
    """
    for model_name, weights in weights_map.items():
        if model_name not in pt2_contents.aoti_runners:
            raise RuntimeError(f"Model {model_name} not found in PT2 archive contents.")
        pt2_contents.aoti_runners[model_name].load_constants(
            weights, check_full_update=True, user_managed=True
        )
