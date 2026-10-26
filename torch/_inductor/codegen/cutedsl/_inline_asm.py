# [MODULE DOC] Lowers elementwise inline PTX assembly over CuteDSL fragments via MLIR/LLVM inline asm:
# constraint parsing, bit-level operand packing, output conversion (including E8M0 decoding) and the
# user-op inline_asm_elementwise_intrinsic. Requires cutlass at import time; no other import-time side effects.
"""Lower elementwise inline PTX over CuteDSL fragments.

Each assembly invocation consumes `pack` adjacent fragment elements. LLVM
constraints select the physical register type for every operand, so inputs are
bitcast into those registers rather than numerically converted. Multiple outputs
are returned as an LLVM struct and extracted before rebuilding the fragment.
Partial packs are zero-padded, and outputs for padded elements are discarded.

Integer register results are narrowed to the requested element width. E8M0 codes
1 through 254 map directly to the matching Float32 exponent bits; codes 0 and 255
map explicitly to `2^-127` and NaN. This lets fused arithmetic use exact Float32
values while preserving E8M0 as the stored result type.
"""

import functools
import hashlib
import math
from pathlib import Path

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm, vector
from cutlass.cute.tensor import _infer_broadcast_shape
from cutlass.cutlass_dsl import dsl_user_op, T


@functools.cache
def inline_asm_cache_key() -> str:
    # [FUNCTION DOC] HELPER: Returns (cached) hex SHA-256 of this module's source file, so generated-kernel cache keys change when it changes.
    """Include this module's source in generated kernel cache keys."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


# PTX register classes, as accepted by the NVPTX inline-asm backend.
CONSTRAINT_TYPES = {
    "h": T.i16,
    "r": T.i32,
    "l": T.i64,
    "f": T.f32,
    "d": T.f64,
}


def split_constraints(constraints: str) -> tuple[list[str], list[str]]:
    # [FUNCTION DOC] HELPER: Splits a comma-separated constraint list into output letters (entries starting with an equals sign) and input letters,
    # stripping equals and ampersand markers. Raises NotImplementedError for any letter not in CONSTRAINT_TYPES.
    """Split an LLVM constraint list into output and input register classes."""
    outputs, inputs = [], []
    for entry in constraints.split(","):
        entry = entry.strip()
        target = outputs if entry.startswith("=") else inputs
        letter = entry.lstrip("=&")
        if letter not in CONSTRAINT_TYPES:
            raise NotImplementedError(
                f"CuteDSL inline asm supports constraints {sorted(CONSTRAINT_TYPES)}, "
                f"got {entry!r}"
            )
        target.append(letter)
    return outputs, inputs


def bitcast_to(value: ir.Value, target: ir.Type) -> ir.Value:
    # [FUNCTION DOC] HELPER: Returns the value unchanged if already of the target type, else bitcasts it (no numeric conversion).
    """Reinterpret a value's bits as an equally wide type."""
    if value.type == target:
        return value
    return arith.bitcast(target, value)


def narrow_integer(value: ir.Value, width: int) -> ir.Value:
    # [FUNCTION DOC] HELPER: Truncates an integer value wider than width to a signless integer of that width; narrower or equal values pass through.
    # Raises TypeError if the value is not an integer type.
    """Truncate an integer register to `width` bits."""
    if not ir.IntegerType.isinstance(value.type):
        raise TypeError(f"expected an integer inline-asm result, got {value.type}")
    if ir.IntegerType(value.type).width > width:
        return arith.trunci(ir.IntegerType.get_signless(width), value)
    return value


def decode_e8m0(value: ir.Value) -> ir.Value:
    # [FUNCTION DOC] HELPER: Decodes an E8M0 code (narrowed to 8 bits, zero-extended to 32) to Float32 bits: code shifted into the exponent field;
    # code 0 becomes the bit pattern of 2^-127; code 255 becomes a quiet NaN. Returns a Float32 value.
    """Decode one biased E8M0 bit pattern to its exact Float32 value."""
    code = arith.extui(T.i32(), narrow_integer(value, 8))
    bits = arith.shli(code, arith.constant(T.i32(), 23))
    is_zero = arith.cmpi(arith.CmpIPredicate.eq, code, arith.constant(T.i32(), 0))
    bits = arith.select(is_zero, arith.constant(T.i32(), 0x00400000), bits)
    is_nan = arith.cmpi(arith.CmpIPredicate.eq, code, arith.constant(T.i32(), 255))
    bits = arith.select(is_nan, arith.constant(T.i32(), 0x7FC00000), bits)
    return arith.bitcast(T.f32(), bits)


def convert_output(value: ir.Value, result_type) -> ir.Value:
    # [FUNCTION DOC] HELPER: Converts a raw asm register result to the logical element type: E8M0 is decoded to Float32; otherwise integers
    # are narrowed to the type width and the bits are reinterpreted as the type's MLIR type.
    """Convert a register result to the requested logical element type."""
    if result_type == cutlass.Float8E8M0FNU:
        return decode_e8m0(value)
    target = result_type.mlir_type
    if value.type == target:
        return value
    if ir.IntegerType.isinstance(value.type):
        value = narrow_integer(value, result_type.width)
    return bitcast_to(value, target)


def fragment_element(source, index: int) -> ir.Value:
    # [FUNCTION DOC] HELPER: Returns element index of a TensorSSA vector; an existing IR value is returned as is; Python int/float becomes a
    # Float32 constant; other objects are unwrapped via their IR value accessor.
    """Extract element `index` from a TensorSSA, or unwrap a scalar value."""
    if isinstance(source, cute.TensorSSA):
        return vector.extract(source.ir_value(), [], [index])
    if isinstance(source, ir.Value):
        return source
    if isinstance(source, (int, float)):
        return cutlass.Float32(source).ir_value()
    return source.ir_value()


def zero_for_constraint(letter: str) -> ir.Value:
    # [FUNCTION DOC] HELPER: Returns a zero constant of the register type for the constraint letter (float zero for f and d, integer zero otherwise).
    """Create a zero in the register type named by an inline-asm constraint."""
    target = CONSTRAINT_TYPES[letter]()
    return arith.constant(target, 0.0 if letter in ("f", "d") else 0)


def packed_operands(
    sources,
    input_letters: list[str],
    scalar_sources: tuple[bool, ...],
    base: int,
    count: int,
    pack: int,
) -> list[ir.Value]:
    # [FUNCTION DOC] HELPER: Builds the operand list for one asm invocation: source-major, pack lanes per source, lane index base plus lane.
    # Scalar sources always use element 0. Lanes past count are zero filled. Each operand is bitcast to its constraint's register type.
    """Gather one source-major inline-asm pack, padding a partial tail with zero."""
    operands = []
    for source_index, source in enumerate(sources):
        for lane in range(pack):
            letter = input_letters[source_index * pack + lane]
            index = base + lane
            operands.append(
                bitcast_to(
                    fragment_element(
                        source, 0 if scalar_sources[source_index] else index
                    )
                    if index < count
                    else zero_for_constraint(letter),
                    CONSTRAINT_TYPES[letter](),
                )
            )
    return operands


@dsl_user_op
def inline_asm_elementwise_intrinsic(
    *sources,
    asm: str,
    constraints: str,
    result_type,
    is_pure: bool = True,
    pack: int = 1,
    scalar_sources: tuple[bool, ...] = (),
    loc=None,
    ip=None,
):
    # [FUNCTION DOC] Emits elementwise inline PTX over TensorSSA fragments (and/or scalars). Validates: pack at least 1, output constraint count equals
    # pack, input constraint count equals number of sources times pack, scalar_sources empty or same length as sources (else ValueError).
    # Fragments are broadcast to a common shape, then the asm runs once per group of pack elements; multi-output results are an LLVM struct
    # unpacked per field; outputs for padded tail lanes are dropped; each is converted to result_type (E8M0 computed as Float32).
    # Returns a TensorSSA of the broadcast shape, or a scalar of the compute type if no fragment sources. has_side_effects is the negation of is_pure.
    """Apply an inline PTX block elementwise across a TensorSSA fragment.

    Args:
        sources: Broadcastable TensorSSA fragments or repeated scalar values.
        asm: PTX text using `$N` operand syntax, output first.
        constraints: LLVM constraint list, e.g. `"=h,r"`.
        result_type: Logical cutlass numeric type of the produced fragment.
            E8M0 integer results are decoded to Float32 for fused consumers.
        is_pure: Whether the block may be reordered and CSEd.
        pack: Elements consumed per asm invocation. Missing tail elements are
            zero-padded and their corresponding outputs are discarded.
        scalar_sources: Sources whose physical TensorSSA value represents one
            logical scalar and must be repeated for every lane.
    """
    if pack < 1:
        raise ValueError(f"CuteDSL inline asm requires pack >= 1, got pack={pack}")
    output_letters, input_letters = split_constraints(constraints)
    if len(output_letters) != pack:
        raise ValueError(
            f"inline asm pack={pack} requires {pack} output constraints, "
            f"got {len(output_letters)} in {constraints!r}"
        )
    expected_inputs = len(sources) * pack
    if len(input_letters) != expected_inputs:
        raise ValueError(
            f"inline asm pack={pack} with {len(sources)} sources requires "
            f"{expected_inputs} input constraints, got {len(input_letters)} "
            f"in {constraints!r}"
        )
    if not scalar_sources:
        scalar_sources = (False,) * len(sources)
    elif len(scalar_sources) != len(sources):
        raise ValueError(
            f"inline asm received {len(scalar_sources)} scalar-source flags "
            f"for {len(sources)} sources"
        )

    output_types = [CONSTRAINT_TYPES[letter]() for letter in output_letters]
    asm_result_type = output_types[0]
    if len(output_types) > 1:
        fields = ", ".join(str(output_type) for output_type in output_types)
        asm_result_type = ir.Type.parse(f"!llvm.struct<({fields})>")
    compute_type = (
        cutlass.Float32 if result_type == cutlass.Float8E8M0FNU else result_type
    )
    fragments = [
        source
        for source, is_scalar in zip(sources, scalar_sources)
        if isinstance(source, cute.TensorSSA) and not is_scalar
    ]
    if fragments:
        shape = fragments[0].shape
        if any(source.shape != shape for source in fragments[1:]):
            shape = _infer_broadcast_shape(*(source.shape for source in fragments))
            sources = tuple(
                source.broadcast_to(shape)
                if isinstance(source, cute.TensorSSA) and not is_scalar
                else source
                for source, is_scalar in zip(sources, scalar_sources)
            )
        first_fragment = next(
            source
            for source, is_scalar in zip(sources, scalar_sources)
            if isinstance(source, cute.TensorSSA) and not is_scalar
        )
        count = math.prod(ir.VectorType(first_fragment.ir_value().type).shape)
    else:
        count = 1
        shape = None

    converted = []
    for base in range(0, count, pack):
        produced = llvm.inline_asm(
            asm_result_type,
            packed_operands(sources, input_letters, scalar_sources, base, count, pack),
            asm,
            constraints,
            has_side_effects=not is_pure,
            is_align_stack=False,
        )
        outputs = [produced]
        if len(output_types) > 1:
            outputs = [
                llvm.extractvalue(output_type, produced, [index])
                for index, output_type in enumerate(output_types)
            ]
        valid_outputs = min(pack, count - base)
        converted.extend(
            convert_output(output, result_type) for output in outputs[:valid_outputs]
        )

    if shape is None:
        return compute_type(converted[0])
    vector_type = ir.VectorType.get([count], compute_type.mlir_type)
    return cute.TensorSSA(
        vector.from_elements(vector_type, converted), shape, compute_type
    )
