# mypy: allow-untyped-defs
# [MODULE DOC] Terminal visualization of how a 1D or 2D DTensor is sharded across devices.
# Provides the public visualize_sharding plus plain-text (tabulate) and colored (rich and
# matplotlib) table renderers. Optional dependencies are imported lazily inside functions.
# Import-time side effects: none beyond importing numpy and DTensor utilities.
import importlib.util

import numpy as np

from torch._prims_common import ShapeType
from torch.distributed.tensor._utils import _compute_local_shape_and_global_offset


__all__ = ["visualize_sharding"]

Color = tuple[float, float, float]


def _create_table(
    shards: list[tuple[tuple[int, int], tuple[int, int], int]], device_kind: str = ""
):
    # [FUNCTION DOC] HELPER: Builds a plain-text table string from shards, each given as a row
    # range, a column range and a device id. Rows and columns are the sorted unique ranges;
    # each cell lists the device ids (prefixed by device_kind) covering it, comma separated.
    # Requires the tabulate package (import error otherwise). Returns the table string.
    """
    Creates a tabulate table given row and column ranges with device name
    """
    from tabulate import tabulate

    # Extract unique row and column ranges
    row_ranges = sorted({block[0] for block in shards})
    col_ranges = sorted({block[1] for block in shards})

    # Create a matrix initialized with empty strings
    matrix = [["" for _ in col_ranges] for _ in row_ranges]

    # Fill the matrix with values
    for block in shards:
        row_index = row_ranges.index(block[0])
        col_index = col_ranges.index(block[1])
        if matrix[row_index][col_index] == "":
            matrix[row_index][col_index] = device_kind + ":" + str(block[2])
        else:
            matrix[row_index][col_index] += "," + str(block[2])

    # Prepare headers
    row_headers = [f"Row {r[0]}-{r[1]}" for r in row_ranges]
    col_headers = [f"Col {c[0]}-{c[1]}" for c in col_ranges]

    return tabulate(matrix, headers=col_headers, showindex=row_headers)


def make_color_iter(color_map, num_rows, num_cols):
    # [FUNCTION DOC] HELPER: Generator yielding one color per table cell: calls the color_map
    # callable with each integer index from zero up to rows times columns (exclusive).
    num_colors = num_rows * num_cols
    for idx in range(num_colors):
        yield color_map(idx)


def _canonicalize_color(color: Color) -> str:
    # [FUNCTION DOC] HELPER: Returns a string color unchanged; converts an RGB triple of floats
    # in the unit interval to an upper-case hex string with a leading hash (values truncated
    # to integers in 0-255).
    if isinstance(color, str):
        return color
    r, g, b = (int(a * 255) for a in color)
    return f"#{r:02X}{g:02X}{b:02X}"


def _get_text_color(color: str) -> str:
    # [FUNCTION DOC] HELPER: Given a hex background color string with a leading hash, returns
    # black for light backgrounds (perceived luminance above 186) and white otherwise.
    r, g, b = map(lambda x: int(x, 16), (color[1:3], color[3:5], color[5:7]))  # noqa: C417
    if (r * 0.299 + g * 0.587 + b * 0.114) > 186:
        return "#000000"
    return "#ffffff"


def _create_rich_table(
    shape: ShapeType,
    shards: list[tuple[tuple[int, int], tuple[int, int], int]],
    device_kind: str = "",
    scale: float = 1.0,
    min_width: int = 9,
    max_width: int = 80,
):
    # [FUNCTION DOC] HELPER: Prints a rich table to the console in which each cell is one unique
    # row/column range block, sized proportionally to its share of the tensor shape (scale
    # controls height; max_width caps console width; min_width is accepted but unused). Cells
    # show device_kind and the device ids owning the block, colored from the matplotlib tab20b
    # map when the console supports color, else drawn with box lines. Requires rich and
    # matplotlib. Returns nothing.
    # KNOWN ISSUE: min_width is never used; an empty-width division is possible if shape[0] is 0.
    import matplotlib
    import rich.align
    import rich.box
    import rich.console
    import rich.padding
    import rich.style
    import rich.table

    dtensor_height = shape[0]
    dtensor_width = shape[1] if len(shape) == 2 else 1

    row_ranges = sorted({s[0] for s in shards})
    col_ranges = sorted({s[1] for s in shards})
    num_rows, num_cols = len(row_ranges), len(col_ranges)

    console = rich.console.Console(width=max_width)
    use_color = console.color_system
    color_iter = make_color_iter(matplotlib.colormaps["tab20b"], num_rows, num_cols)

    base_height = int(10 * scale)
    aspect_ratio = (shape[1] if len(shape) == 2 else 1) / shape[0]
    base_width = int(base_height * aspect_ratio)
    height_to_width_ratio = 2.5

    table = rich.table.Table(
        show_header=False,
        show_lines=not use_color,
        padding=0,
        highlight=not use_color,
        pad_edge=False,
        box=rich.box.SQUARE if not use_color else None,
    )
    for row in range(num_rows):
        table_row = []
        for col in range(num_cols):
            entry = (
                device_kind
                + ":"
                + ",".join(
                    [
                        str(device_id)
                        for row_range, col_range, device_id in shards
                        if row_range == row_ranges[row] and col_range == col_ranges[col]
                    ]
                )
            )
            width = (col_ranges[col][1] - col_ranges[col][0]) / dtensor_width
            width = int(width * base_width * height_to_width_ratio)
            height = (row_ranges[row][1] - row_ranges[row][0]) / dtensor_height
            height = int(height * base_height)
            left_padding, remainder = divmod(width - len(entry) - 2, 2)
            right_padding = left_padding + remainder
            top_padding, remainder = divmod(height - 2, 2)
            bottom_padding = top_padding + remainder
            if use_color:
                color = _canonicalize_color(next(color_iter)[:3])
                text_color = _get_text_color(color)
                top_padding += 1
                bottom_padding += 1
                left_padding += 1
                right_padding += 1
            else:
                color = None
                text_color = None
            padding = (
                max(top_padding, 0),
                max(right_padding, 0),
                max(bottom_padding, 0),
                max(left_padding, 0),
            )
            table_row.append(
                rich.padding.Padding(
                    rich.align.Align(entry, "center", vertical="middle"),
                    padding,
                    style=rich.style.Style(bgcolor=color, color=text_color),
                )
            )
        table.add_row(*table_row)
    console.print(table, end="\n\n")


def visualize_sharding(dtensor, header="", use_rich: bool = False):
    # [FUNCTION DOC] Prints the sharding layout of a 1D or 2D DTensor. Returns silently for empty
    # tensors, for ranks outside the mesh, and for all ranks except the one at coordinate zero
    # in every mesh dimension. Raises RuntimeError for tensors with three or more dims. Computes
    # each device's local shape and global offset, converts them to inclusive row/column ranges
    # (1D tensors are treated as one column), then prints a rich table if use_rich is set and
    # rich plus matplotlib are installed, else a tabulate table. Raises ValueError if neither
    # backend is available.
    # KNOWN ISSUE: the header argument is never used. Ranges are inclusive (end = offset + size - 1).
    """
    Visualizes sharding in the terminal for :class:`DTensor` that are 1D or 2D.

    .. note:: This requires the ``tabulate`` package, or ``rich`` and ``matplotlib``.
              No sharding info will be printed for empty tensors
    """
    if dtensor.numel() == 0:  # Do not print empty dtensors.
        return

    if len(dtensor.shape) >= 3:
        raise RuntimeError("visualize sharding supports only 1D or 2D DTensor")

    if dtensor.device_mesh.get_coordinate() is None:  # current rank is not in the mesh
        return

    # Only display the visualization once for each DTensor, on the rank whose
    # coordinate is 0 on all dimensions. For example, if the mesh is a full mesh,
    # we will only print on rank 0.
    local_rank_zero_on_all_dim = all(
        dtensor.device_mesh.get_local_rank(mesh_dim=dim) == 0
        for dim in range(dtensor.device_mesh.ndim)
    )
    if not local_rank_zero_on_all_dim:
        return

    device_coords = {
        int(device_index.item()): list(coord)
        for coord, device_index in np.ndenumerate(
            np.array(dtensor.device_mesh.mesh.tolist())
        )
    }

    device_shard_shape_and_offsets = {
        device_index: _compute_local_shape_and_global_offset(
            dtensor.shape,
            dtensor.device_mesh.shape,
            lambda i: device_coords[device_index][i],
            dtensor.placements,
        )
        for device_index in device_coords
    }

    # Extend shards in a 1D tensor to 2D
    device_shard_shape_and_offsets = {
        device_index: (
            shape if len(shape) == 2 else (shape[0], 1),
            offset if len(offset) == 2 else (offset[0], 0),
        )
        for device_index, (shape, offset) in device_shard_shape_and_offsets.items()
    }

    shards = [
        (
            (offset[0], offset[0] + shape[0] - 1),
            (offset[1], offset[1] + shape[1] - 1),
            device_index,
        )
        for device_index, (shape, offset) in device_shard_shape_and_offsets.items()
    ]

    if (
        importlib.util.find_spec("rich")
        and importlib.util.find_spec("matplotlib")
        and use_rich
    ):
        _create_rich_table(
            dtensor.shape, shards, device_kind=dtensor.device_mesh.device_type
        )
    elif importlib.util.find_spec("tabulate"):
        print(_create_table(shards, device_kind=dtensor.device_mesh.device_type))
    else:
        raise ValueError("`visualize_sharding` requires either `rich` or `tabulate`.")
