# mypy: allow-untyped-defs
# [MODULE DOC]
# Renders many Measurement objects as aligned text tables. Public: Colorize (coloring mode
# enum) and Compare (collects measurements, merges replicates, groups them by label and prints
# one table per label). Internal: _Column and _Row (per-cell and per-row formatting), Table
# (builds rows and columns and renders a table) and ANSI color constants. No import-time side
# effects beyond defining names.
"""Display class to aggregate and print the results of many measurements."""
import collections
import enum
import itertools as it

from torch.utils.benchmark.utils import common
from torch import tensor as _tensor
import operator

__all__ = ["Colorize", "Compare"]

BEST = "\033[92m"
GOOD = "\033[34m"
BAD = "\033[2m\033[91m"
VERY_BAD = "\033[31m"
BOLD = "\033[1m"
TERMINATE = "\033[0m"


class Colorize(enum.Enum):
    # [CLASS DOC] Enumeration of table coloring modes: no color, color relative to the best
    # value in each column, or relative to the best value in each row.
    NONE = "none"
    COLUMNWISE = "columnwise"
    ROWWISE = "rowwise"


# Classes to separate internal bookkeeping from what is rendered.
class _Column:
    # [CLASS DOC] HELPER: One table column of measurements, stored as groups of rows (one group
    # per thread count), with the formatting settings shared by the table. Holds a number
    # format template sized to the widest value so all cells align.
    def __init__(
        self,
        grouped_results: list[tuple[common.Measurement | None, ...]],
        time_scale: float,
        time_unit: str,
        trim_significant_figures: bool,
        highlight_warnings: bool,
    ) -> None:
        # [FUNCTION DOC] HELPER: Stores the grouped and flattened results and the scale and unit.
        # Warning highlighting is enabled only if requested and at least one measurement has a
        # warning. Computes the digit widths from the largest leading digits and (when trimming)
        # the fewest decimals allowed by the significant figures, and builds the format template.
        self._grouped_results = grouped_results
        self._flat_results = [*it.chain.from_iterable(grouped_results)]
        self._time_scale = time_scale
        self._time_unit = time_unit
        self._trim_significant_figures = trim_significant_figures
        self._highlight_warnings = (
            highlight_warnings
            and any(r.has_warnings for r in self._flat_results if r)
        )
        leading_digits = [
            int(_tensor(r.median / self._time_scale).log10().ceil()) if r else None
            for r in self._flat_results
        ]
        unit_digits = max(d for d in leading_digits if d is not None)
        decimal_digits = min(
            max(m.significant_figures - digits, 0)
            for digits, m in zip(leading_digits, self._flat_results, strict=True)
            if (m is not None) and (digits is not None)
        ) if self._trim_significant_figures else 1
        length = unit_digits + decimal_digits + (1 if decimal_digits else 0)
        self._template = f"{{:>{length}.{decimal_digits}f}}{{:>{7 if self._highlight_warnings else 0}}}"

    def get_results_for(self, group):
        # [FUNCTION DOC] HELPER: Returns the column's measurements (some may be none) for the
        # given row group index.
        return self._grouped_results[group]

    def num_to_str(self, value: float | None, estimated_sigfigs: int, spread: float | None):
        # [FUNCTION DOC] HELPER: Formats a scaled time using the column template, optionally
        # trimmed to the estimated significant figures and followed by a variance marker with the
        # spread as a percent when warnings are highlighted. A missing value yields blank space of
        # the same width as a real cell.
        if value is None:
            return " " * len(self.num_to_str(1, estimated_sigfigs, None))

        if self._trim_significant_figures:
            value = common.trim_sigfig(value, estimated_sigfigs)

        return self._template.format(
            value,
            f" (! {spread * 100:.0f}%)" if self._highlight_warnings and spread is not None else "")


def optional_min(seq):
    # [FUNCTION DOC] HELPER: Returns the smallest element of the iterable, or none if empty.
    l = list(seq)
    return None if len(l) == 0 else min(l)


class _Row:
    # [CLASS DOC] HELPER: One table row: a list of measurements (none where missing) across the
    # columns, the thread-count group it belongs to, rendering options (environment label
    # shown or not, widths, time scale, coloring mode) and, on the first row of a group, the
    # thread count for a separator line. Columns are attached later by register_columns.
    def __init__(self, results, row_group, render_env, env_str_len,
                 row_name_str_len, time_scale, colorize, num_threads=None) -> None:
        # [FUNCTION DOC] HELPER: Stores the row data and options; columns start empty.
        super().__init__()
        self._results = results
        self._row_group = row_group
        self._render_env = render_env
        self._env_str_len = env_str_len
        self._row_name_str_len = row_name_str_len
        self._time_scale = time_scale
        self._colorize = colorize
        self._columns: tuple[_Column, ...] = ()
        self._num_threads = num_threads

    def register_columns(self, columns: tuple[_Column, ...]) -> None:
        # [FUNCTION DOC] HELPER: Attaches the table's column objects used for formatting.
        self._columns = columns

    def as_column_strings(self):
        # [FUNCTION DOC] HELPER: Returns the row's cell strings: first the row name (with the
        # environment in parentheses when rendered), then one formatted number per column, with
        # blanks for missing measurements.
        concrete_results = [r for r in self._results if r is not None]
        env = f"({concrete_results[0].env})" if self._render_env else ""
        env = env.ljust(self._env_str_len + 4)
        output = ["  " + env + concrete_results[0].as_row_name]
        for m, col in zip(self._results, self._columns or (), strict=False):
            if m is None:
                output.append(col.num_to_str(None, 1, None))
            else:
                output.append(col.num_to_str(
                    m.median / self._time_scale,
                    m.significant_figures,
                    m.iqr / m.median if m.has_warnings else None
                ))
        return output

    @staticmethod
    def color_segment(segment, value, best_value):
        # [FUNCTION DOC] HELPER: Wraps the text in ANSI color codes by how it compares to the best
        # value: best (within one percent or 100 ns), good (within ten percent), very bad (five
        # times or more), bad (twice or more); otherwise returns the text unchanged.
        if value <= best_value * 1.01 or value <= best_value + 100e-9:
            return BEST + BOLD + segment + TERMINATE * 2
        if value <= best_value * 1.1:
            return GOOD + BOLD + segment + TERMINATE * 2
        if value >= best_value * 5:
            return VERY_BAD + BOLD + segment + TERMINATE * 2
        if value >= best_value * 2:
            return BAD + segment + TERMINATE * 2

        return segment

    def row_separator(self, overall_width):
        # [FUNCTION DOC] HELPER: Returns a one-element list with a dashed line labeled with the
        # thread count when this row starts a thread group; otherwise an empty list.
        return (
            [f"{self._num_threads} threads: ".ljust(overall_width, "-")]
            if self._num_threads is not None else []
        )

    def finalize_column_strings(self, column_strings, col_widths):
        # [FUNCTION DOC] HELPER: Pads the row name left-aligned and centers each other cell to
        # its column width, applying color according to the coloring mode: row mode compares to
        # the row's minimum median, column mode to the minimum within the same column and thread
        # group. Returns the list of final cell strings.
        best_values = [-1 for _ in column_strings]
        if self._colorize == Colorize.ROWWISE:
            row_min = min(r.median for r in self._results if r is not None)
            best_values = [row_min for _ in column_strings]
        elif self._colorize == Colorize.COLUMNWISE:
            best_values = [
                optional_min(r.median for r in column.get_results_for(self._row_group) if r is not None)
                for column in (self._columns or ())
            ]

        row_contents = [column_strings[0].ljust(col_widths[0])]
        for col_str, width, result, best_value in zip(column_strings[1:], col_widths[1:], self._results, best_values, strict=False):
            col_str = col_str.center(width)
            if self._colorize != Colorize.NONE and result is not None and best_value is not None:
                col_str = self.color_segment(col_str, result.median, best_value)
            row_contents.append(col_str)
        return row_contents


class Table:
    # [CLASS DOC] HELPER: Builds and renders one table for measurements that share a label. Rows
    # are keyed by thread count, environment and row name; columns by description. Attributes:
    # results, label, time unit and scale (chosen from the smallest median), row and column
    # keys, and the built rows and columns. Use render to get the text.
    def __init__(
            self,
            results: list[common.Measurement],
            colorize: Colorize,
            trim_significant_figures: bool,
            highlight_warnings: bool
    ) -> None:
        # [FUNCTION DOC] HELPER: Validates that all results share one label (raises an assertion
        # error otherwise), picks the time unit, computes ordered unique row and column keys and
        # builds the rows and columns.
        if len({r.label for r in results}) != 1:
            raise AssertionError("All results must share the same label")

        self.results = results
        self._colorize = colorize
        self._trim_significant_figures = trim_significant_figures
        self._highlight_warnings = highlight_warnings
        self.label = results[0].label
        self.time_unit, self.time_scale = common.select_unit(
            min(r.median for r in results)
        )

        self.row_keys = common.ordered_unique([self.row_fn(i) for i in results])
        self.row_keys.sort(key=operator.itemgetter(slice(2)))  # preserve stmt order
        self.column_keys = common.ordered_unique([self.col_fn(i) for i in results])
        self.rows, self.columns = self.populate_rows_and_columns()

    @staticmethod
    def row_fn(m: common.Measurement) -> tuple[int, str | None, str]:
        # [FUNCTION DOC] HELPER: Row key of a measurement: thread count, environment, row name.
        return m.num_threads, m.env, m.as_row_name

    @staticmethod
    def col_fn(m: common.Measurement) -> str | None:
        # [FUNCTION DOC] HELPER: Column key of a measurement: its description.
        return m.description

    def populate_rows_and_columns(self) -> tuple[tuple[_Row, ...], tuple[_Column, ...]]:
        # [FUNCTION DOC] HELPER: Places each measurement into a grid by row and column key,
        # creates a row object per row key (starting a new group whenever the thread count
        # changes, and showing the environment only when several exist and it changes) and a
        # column object per column key built from the per-group cells, then attaches the columns
        # to every row. Returns the tuples of rows and columns.
        rows: list[_Row] = []
        columns: list[_Column] = []
        ordered_results: list[list[common.Measurement | None]] = [
            [None for _ in self.column_keys]
            for _ in self.row_keys
        ]
        row_position = {key: i for i, key in enumerate(self.row_keys)}
        col_position = {key: i for i, key in enumerate(self.column_keys)}
        for r in self.results:
            i = row_position[self.row_fn(r)]
            j = col_position[self.col_fn(r)]
            ordered_results[i][j] = r

        unique_envs = {r.env for r in self.results}
        render_env = len(unique_envs) > 1
        env_str_len = max(len(i) for i in unique_envs) if render_env else 0

        row_name_str_len = max(len(r.as_row_name) for r in self.results)

        prior_num_threads = -1
        prior_env = ""
        row_group = -1
        rows_by_group: list[list[list[common.Measurement | None]]] = []
        for (num_threads, env, _), row in zip(self.row_keys, ordered_results, strict=True):
            thread_transition = (num_threads != prior_num_threads)
            if thread_transition:
                prior_num_threads = num_threads
                prior_env = ""
                row_group += 1
                rows_by_group.append([])
            rows.append(
                _Row(
                    results=row,
                    row_group=row_group,
                    render_env=(render_env and env != prior_env),
                    env_str_len=env_str_len,
                    row_name_str_len=row_name_str_len,
                    time_scale=self.time_scale,
                    colorize=self._colorize,
                    num_threads=num_threads if thread_transition else None,
                )
            )
            rows_by_group[-1].append(row)
            prior_env = env

        for i in range(len(self.column_keys)):
            grouped_results = [tuple(row[i] for row in g) for g in rows_by_group]
            column = _Column(
                grouped_results=grouped_results,
                time_scale=self.time_scale,
                time_unit=self.time_unit,
                trim_significant_figures=self._trim_significant_figures,
                highlight_warnings=self._highlight_warnings,)
            columns.append(column)

        rows_tuple, columns_tuple = tuple(rows), tuple(columns)
        for ri in rows_tuple:
            ri.register_columns(columns_tuple)
        return rows_tuple, columns_tuple

    def render(self) -> str:
        # [FUNCTION DOC] HELPER: Produces the table text: a centered label header, the column
        # header, rows separated by thread-count lines, a unit footnote, and a note about variance
        # markers when warnings are highlighted and present.
        string_rows = [[""] + self.column_keys]
        string_rows.extend(r.as_column_strings() for r in self.rows)
        num_cols = max(len(i) for i in string_rows)
        for sr in string_rows:
            sr.extend(["" for _ in range(num_cols - len(sr))])

        col_widths = [max(len(j) for j in i) for i in zip(*string_rows, strict=True)]
        finalized_columns = ["  |  ".join(i.center(w) for i, w in zip(string_rows[0], col_widths, strict=True))]
        overall_width = len(finalized_columns[0])
        for string_row, row in zip(string_rows[1:], self.rows, strict=True):
            finalized_columns.extend(row.row_separator(overall_width))
            finalized_columns.append("  |  ".join(row.finalize_column_strings(string_row, col_widths)))

        newline = "\n"
        has_warnings = self._highlight_warnings and any(ri.has_warnings for ri in self.results)
        return f"""
[{(' ' + (self.label or '') + ' ').center(overall_width - 2, '-')}]
{newline.join(finalized_columns)}

Times are in {common.unit_to_english(self.time_unit)}s ({self.time_unit}).
{'(! XX%) Measurement has high variance, where XX is the IQR / median * 100.' + newline if has_warnings else ""}"""[1:]


class Compare:
    # [CLASS DOC] Public collector that renders measurements as text tables. State: the list of
    # measurements and three display options (trim significant figures, coloring mode, highlight
    # warnings), all off by default. Configure with the option methods, then print or convert to
    # a string; one table is produced per label after merging replicates.
    """Helper class for displaying the results of many measurements in a
    formatted table.

    The table format is based on the information fields provided in
    :class:`torch.utils.benchmark.Timer` (`description`, `label`, `sub_label`,
    `num_threads`, etc).

    The table can be directly printed using :meth:`print` or casted as a `str`.

    For a full tutorial on how to use this class, see:
    https://pytorch.org/tutorials/recipes/recipes/benchmark.html

    Args:
        results: List of Measurement to display.
    """
    def __init__(self, results: list[common.Measurement]) -> None:
        # [FUNCTION DOC] Stores the (validated) measurements and sets all display options off.
        # Raises a value error if any item is not a Measurement.
        self._results: list[common.Measurement] = []
        self.extend_results(results)
        self._trim_significant_figures = False
        self._colorize = Colorize.NONE
        self._highlight_warnings = False

    def __str__(self) -> str:
        # [FUNCTION DOC] Renders all tables and joins them with newlines.
        return "\n".join(self._render())

    def extend_results(self, results) -> None:
        # [FUNCTION DOC] Validates that every item is a Measurement (raises a value error
        # otherwise) and appends them all to the stored list.
        """Append results to already stored ones.

        All added results must be instances of ``Measurement``.
        """
        for r in results:
            if not isinstance(r, common.Measurement):
                raise ValueError(
                    "Expected an instance of `Measurement`, " f"got {type(r)} instead."
                )
        self._results.extend(results)

    def trim_significant_figures(self) -> None:
        # [FUNCTION DOC] Turns on significant figure trimming for later rendering.
        """Enables trimming of significant figures when building the formatted table."""
        self._trim_significant_figures = True

    def colorize(self, rowwise=False) -> None:
        # [FUNCTION DOC] Turns on coloring: per row when rowwise is true, else per column.
        """Colorize formatted table.

        Colorize columnwise by default.
        """
        self._colorize = Colorize.ROWWISE if rowwise else Colorize.COLUMNWISE

    def highlight_warnings(self) -> None:
        # [FUNCTION DOC] Turns on variance warning markers in cells for later rendering.
        """Enables warning highlighting when building formatted table."""
        self._highlight_warnings = True

    def print(self) -> None:
        # [FUNCTION DOC] Writes the rendered tables to standard output.
        """Print formatted table"""
        print(str(self))

    def _render(self):
        # [FUNCTION DOC] HELPER: Merges replicate measurements, groups them by label and returns
        # a list with one rendered table string per label.
        results = common.Measurement.merge(self._results)
        grouped_results = self._group_by_label(results)
        output = [self._layout(group) for group in grouped_results.values()]
        return output

    def _group_by_label(self, results: list[common.Measurement]):
        # [FUNCTION DOC] HELPER: Returns a mapping from label to the list of measurements with
        # that label, in order of first appearance.
        grouped_results: collections.defaultdict[str, list[common.Measurement]] = collections.defaultdict(list)
        for r in results:
            grouped_results[r.label].append(r)
        return grouped_results

    def _layout(self, results: list[common.Measurement]):
        # [FUNCTION DOC] HELPER: Builds a table for one label's measurements with the current
        # display options and returns its rendered text.
        table = Table(
            results,
            self._colorize,
            self._trim_significant_figures,
            self._highlight_warnings
        )
        return table.render()
