# mypy: allow-untyped-defs
# [MODULE DOC] Pluggable backend abstraction for DataFrame operations used by datapipes.
# Provides PandasWrapper (default backend, pandas imported lazily and cached in module
# globals) and module-level functions that delegate to the currently selected wrapper
# class; set_df_wrapper swaps it. No import-time side effects (pandas is not imported yet).
from typing import Any


_pandas: Any = None
_WITH_PANDAS: bool | None = None


def _try_import_pandas() -> bool:
    # [FUNCTION DOC] HELPER: Attempts to import pandas; on success stores the module in
    # the global _pandas and returns True, on ImportError returns False. Mutates globals.
    try:
        import pandas  # type: ignore[import]

        global _pandas
        _pandas = pandas
        return True
    except ImportError:
        return False


# pandas used only for prototyping, will be shortly replaced with TorchArrow
def _with_pandas() -> bool:
    # [FUNCTION DOC] HELPER: Returns whether pandas is available. The import attempt is
    # made once and its result cached in the global _WITH_PANDAS.
    global _WITH_PANDAS
    if _WITH_PANDAS is None:
        _WITH_PANDAS = _try_import_pandas()
    return _WITH_PANDAS


class PandasWrapper:
    # [CLASS DOC] Stateless backend implementing the DataFrame operations (create, type
    # checks, iterate, concat, slice, length, columns) with pandas. All methods are class
    # methods; most raise RuntimeError when pandas is not installed.
    @classmethod
    def create_dataframe(cls, data, columns):
        # [FUNCTION DOC] Builds a pandas DataFrame from data with the given column names.
        # Raises RuntimeError if pandas is unavailable.
        if not _with_pandas():
            raise RuntimeError("DataFrames prototype requires pandas to function")
        return _pandas.DataFrame(data, columns=columns)  # type: ignore[union-attr]

    @classmethod
    def is_dataframe(cls, data):
        # [FUNCTION DOC] Returns True if data is a pandas DataFrame; False when pandas is
        # unavailable (never raises).
        if not _with_pandas():
            return False
        return isinstance(data, _pandas.core.frame.DataFrame)  # type: ignore[union-attr]

    @classmethod
    def is_column(cls, data):
        # [FUNCTION DOC] Returns True if data is a pandas Series (a column); False when
        # pandas is unavailable (never raises).
        if not _with_pandas():
            return False
        return isinstance(data, _pandas.core.series.Series)  # type: ignore[union-attr]

    @classmethod
    def iterate(cls, data):
        # [FUNCTION DOC] Generator yielding each row of the DataFrame as a named tuple
        # without the index. Raises RuntimeError if pandas is unavailable.
        if not _with_pandas():
            raise RuntimeError("DataFrames prototype requires pandas to function")
        yield from data.itertuples(index=False)

    @classmethod
    def concat(cls, buffer):
        # [FUNCTION DOC] Concatenates a list of DataFrames row-wise into one DataFrame.
        # Raises RuntimeError if pandas is unavailable.
        if not _with_pandas():
            raise RuntimeError("DataFrames prototype requires pandas to function")
        return _pandas.concat(buffer)  # type: ignore[union-attr]

    @classmethod
    def get_item(cls, data, idx):
        # [FUNCTION DOC] Returns the single-row slice at position idx as a DataFrame (not a
        # scalar row). Raises RuntimeError if pandas is unavailable.
        if not _with_pandas():
            raise RuntimeError("DataFrames prototype requires pandas to function")
        return data[idx : idx + 1]

    @classmethod
    def get_len(cls, df):
        # [FUNCTION DOC] Returns the number of rows of the DataFrame. Raises RuntimeError
        # if pandas is unavailable.
        if not _with_pandas():
            raise RuntimeError("DataFrames prototype requires pandas to function")
        return len(df.index)

    @classmethod
    def get_columns(cls, df):
        # [FUNCTION DOC] Returns the DataFrame column names as a plain list. Raises
        # RuntimeError if pandas is unavailable.
        if not _with_pandas():
            raise RuntimeError("DataFrames prototype requires pandas to function")
        return list(df.columns.values.tolist())


# When you build own implementation just override it with dataframe_wrapper.set_df_wrapper(new_wrapper_class)
default_wrapper = PandasWrapper


def get_df_wrapper():
    # [FUNCTION DOC] Returns the currently active wrapper class (module global
    # default_wrapper, initially PandasWrapper).
    return default_wrapper


def set_df_wrapper(wrapper) -> None:
    # [FUNCTION DOC] Replaces the active wrapper class used by all module-level functions.
    # Side effect: mutates the global default_wrapper. No validation of the wrapper.
    global default_wrapper
    default_wrapper = wrapper


def create_dataframe(data, columns=None):
    # [FUNCTION DOC] Delegates to the active wrapper to build a DataFrame from data and
    # optional column names.
    wrapper = get_df_wrapper()
    return wrapper.create_dataframe(data, columns)


def is_dataframe(data):
    # [FUNCTION DOC] Delegates to the active wrapper; True if data is a DataFrame.
    wrapper = get_df_wrapper()
    return wrapper.is_dataframe(data)


def get_columns(data):
    # [FUNCTION DOC] Delegates to the active wrapper; returns the list of column names.
    wrapper = get_df_wrapper()
    return wrapper.get_columns(data)


def is_column(data):
    # [FUNCTION DOC] Delegates to the active wrapper; True if data is a single column.
    wrapper = get_df_wrapper()
    return wrapper.is_column(data)


def concat(buffer):
    # [FUNCTION DOC] Delegates to the active wrapper; concatenates a list of DataFrames.
    wrapper = get_df_wrapper()
    return wrapper.concat(buffer)


def iterate(data):
    # [FUNCTION DOC] Delegates to the active wrapper; returns an iterator over the rows.
    wrapper = get_df_wrapper()
    return wrapper.iterate(data)


def get_item(data, idx):
    # [FUNCTION DOC] Delegates to the active wrapper; returns the one-row slice at idx.
    wrapper = get_df_wrapper()
    return wrapper.get_item(data, idx)


def get_len(df):
    # [FUNCTION DOC] Delegates to the active wrapper; returns the number of rows.
    wrapper = get_df_wrapper()
    return wrapper.get_len(df)
