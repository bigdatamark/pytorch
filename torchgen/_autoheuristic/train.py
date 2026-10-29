# mypy: ignore-errors
# [MODULE DOC]
# Base class (AHTrain) for the AutoHeuristic training scripts: command line parsing, reading
# the collected data log (metadata line plus CSV), feature engineering hooks, categorical
# one-hot handling and helpers to write the generated learned-heuristic source file. Import-time
# side effect: installs two global warning filters that silence pandas deprecation warnings.

import argparse
import json
import warnings

import pandas as pd  # type: ignore[import-untyped]

from torch._inductor.autoheuristic.autoheuristic_utils import (
    CHOICE_COL,
    get_metadata_str_from_log,
)


# TODO (AlnisM): Fix these warnings
warnings.filterwarnings(
    "ignore",
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated",
)
warnings.filterwarnings(
    "ignore",
    message="DataFrameGroupBy.apply operated on the grouping columns.",
)


class AHTrain:
    # [CLASS DOC] Abstract-ish base for training a learned heuristic. State: parser (argparse
    # parser with base options dataset, --nrows, --heuristic-name, --data, --save-dot,
    # --ranking) and args (None until generate_heuristic parses them). Subclasses provide main
    # and the codegen hooks; several methods here are overridable no-op defaults.
    """
    Base class for AutoHeuristic training.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Creates the argument parser, registers the base arguments and sets
        # args to None.
        self.parser = argparse.ArgumentParser()
        self.add_base_arguments()
        self.args = None

    def add_base_arguments(self):
        # [FUNCTION DOC] Registers: positional dataset path; --nrows (limit rows read);
        # --heuristic-name (default learned_heuristic); --data (repeatable pair of type name and
        # path of extra datasets to evaluate); --save-dot (flag); --ranking (int, number of
        # ranked choices to provide). Mutates self.parser only.
        self.parser.add_argument(
            "dataset",
            type=str,
            help="Path to text file containing data collected with AutoHeuristic.",
        )
        self.parser.add_argument(
            "--nrows",
            type=int,
            default=None,
            help="Only read first n rows of the dataset.",
        )
        self.parser.add_argument(
            "--heuristic-name",
            type=str,
            default="learned_heuristic",
            help="Name of the heuristic to be generated.",
        )
        self.parser.add_argument(
            "--data",
            nargs=2,
            action="append",
            metavar=("TYPE", "PATH"),
            help="Specify name of datasets and file paths to be evaluated.",
        )
        self.parser.add_argument(
            "--save-dot",
            action="store_true",
            help="Export heuristic to graphviz dot.",
        )
        self.parser.add_argument(
            "--ranking",
            type=int,
            default=None,
            help="""
                Makes AutoHeuristic learn a heuristic that ranks choices instead of predicting a single choice.
                The argument is the number of choices the heuristic will provide.
            """,
        )

    def parse_args(self):
        # [FUNCTION DOC] Parses the process command line with the configured parser and returns
        # the namespace (exits the process on invalid arguments).
        return self.parser.parse_args()

    def parse_log(self, log_path, nrows=None):
        # [FUNCTION DOC] Loads the data log into a table plus metadata. Returns a tuple of
        # (table, metadata, feature names = numerical then categorical, categorical feature
        # names, list of distinct choices). Choices are computed before the optional row limit
        # nrows and before filter_df is applied.
        (df, metadata) = self.deserialize_data(log_path)
        numerical_features = metadata["numerical_features"]
        categorical_features = metadata["categorical_features"]
        choices = df[CHOICE_COL].unique().tolist()
        features = numerical_features + categorical_features
        if nrows is not None:
            df = df.head(nrows)
        df = self.filter_df(df)
        return (df, metadata, features, categorical_features, choices)

    def generate_heuristic(self):
        # [FUNCTION DOC] Entry point: parses command line arguments into self.args and calls
        # main with dataset path, extra datasets, row limit, heuristic name, save-dot flag and a
        # boolean for whether ranking was requested. Requires subclasses to define main.
        self.args = self.parse_args()
        self.main(
            self.args.dataset,
            self.args.data,
            self.args.nrows,
            self.args.heuristic_name,
            self.args.save_dot,
            self.args.ranking is not None,
        )

    def filter_df(self, df):
        # [FUNCTION DOC] Hook for subclasses to filter the table; default returns it unchanged.
        return df

    def add_new_features(self, results):
        # [FUNCTION DOC] Hook for subclasses to add derived feature columns; default adds none
        # and returns the table with an empty list of new categorical feature names.
        return (results, [])

    def add_real_datasets(self, datasets, other_datasets, cat_feature2cats):
        # [FUNCTION DOC] For each (name, path) pair in other_datasets (may be empty or none),
        # loads the data via self.get_df without filters, reusing the known category lists, and
        # stores the table in the datasets mapping under name. Mutates datasets.
        # KNOWN ISSUE: get_df is not defined on this base class (only in subclasses), and the
        # subclass override takes an extra ranking argument.
        if other_datasets:
            for name, path in other_datasets:
                (df_other, choices, _, _, _) = self.get_df(
                    path, cat_feature2cats=cat_feature2cats, apply_filters=False
                )
                datasets[name] = df_other

    def handle_categorical_features(
        self, cat_feature2cats, categorical_features, results
    ):
        # [FUNCTION DOC] Converts each categorical column to a pandas categorical using a fixed
        # category list (taken from cat_feature2cats when known, otherwise from the column's
        # distinct values and then recorded in the mapping), then one-hot encodes them. Returns
        # (encoded table, mapping of feature to categories, mapping of each dummy column name to
        # its (original column, value) pair). Mutates the input mapping and table columns.
        # Doing this here because if we create another df for testing purposes
        # and that other df does not contain all categories for a categorical feature,
        # pd.dummies will not create columns for the missing categories
        if not cat_feature2cats:
            cat_feature2cats = {}
        for cat_feature in categorical_features:
            if cat_feature in cat_feature2cats:
                categories = cat_feature2cats[cat_feature]
            else:
                categories = results[cat_feature].unique()
                cat_feature2cats[cat_feature] = categories
            results[cat_feature] = pd.Categorical(
                results[cat_feature], categories=categories
            )

        dummy_col_2_col_val = {}
        for col in categorical_features:
            unique_vals = results[col].unique()
            for val in unique_vals:
                dummy_col_2_col_val[f"{col}_{val}"] = (col, val)
        # one-hot encode categorical features
        results = pd.get_dummies(results, columns=categorical_features)
        return (results, cat_feature2cats, dummy_col_2_col_val)

    def gen_precondition(self, opt_name, shared_memory, device_capa):
        # [FUNCTION DOC] Returns the source text of the generated check_precondition method,
        # which requires the metadata name to match and the GPU shared memory and compute
        # capability to equal the given values. opt_name is unused.
        return f"""    def check_precondition(self, metadata: AHMetadata, context: AHContext,) -> bool:
        return (
            metadata.name == self.get_name()
            and metadata.shared_memory == {shared_memory}
            and str(metadata.device_capa) == "{device_capa}"
        )"""

    def codegen_boilerplate(
        self, heuristic_name, opt_name, threshold, shared_memory, device_capa, dt
    ):
        # [FUNCTION DOC] Hook for subclasses: should return the boilerplate source text of the
        # generated heuristic class. Default does nothing and returns None.
        pass

    def gen_predict_fn_def(self):
        # [FUNCTION DOC] Hook for subclasses: should return the definition line of the generated
        # prediction method. Default does nothing and returns None.
        pass

    def write_heuristic_to_file(self, lines, heuristic_name):
        # [FUNCTION DOC] Writes the lines joined by newlines (plus a trailing newline) to the
        # artifacts directory of torch inductor autoheuristic as a file named after the
        # heuristic with a leading underscore. The path is relative to the current working
        # directory (expects to run from the script directory); overwrites; raises OSError if
        # the directory does not exist.
        output_file = (
            f"../../../torch/_inductor/autoheuristic/artifacts/_{heuristic_name}.py"
        )
        path = f"{output_file}"
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

    def deserialize_data(self, log_path):
        # [FUNCTION DOC] Reads the metadata JSON from the log's first line and the rest of the
        # file as CSV (skipping malformed lines). Returns (table, metadata mapping).
        json_string = get_metadata_str_from_log(log_path)
        metadata = self.deserialize_metadata(json_string)

        df = pd.read_csv(log_path, skiprows=1, on_bad_lines="skip")
        return (df, metadata)

    def deserialize_metadata(self, json_string):
        # [FUNCTION DOC] Parses a JSON string into a mapping; raises on invalid JSON.
        return json.loads(json_string)


if __name__ == "__main__":
    train = AHTrain()
    train.generate_heuristic()
