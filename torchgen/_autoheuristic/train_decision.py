# mypy: ignore-errors
# [MODULE DOC]
# Trains a scikit-learn decision tree classifier on AutoHeuristic data and generates the
# source of a learned heuristic from it. Contains the trainer class AHTrainDecisionTree, small
# dataclasses holding evaluation metrics, and DecisionEvaluator which scores a tree (accuracy,
# unsure rate, speedups, top-k ranking quality, comparison with a default choice). Import-time
# side effects: a global pandas warning filter and a module logger with a debug-level stream
# handler attached (DEBUG constant is true).

import itertools
import json
import logging
import math
import warnings


warnings.filterwarnings(
    "ignore",
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated",
)

from dataclasses import dataclass

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from ah_tree import DecisionTree
from scipy.stats import gmean
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from train import AHTrain


log = logging.getLogger(__name__)
DEBUG = True
if DEBUG:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(formatter)
    log.addHandler(ch)


class AHTrainDecisionTree(AHTrain):
    # [CLASS DOC] Trainer that fits decision trees (grid search over depth, leaf size and
    # criterion), picks the best model whose wrong prediction rate on validation stays within
    # the allowed limit, marks unsafe leaves, and generates the heuristic source. Subclasses
    # override hooks such as add_new_features, get_default_config, is_unsafe_leaf and the
    # grid. Sets dummy_col_2_col_val dynamically in main.
    def __init__(self):
        # [FUNCTION DOC] Calls the base class initializer.
        super().__init__()

    def debug_time(self, row, top_k_choices):
        # [FUNCTION DOC] HELPER: Debug aid. Prints each choice with its measured time for the
        # row, sorted fastest first, appending a marker for choices in top_k_choices.
        choices_feedback = json.loads(row["choice2time"])
        timings = sorted(choices_feedback.items(), key=lambda x: x[1])
        for choice, time in timings:
            result = f"{choice} {time}"
            if choice in top_k_choices:
                result += " TOPK"
            print(result)

    def is_unsafe_leaf(self, row, predicted_config, choice2time):
        # [FUNCTION DOC] Hook: decides whether a leaf containing the given sample is unsafe,
        # given the leaf's predicted choice and a mapping of choice to measured time. Default
        # never marks a leaf unsafe (returns false).
        """
        Can be overridden by subclasses to define their own logic for deciding when a leaf is unsafe. Returns a sample
        that landed in the leaf, the choice predicted by the tree, and a dictionary that maps each choice to the
        execution time. One can for example decide to mark a leaf as unsafe if the predicted choice is 2x slower
        than the fastest choice.
        If a leaf is unsafe, the learned heuristic will always return 'unsure' if an input lands in that leaf.
        """

        return False

    def get_unsafe_leaves(self, model, df, feature_columns):
        # [FUNCTION DOC] For every leaf reached by the data, takes the leaf's predicted choice
        # (from its first sample) and asks is_unsafe_leaf for each sample in it; a leaf is unsafe
        # as soon as one sample says so. Returns the list of unsafe leaf ids.
        """
        Given a trained decision tree, and a dataframe containing the training data, returns a list of unsafe leaves.
        """
        X = df[feature_columns]
        leaf_ids = model.apply(X)
        unique_leaves = np.unique(leaf_ids)

        unsafe_leaves = []
        # Iterate over each leaf
        for leaf in unique_leaves:
            leaf_mask = leaf_ids == leaf
            # Get samples that land in this leaf
            leaf_X = X[leaf_mask]

            predicted_config = model.predict(leaf_X.iloc[[0]])[0]

            # For each sample, check if we should mark the leaf as unsafe
            for idx, row in leaf_X.iterrows():
                choice2time = json.loads(df.loc[idx, "choice2time"])
                if self.is_unsafe_leaf(row, predicted_config, choice2time):
                    unsafe_leaves.append(leaf)
                    break
        return unsafe_leaves

    def get_allowed_wrong_prediction_pct(self):
        # [FUNCTION DOC] Hook: returns the largest tolerated fraction of wrong predictions on the
        # validation set; default 0.01.
        """
        This is used to determine a threshold for when a learned heuristic returns 'unsure'.
        If this function returns 0.01, we will set the probability required for the decision tree to return a decision
        such that at most 1% of the predictions will be wrong on the validation set.
        """
        return 0.01

    def get_grid_search_values(self):
        # [FUNCTION DOC] Hook: returns the default grid as a mapping with lists for max_depth
        # (5, 6, 7), min_samples_leaf and criterion (gini, entropy).
        """
        Standard values for grid search. Can be overridden.
        """
        return {
            "max_depth": [5, 6, 7],
            "min_samples_leaf": [1, 5, 10, 0.01, 0.05, 0.02],
            "criterion": ["gini", "entropy"],
        }

    def predict(self, model, df, feature_columns):
        # [FUNCTION DOC] Runs the model on the feature columns of df and returns a tuple of
        # predicted classes, class probabilities and leaf ids.
        """
        Returns the predictions, probabilities, and leaf ids for a given dataframe.
        """
        predictions = model.predict(df[feature_columns])
        proba = model.predict_proba(df[feature_columns])
        leaf_ids = model.apply(df[feature_columns])
        return predictions, proba, leaf_ids

    def ranking_num_choices(self):
        # [FUNCTION DOC] Returns how many ranked choices the heuristic provides: the --ranking
        # argument, or 5 if it was not given. Requires args to be set (parsed).
        # if the heuristic is used for ranking, this function returns the number
        # of choices that the heuristic will return
        if self.args.ranking is None:
            return 5
        return self.args.ranking

    def train_and_evaluate_models(
        self,
        datasets,
        max_depths,
        min_samples_leafs,
        criterion_list,
        feature_columns,
        ranking=False,
    ):
        # [FUNCTION DOC] Grid search over every combination of depth, leaf size and criterion.
        # Each model is fit on the train set (weighted by relative performance when ranking),
        # wrapped, pruned to the ranking size if ranking, scored on validation to find unsafe
        # leaves and a safe probability threshold, then evaluated on every dataset. A model
        # becomes best if validation wrong count is within the allowed fraction and it has more
        # correct predictions than the best so far. Returns (table of metrics per dataset and
        # setting, best model or None, its safe probability, its unsafe leaves). Prints progress.
        """
        Does a grid search over max_depths, min_samples_leafs, and criterion_list and returns the best model.
        """

        results = []
        best_model = None
        best_model_safe_proba = 0
        best_model_num_correct = 0
        best_model_unsafe_leaves = []
        columns = ["set", "crit", "max_depth", "min_samples_leaf"]
        metrics_columns = []
        for max_depth, min_samples_leaf, criterion in itertools.product(
            max_depths, min_samples_leafs, criterion_list
        ):
            print(
                f"max_depth={max_depth} min_samples_leaf={min_samples_leaf} criterion={criterion}"
            )
            model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                criterion=criterion,
                random_state=42,
            )
            df_train = datasets["train"]
            df_val = datasets["val"]
            if ranking:
                model.fit(
                    df_train[feature_columns],
                    df_train["winner"],
                    sample_weight=df_train["relative_performance"],
                )
            else:
                model.fit(df_train[feature_columns], df_train["winner"])

            model = DecisionTree(model, feature_columns)

            if ranking:
                model.prune(df_train, "winner", k=self.ranking_num_choices())

            unsafe_leaves = self.get_unsafe_leaves(model, df_train, feature_columns)
            predictions, proba, leaf_ids = self.predict(model, df_val, feature_columns)

            wrong_pct = self.get_allowed_wrong_prediction_pct()
            evaluator = DecisionEvaluator(
                self,
                model,
                predictions,
                df_val,
                proba,
                wrong_pct=wrong_pct,
                unsafe_leaves=unsafe_leaves,
                leaf_ids=leaf_ids,
                k=self.ranking_num_choices(),
                ranking=ranking,
            )
            safe_proba = evaluator.get_safe_proba()
            print(f"safe_proba={safe_proba}")

            def eval(name, df):
                # [FUNCTION DOC] HELPER: Evaluates the current model on df using the safe
                # probability threshold. When ranking, first drops the duplicated near-best rows
                # (keeping rows whose winner equals the actual winner). Returns EvalResults.
                # The name argument is unused.
                if ranking:
                    # when ranking is enabled, we duplicate each input for each choice that
                    # is almost as good as the best choice
                    # we do not want to evaluate the same input multiple times, so we remove duplicates here
                    df = df[df["winner"] == df["actual_winner"]]
                predictions, proba, leaf_ids = self.predict(model, df, feature_columns)
                evaluator = DecisionEvaluator(
                    self,
                    model,
                    predictions,
                    df,
                    proba,
                    wrong_pct=wrong_pct,
                    threshold=safe_proba,
                    unsafe_leaves=unsafe_leaves,
                    leaf_ids=leaf_ids,
                    k=self.ranking_num_choices(),
                    ranking=ranking,
                )
                return evaluator.get_results()

            for dataset_name, dataset in datasets.items():
                eval_result: EvalResults = eval(dataset_name, dataset)
                eval_result_metrics = eval_result.to_map()
                if dataset_name == "val":
                    num_correct = eval_result.accuracy.num_correct
                    num_wrong = eval_result.accuracy.num_wrong
                    num_total = eval_result.accuracy.total
                    if num_wrong <= num_total * wrong_pct:
                        if num_correct > best_model_num_correct:
                            print(
                                f"new best model with {num_correct} correct and {num_wrong} wrong"
                            )
                            best_model = model
                            best_model_num_correct = num_correct
                            best_model_safe_proba = safe_proba
                            best_model_unsafe_leaves = unsafe_leaves

                result = (dataset_name, criterion, max_depth, min_samples_leaf)
                result += tuple(eval_result_metrics.values())
                results.append(result)
                if len(metrics_columns) == 0:
                    metrics_columns = list(eval_result_metrics.keys())
                    columns += metrics_columns

        return (
            pd.DataFrame(results, columns=columns),
            best_model,
            best_model_safe_proba,
            best_model_unsafe_leaves,
        )

    def get_test_and_val_size(self):
        # [FUNCTION DOC] Hook: returns the (test, validation) fractions; default (0.15, 0.15).
        """
        Returns the size of the test and validation sets.
        """
        return (0.15, 0.15)

    def prepare_datasets(self, df, other_datasets, cat_feature2cats, ranking=False):
        # [FUNCTION DOC] Splits df with a fixed seed into train, validation and test tables
        # (test first, then validation from the rest), stores them in a mapping under train,
        # val and test, then adds the user supplied datasets by name. Returns the mapping.
        """
        Splits the dataframe into train, val, and test sets.
        Also adds other datasets, specified by the user, to the train set.
        """
        test_size, val_size = self.get_test_and_val_size()
        # Split into train+val and test
        df_train_val, df_test = train_test_split(
            df, test_size=test_size, random_state=42
        )

        # Split train+val inputs into train and val
        train_val_size = 1 - test_size
        df_train, df_val = train_test_split(
            df_train_val, test_size=val_size / train_val_size, random_state=42
        )
        datasets = {"train": df_train, "val": df_val, "test": df_test}
        self.add_real_datasets(datasets, other_datasets, cat_feature2cats, ranking)
        return datasets

    def export_to_dot(self, best_model, df, feature_columns):
        # [FUNCTION DOC] Writes the model's graphviz dot text to the file best_model.dot in the
        # current directory (overwrites). df and feature_columns are unused.
        """
        Export a learned decision tree to a dot file.
        """
        dot_str = best_model.to_dot()
        with open("best_model.dot", "w") as f:
            f.write(dot_str)

    def get_feature_columns(self, df):
        # [FUNCTION DOC] Returns the column names of df that are model features, i.e. all
        # columns except the fixed list of bookkeeping columns (speedup, winner, target,
        # avail_choices, choice2time, index, actual_winner, relative_performance).
        """
        The dataframe contains columns that are not features, such as 'winner', 'speedup' that are only used for
        debugging purposes. This function returns the columns that are actually features.
        """
        exclude_columns = [
            "speedup",
            "winner",
            "target",
            "avail_choices",
            "choice2time",
            "index",
            "actual_winner",
            "relative_performance",
        ]
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        return feature_columns

    def add_training_data(self, df_train, datasets):
        # [FUNCTION DOC] Hook to extend the training table; default returns the "train" table
        # from datasets (df_train is ignored).
        return datasets["train"]

    def main(
        self,
        log_path,
        other_datasets,
        nrows,
        heuristic_name,
        save_dot=False,
        ranking=False,
    ):
        # [FUNCTION DOC] Full pipeline. Loads and featurizes the data, stores the dummy column
        # mapping on self, builds datasets, extends the training set, runs the grid search,
        # prints result tables (restricted to ranking columns when ranking), optionally exports
        # a dot file and, if a valid best model exists, generates and writes the heuristic;
        # otherwise prints that no heuristic was generated. Side effects: stdout, files.
        """
        Main function that trains a decision tree and generates a heuristic.
        """
        # TODO: Enable apply_filters
        (df, choices, cat_feature2cats, dummy_col_2_col_val, metadata) = self.get_df(
            log_path, nrows=nrows, apply_filters=False, add_near_best=ranking
        )
        self.dummy_col_2_col_val = dummy_col_2_col_val
        datasets = self.prepare_datasets(df, other_datasets, cat_feature2cats, ranking)
        df_train = self.add_training_data(datasets["train"], datasets)
        datasets["train"] = df_train
        print(datasets["train"]["winner"].value_counts().to_string())

        feature_columns = self.get_feature_columns(df)
        grid_search_values = self.get_grid_search_values()
        max_depths = grid_search_values["max_depth"]
        min_samples_leafs = grid_search_values["min_samples_leaf"]
        criterion_list = grid_search_values["criterion"]
        (
            results_df,
            best_model,
            best_model_safe_proba,
            unsafe_leaves,
        ) = self.train_and_evaluate_models(
            datasets,
            max_depths,
            min_samples_leafs,
            criterion_list,
            feature_columns,
            ranking=ranking,
        )

        if ranking:
            columns_to_keep = [
                "set",
                "crit",
                "max_depth",
                "min_samples_leaf",
                "total",
                "top_k_correct",
                "top_k_wrong",
                "top_k_unsure",
                "wrong_max_speedup_k",
                "wrong_gmean_speedup_k",
            ]
            results_df = results_df[columns_to_keep]
        # prints results for all models and datasets
        print(results_df.to_string())

        sort_metric = "top_k_correct" if ranking else "correct"
        # prints results grouped by dataset
        for set_name in results_df["set"].unique():
            dataset_results = results_df[results_df["set"] == set_name]
            dataset_results = dataset_results.sort_values(by=sort_metric)
            print(dataset_results.to_string() + "\n")

        if best_model is not None:
            if save_dot:
                self.export_to_dot(best_model, df, feature_columns)
            self.codegen(
                best_model,
                metadata,
                heuristic_name,
                best_model_safe_proba,
                dummy_col_2_col_val,
                unsafe_leaves,
            )
        else:
            print(
                "All learned models have too many wrong predictions, so no heuristic was generated"
            )

    def get_df(
        self,
        log_path,
        cat_feature2cats=None,
        nrows=None,
        apply_filters=False,
        add_near_best=False,
    ):
        # [FUNCTION DOC] Parses the log, then groups measurements by input features and choice
        # to compute count, relative standard deviation and median time (logging high variance
        # at debug level). Optionally keeps only inputs with at least 3 measurements and low
        # variance (apply_filters). For each input with at least 2 choices computes winner,
        # speedup over second best, available choices and a JSON map of choice to time. When
        # add_near_best, duplicates each row per choice within 2 percent of the best, with a
        # relative performance weight. Then adds subclass features and one-hot encodes
        # categoricals. Returns (table, choices, category mapping, dummy column mapping, metadata).
        """
        Parses the log file and processes the data into a dataframe that can be used for training.
        """
        (df, metadata, features, categorical_features, choices) = self.parse_log(
            log_path, nrows
        )

        def calculate_stats(group):
            # [FUNCTION DOC] HELPER: For a group of measurements returns a series with the
            # count, relative standard deviation (percent of the mean; infinite if any
            # measurement is infinite or the mean is zero) and median time.
            count = len(group)
            has_inf = np.isinf(group["feedback"]).any()
            if has_inf:
                relative_std = np.inf
                median = np.inf
            else:
                mean = group["feedback"].mean()
                std = group["feedback"].std()
                relative_std = (std / mean) * 100 if mean != 0 else np.inf
                median = group["feedback"].median()
            if relative_std > 5:
                times = group["feedback"].tolist()
                times_str = ", ".join([f"{t:.3f}" for t in sorted(times)])
                log.debug("High relative std: %f. times=%s", relative_std, times_str)
            return pd.Series(
                {
                    "count": count,
                    "relative_std": relative_std,
                    "median_execution_time": median,
                }
            )

        feature_columns = features
        stats = (
            df.groupby(feature_columns + ["choice"], as_index=False)
            .apply(calculate_stats, include_groups=False)
            .reset_index()
        )

        # TODO: We have to be careful with removing certain choices, because if we e.g. remove the winner, the
        # heuristic will end up learning wrong things. But, execution times with high variance are also bad
        if apply_filters:
            # Filter out inputs with less than 3 measurements or high relative std
            valid_stats = stats[(stats["count"] >= 3) & (stats["relative_std"] <= 5)]
            # Group by input features and count how many valid choices we have for each input
            valid_inputs = valid_stats.groupby(feature_columns).filter(
                lambda x: len(x) >= 2
            )
        else:
            valid_inputs = stats

        # Compute the winner and speedup for each valid input
        def get_winner_and_speedup(group):
            # [FUNCTION DOC] HELPER: For the rows of one input (one per choice) returns a series
            # with the fastest choice (winner), speedup of winner over second best, the unique
            # choices and a JSON string mapping choice to median time. Raises AssertionError if
            # fewer than 2 rows or duplicate choices exist.
            if len(group) < 2:
                raise AssertionError("Need at least 2 choices")

            sorted_group = group.sort_values("median_execution_time")
            winner = sorted_group.iloc[0]["choice"]
            winning_time = sorted_group.iloc[0]["median_execution_time"]
            second_best_time = sorted_group.iloc[1]["median_execution_time"]
            speedup = second_best_time / winning_time
            unique_choices = group["choice"].unique()

            choice2time = {}
            for row in group.itertuples():
                choice2time[row.choice] = row.median_execution_time

            if len(unique_choices) != len(group):
                raise AssertionError(
                    f"len(unique_choices) != len(group): "
                    f"{len(unique_choices)} != {len(group)}"
                )

            return pd.Series(
                {
                    "winner": winner,
                    "speedup": speedup,
                    "avail_choices": unique_choices,
                    "choice2time": json.dumps(choice2time),
                }
            )

        results = (
            valid_inputs.groupby(feature_columns, as_index=False)
            .filter(lambda x: len(x) >= 2)
            .groupby(feature_columns, as_index=False)
            .apply(get_winner_and_speedup, include_groups=False)
            .reset_index()
        )

        def add_near_best_configs(df):
            # [FUNCTION DOC] HELPER: Expands each row into one row per choice whose time is
            # within 2 percent of the best (relative performance at least 0.98), setting the
            # winner column to that choice, remembering the true winner in actual_winner and
            # storing relative_performance. Returns the new table with a fresh index.
            # KNOWN ISSUE: the checks against none and infinity use identity comparison, which
            # does not reliably detect a numpy infinity, and a debugger breakpoint is called.
            new_rows = []

            for index, row in df.iterrows():
                dictionary = json.loads(row["choice2time"])
                min_value = min(dictionary.values())

                for key, value in dictionary.items():
                    new_row = row.copy()
                    relative_performance = min_value / value
                    new_row["relative_performance"] = relative_performance
                    if relative_performance is None or relative_performance is np.inf:
                        breakpoint()
                    new_row["actual_winner"] = row["winner"]
                    new_row["winner"] = key
                    if relative_performance >= 0.98:
                        new_rows.append(new_row)

            return pd.DataFrame(new_rows).reset_index(drop=True)

        if add_near_best:
            results = add_near_best_configs(results)
        (results, added_categorical_features) = self.add_new_features(results)
        categorical_features += added_categorical_features

        (
            results,
            cat_feature2cats,
            dummy_col_2_col_val,
        ) = self.handle_categorical_features(
            cat_feature2cats, categorical_features, results
        )
        return (results, choices, cat_feature2cats, dummy_col_2_col_val, metadata)

    def ranking_always_included_choices(self):
        # [FUNCTION DOC] Hook: returns choices always appended to the top-k list when ranking;
        # default is an empty list.
        return []

    def gen_classes(self, classes, num_spaces):
        # [FUNCTION DOC] Returns source text: one append statement per class name, each indented
        # by num_spaces spaces and separated by newlines, for the generated choices filler.
        """
        If classes=['choice1', 'choice2', 'choice3'], then this function returns
        the following string:
        self.choices.append('choice1')
        self.choices.append('choice2')
        self.choices.append('choice3')
        Used in the generated heuristic to map the index of a choice to its name.
        """
        indent = " " * num_spaces
        return "\n".join([f"{indent}self.choices.append('{c}')" for c in classes])

    def get_default_config(self, row):
        # [FUNCTION DOC] Hook: returns the default choice for a sample (for example what a
        # handwritten heuristic picks); default none, which disables default comparison.
        """
        Returns the default config for a given sample. The default config could for example be the config that is
        the chosen by a current handwritten heuristic. This can for example be used in get_unsafe_leaf to
        compare the predicted config with the default config.
        """
        return None

    def gen_predict_fn_def(self):
        # [FUNCTION DOC] Returns the definition line text of the generated best-choices method.
        """
        Generates the definition of the predict function.
        """
        return "def get_best_choices(self, context: AHContext) -> Optional[list[tuple[float, int]]]:"

    def codegen_boilerplate(
        self, heuristic_name, opt_name, threshold, shared_memory, device_capa, classes
    ):
        # [FUNCTION DOC] Returns the source text of the generated decision heuristic class
        # header: file header comments, imports, class with choices filling, precondition,
        # confidence threshold, choice lookup and name getter. The tree body is appended later.
        """
        Generates the boilerplate code for the generated heuristic. This includes things like imports, class definition,
        etc.
        """

        boiler_plate = f"""# flake8: noqa: B950
# fmt: off
# This file was generated by AutoHeuristic. Do not modify it manually!
# To regenerate this file, take a look at the steps in the README.md file inside torchgen/_autoheuristic/{opt_name}/
from typing import Optional

from torch._inductor.autoheuristic.autoheuristic_utils import (
    AHContext,
    AHMetadata,
    Choice,
)
from torch._inductor.autoheuristic.learnedheuristic_interface import (
    LearnedHeuristicDecision,
)


class {heuristic_name}(LearnedHeuristicDecision):

    def __init__(self) -> None:
        self.choices: list[Choice] = []
        self.fill_choices()

{self.gen_precondition(opt_name, shared_memory, device_capa)}

    def get_confidence_threshold(self) -> float:
        return {threshold}

    def get_choice(self, idx: int) -> Optional[str]:
        if idx < len(self.choices):
            return self.choices[idx]
        return None

    def fill_choices(self) -> None:
{self.gen_classes(classes, num_spaces=8)}

    def get_name(self) -> str:
        return '{opt_name}'"""
        return boiler_plate

    def add_real_datasets(
        self, datasets, other_datasets, cat_feature2cats, ranking=False
    ):
        # [FUNCTION DOC] For each (name, path) pair in other_datasets (may be none) loads the
        # table via get_df (no filters, near-best expansion when ranking) and stores it in the
        # datasets mapping under name. Mutates datasets.
        """
        Adds datasets specified by the user to the datasets dictionary.
        """
        if other_datasets:
            for name, path in other_datasets:
                (df_other, choices, _, _, _) = self.get_df(
                    path,
                    cat_feature2cats=cat_feature2cats,
                    apply_filters=False,
                    add_near_best=ranking,
                )
                datasets[name] = df_other

    def codegen(
        self,
        tree,
        metadata,
        heuristic_name,
        threshold,
        dummy_col_2_col_val,
        unsafe_leaves,
    ):
        # [FUNCTION DOC] Generates the heuristic source from a trained tree: reads device
        # capability, name and shared memory from metadata, builds the boilerplate and the
        # predict method definition, lets the tree emit its body (with unsafe leaves returning
        # unsure), then writes the lines to the artifact file. Side effect: file written.
        lines = []
        device_capa = metadata["device_capa"]
        device_capa_str = f"({device_capa[0]}, {device_capa[1]})"
        opt_name = metadata["name"]
        lines.append(
            self.codegen_boilerplate(
                heuristic_name,
                opt_name,
                threshold,
                metadata["shared_memory"],
                device_capa_str,
                tree.classes_,
            )
        )
        fn_def = f"\n    {self.gen_predict_fn_def()}"
        lines.append(fn_def)
        tree.codegen(dummy_col_2_col_val, lines, unsafe_leaves)
        self.write_heuristic_to_file(lines, heuristic_name)


@dataclass
class AccuracyMetrics:
    # [CLASS DOC] Record of prediction counts: correct, wrong, unsure and total.
    # Number of correct predictions
    num_correct: int
    # Number of wrong predictions
    num_wrong: int
    # Number of predictions where model is unsure
    num_unsure: int
    # Total number of predictions
    total: int

    def to_map(self):
        # [FUNCTION DOC] Returns a mapping with keys correct, wrong, unsure and total.
        return {
            "correct": self.num_correct,
            "wrong": self.num_wrong,
            "unsure": self.num_unsure,
            "total": self.total,
        }


@dataclass
class WrongSpeedupMetrics:
    # [CLASS DOC] Record of how costly wrong predictions were: maximum and geometric mean
    # ratio of the predicted choice's time to the best choice's time.
    # If the model predicted the wrong choice, this is the maximum speedup of the best choice over the predicted choice
    max_speedup: float
    # For all wrong predictions, this is the geometric mean of the speedups of the best choices over the predicted choices
    gmean_speedup: float

    def to_map(self):
        # [FUNCTION DOC] Returns a mapping with keys wrong_max_speedup and wrong_gmean_speedup.
        return {
            "wrong_max_speedup": self.max_speedup,
            "wrong_gmean_speedup": self.gmean_speedup,
        }


@dataclass
class RankingMetrics:
    # [CLASS DOC] Record of top-k ranking quality: counts of hits, misses and unsure cases and
    # the speedups lost when the best choice is missing from the top k.
    # Number of predictions where best choice is in top k choices
    num_correct: int
    # Number of predictions where best choice is not in top k choices
    num_wrong: int
    # Maximum speedup of best choice over best choice in top k (this tells us how much better the best choice, which
    # is not in top k, is over the best choice in top k)
    max_speedup: float
    # Geometric mean of speedups of best choice over best choice in top k
    gmean_speedup: float
    # Number of predictions where model is unsure
    unsure: int

    def to_map(self):
        # [FUNCTION DOC] Returns a mapping with keys top_k_correct, top_k_wrong,
        # wrong_max_speedup_k, wrong_gmean_speedup_k and top_k_unsure.
        return {
            "top_k_correct": self.num_correct,
            "top_k_wrong": self.num_wrong,
            "wrong_max_speedup_k": self.max_speedup,
            "wrong_gmean_speedup_k": self.gmean_speedup,
            "top_k_unsure": self.unsure,
        }


@dataclass
class DefaultComparisonMetrics:
    # [CLASS DOC] Record comparing predictions with the default choice: speedup statistics,
    # number of non-default predictions and how often the default was better.
    # KNOWN ISSUE: default_better is annotated as a boolean but holds a count, and max_slowdown
    # is filled with the minimum speedup over default (a ratio below one), not a slowdown factor.
    # Maximum speedup of predicted choice over default choice
    max_speedup: float
    # Geometric mean of speedups of predicted choices over default choices
    gmean_speedup: float
    # Maximum speedup of default choice over predicted choice
    max_slowdown: float
    # Number of predictions where the predicted choice is not the default choice
    non_default_predictions: int
    # Number of predictions where the default choice is better than the predicted choice
    default_better: bool

    def to_map(self):
        # [FUNCTION DOC] Returns a mapping with keys max_speedup_over_default,
        # gmean_speedup_over_default, max_speedup_default_over_heuristic,
        # non_default_predictions and default_better.
        return {
            "max_speedup_over_default": self.max_speedup,
            "gmean_speedup_over_default": self.gmean_speedup,
            "max_speedup_default_over_heuristic": self.max_slowdown,
            "non_default_predictions": self.non_default_predictions,
            "default_better": self.default_better,
        }


@dataclass
class EvalResults:
    # [CLASS DOC] Bundle of the four metric records (accuracy, speedup, ranking and default
    # comparison) produced by one evaluation.
    accuracy: AccuracyMetrics
    speedup: WrongSpeedupMetrics
    ranking: RankingMetrics
    default_comparison: DefaultComparisonMetrics

    def to_map(self):
        # [FUNCTION DOC] Returns one flat mapping merging the maps of all four records.
        return {
            **self.accuracy.to_map(),
            **self.speedup.to_map(),
            **self.ranking.to_map(),
            **self.default_comparison.to_map(),
        }


class DecisionEvaluator:
    # [CLASS DOC] Scores a trained decision tree on a table. Inputs: trainer, model, predictions,
    # table, class probabilities, allowed wrong fraction, probability threshold, top-k size,
    # unsafe leaves, leaf ids and ranking flag. Accumulates counters and lists (correct, wrong,
    # unsure, wrong probabilities and speedups, top-k stats, default comparison) during
    # get_results, so one instance should be used for a single evaluation.
    def __init__(
        self,
        train,
        model,
        predictions,
        df,
        probas,
        wrong_pct=0.01,
        threshold=0.0,
        k=10,
        unsafe_leaves=None,
        leaf_ids=None,
        ranking=False,
    ) -> None:
        # [FUNCTION DOC] Stores the arguments and zero-initializes all accumulators.
        self.train = train
        self.model = model
        self.predictions = predictions
        self.df = df
        self.probas = probas
        self.wrong_pct = wrong_pct
        self.threshold = threshold
        self.k = k
        self.unsafe_leaves = unsafe_leaves
        self.leaf_ids = leaf_ids
        self.ranking = ranking

        self.num_correct = 0
        self.num_wrong = 0
        self.num_unsure = 0
        self.wrong_probas = []
        self.speedups_wrong = []
        self.num_correct_top_k = 0
        self.num_wrong_top_k = 0
        self.wrong_speedups_top_k = []
        self.top_k_unsure = 0
        self.num_non_default_predictions = 0
        self.speedups_over_default = []
        self.num_default_better = 0

    def compute_speedup_over_default(self, default_config, pred, i, predicted_time):
        # [FUNCTION DOC] HELPER: If a default choice exists, counts a non-default prediction,
        # looks up the default's time for row i and, when finite, records default time divided
        # by predicted_time (counting cases below one as default better). Otherwise logs a debug
        # message. Mutates accumulators.
        # KNOWN ISSUE: the debug log formats a possibly absent time with an integer format.
        if default_config is not None:
            if pred != default_config:
                self.num_non_default_predictions += 1
            default_time = self.get_time(self.df.iloc[i], default_config)
            # TODO: We should keep track of how often this happens
            if default_time is not None and not math.isinf(default_time):
                speedup_over_default = default_time / predicted_time
                if speedup_over_default < 1:
                    self.num_default_better += 1
                self.speedups_over_default.append(speedup_over_default)
            else:
                log.debug(
                    "cannot compute speedup over default because default_time=%d",
                    default_time,
                )

    def get_time(self, row, choice):
        # [FUNCTION DOC] HELPER: Returns the measured time of the choice from the row's JSON
        # choice-to-time map, or none if the choice is not present.
        choices_feedback = json.loads(row["choice2time"])
        return choices_feedback.get(choice, None)

    def top_k_classes(self, model, probas, k, avail_choices):
        # [FUNCTION DOC] Returns up to k classes with the highest nonzero probability that are
        # available, then appends the trainer's always-included choices and removes duplicates
        # keeping order. Note always-included choices are added even if unavailable.
        # Get classes and their corresponding probabilities
        classes = model.classes_

        # Sort by probability (descending) and filter out zero probabilities
        sorted_classes = [
            c
            for c, p in sorted(zip(classes, probas), key=lambda x: x[1], reverse=True)
            if p > 0 and c in avail_choices
        ]

        # Return top k choices
        top_k_choices = sorted_classes[:k]
        top_k_choices += self.train.ranking_always_included_choices()
        top_k_choices = list(dict.fromkeys(top_k_choices))
        return top_k_choices

    def eval_prediction(
        self, avail_choices, leaf_id, pred, true, prob, threshold, default_config, i
    ):
        # [FUNCTION DOC] Classifies the prediction for row i as unsure (unsafe leaf, predicted
        # choice unavailable, or top probability not above the threshold unless it is exactly
        # one), correct (equals true) or wrong. Correct and wrong update the default comparison;
        # wrong also records the probability and predicted-to-best time ratio. Mutates counters.
        predicted_time = self.get_time(self.df.iloc[i], pred)
        max_prob = max(prob)
        if (
            leaf_id in self.unsafe_leaves
            or pred not in avail_choices
            or (max_prob != 1.0 and max_prob <= threshold)
        ):
            self.num_unsure += 1
            self.speedups_over_default.append(1.0)
        elif pred == true:
            self.compute_speedup_over_default(default_config, pred, i, predicted_time)
            self.num_correct += 1
        else:
            self.compute_speedup_over_default(default_config, pred, i, predicted_time)
            self.num_wrong += 1
            self.wrong_probas.append(max_prob)
            best_time = self.get_time(self.df.iloc[i], true)
            wrong_speedup = predicted_time / best_time
            self.speedups_wrong.append(wrong_speedup)

    def eval_ranking_prediction(self, true, top_k_choices, i):
        # [FUNCTION DOC] Counts a top-k hit if the true best choice is among top_k_choices.
        # Otherwise finds the fastest available time among them: if any exists, records the ratio
        # of that time to the best time and counts a miss; if none, counts an unsure case.
        if true in top_k_choices:
            self.num_correct_top_k += 1
        else:
            top_k_choices_times = []
            for choice in top_k_choices:
                time = self.get_time(self.df.iloc[i], choice)
                if time is not None:
                    top_k_choices_times.append(time)
            best_time = self.get_time(self.df.iloc[i], true)
            min_time = min(top_k_choices_times, default=None)
            if min_time is not None:
                speedup = min_time / best_time
                self.wrong_speedups_top_k.append(speedup)
                self.num_wrong_top_k += 1
            else:
                self.top_k_unsure += 1
                # TODO (AlnisM): print more info (input and choices)
                log.debug(
                    "All top k choices have no time which means all top k are unavailable"
                )

    def get_safe_proba(self):
        # [FUNCTION DOC] Runs the evaluation and returns the probability threshold above which
        # the allowed wrong fraction is respected.
        return self.get_results(return_safe_proba=True)

    def compute_safe_proba(self, num_predictions, wrong_probas, wrong_pct):
        # [FUNCTION DOC] Sorts wrong_probas in place. If the number of wrong predictions is within
        # the allowance (predictions times wrong_pct, truncated) returns 0.0; otherwise returns
        # the sorted probability at the index equal to the excess count, capped at the last one.
        wrong_probas.sort()
        num_wrong = len(wrong_probas)
        allowed_wrong = int(num_predictions * wrong_pct)
        if allowed_wrong >= num_wrong:
            return 0.0
        too_many_wrong = num_wrong - allowed_wrong
        idx = min(too_many_wrong, len(wrong_probas) - 1)
        return wrong_probas[idx]

    def get_results(self, return_safe_proba=False) -> EvalResults:
        # [FUNCTION DOC] Evaluates every row (true label is the actual winner when ranking, else
        # the winner), computing top-k choices, the default config, the prediction outcome and
        # the ranking outcome. Raises AssertionError if the true best choice is unavailable.
        # With return_safe_proba returns the safe probability number instead; otherwise returns
        # EvalResults with accuracy, wrong-speedup, ranking and default-comparison metrics.
        """
        Custom evaluation function that evaluates a learned decision tree.
        """

        y_true = self.df["actual_winner"] if self.ranking else self.df["winner"]
        for i, (pred, true, prob, leaf_id) in enumerate(
            zip(self.predictions, y_true, self.probas, self.leaf_ids)
        ):
            avail_choices = self.df["avail_choices"].iloc[i]
            top_k_choices = self.top_k_classes(
                self.model, prob, k=self.k, avail_choices=avail_choices
            )
            if true not in avail_choices:
                raise AssertionError(
                    f"Best choice {true} not in available choices {avail_choices}"
                )
            default_config = self.train.get_default_config(self.df.iloc[i])
            self.eval_prediction(
                avail_choices,
                leaf_id,
                pred,
                true,
                prob,
                self.threshold,
                default_config,
                i,
            )
            self.eval_ranking_prediction(true, top_k_choices, i)

        total = len(self.predictions)
        if return_safe_proba:
            return self.compute_safe_proba(total, self.wrong_probas, self.wrong_pct)

        def safe_gmean(x):
            # [FUNCTION DOC] HELPER: Geometric mean of a list, or 0 for an empty list.
            return gmean(x) if x else 0

        max_speedup = max(self.speedups_wrong, default=0)
        gmean_speedup = safe_gmean(self.speedups_wrong)
        max_speedup_top_k = max(self.wrong_speedups_top_k, default=0)
        gmean_speedup_top_k = safe_gmean(self.wrong_speedups_top_k)
        max_speedup_over_default = max(self.speedups_over_default, default=0)
        gmean_speedup_over_default = safe_gmean(self.speedups_over_default)
        max_slowdown_over_default = min(self.speedups_over_default, default=0)

        accuracyMetrics = AccuracyMetrics(
            self.num_correct, self.num_wrong, self.num_unsure, total
        )
        wrongSpeedupMetrics = WrongSpeedupMetrics(max_speedup, gmean_speedup)
        rankingMetrics = RankingMetrics(
            self.num_correct_top_k,
            self.num_wrong_top_k,
            max_speedup_top_k,
            gmean_speedup_top_k,
            self.top_k_unsure,
        )
        defaultComparisonMetrics = DefaultComparisonMetrics(
            max_speedup_over_default,
            gmean_speedup_over_default,
            max_slowdown_over_default,
            self.num_non_default_predictions,
            self.num_default_better,
        )
        return EvalResults(
            accuracyMetrics,
            wrongSpeedupMetrics,
            rankingMetrics,
            defaultComparisonMetrics,
        )


if __name__ == "__main__":
    train = AHTrainDecisionTree()
    train.generate_heuristic()
