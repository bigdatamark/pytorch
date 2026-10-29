# mypy: ignore-errors
# [MODULE DOC]
# Decision tree training script for the mm heuristic. Subclasses AHTrainDecisionTree with mm
# features, fallback choice, split sizes, grid and extra weighting of the TIMM and HuggingFace
# datasets. Appends the parent directory to sys.path at import time; running it as a script
# trains and generates the heuristic.
import sys
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]


sys.path.append(str(Path(__file__).absolute().parents[1]))

from train_decision import AHTrainDecisionTree

from torch._inductor.autoheuristic.autoheuristic_utils import mm_operations


class AHTrainDecisionTreeMM(AHTrainDecisionTree):
    # [CLASS DOC] Decision tree trainer for mm; customizes feature engineering, the default
    # choice, error tolerance, split sizes, grid, training data composition and ranking.
    def __init__(self):
        # [FUNCTION DOC] Calls the base class initializer.
        super().__init__()

    def add_new_features(self, results):
        # [FUNCTION DOC] Adds one column per mm feature operation by applying its function to
        # each row of the results table. Returns the table and the names of the added columns
        # that are categorical. Mutates the given table.
        ops = mm_operations()
        added_categorical_features = []
        for op in ops:
            results[op.name] = results.apply(op.func, axis=1)
            if op.is_categorical:
                added_categorical_features.append(op.name)
        return (results, added_categorical_features)

    def get_default_config(self, row):
        # [FUNCTION DOC] Returns the fallback choice name, always the extern mm choice.
        return "extern_mm"

    def get_allowed_wrong_prediction_pct(self):
        # [FUNCTION DOC] Returns 1.0, i.e. any fraction of wrong predictions is tolerated.
        return 1.0

    def get_test_and_val_size(self):
        # [FUNCTION DOC] Returns the (test, validation) split fractions: (0.01, 0.19).
        return (0.01, 0.19)

    def get_grid_search_values(self):
        # [FUNCTION DOC] Returns a single-point hyperparameter grid: max_depth 5,
        # min_samples_leaf 0.01, entropy criterion.
        return {"max_depth": [5], "min_samples_leaf": [0.01], "criterion": ["entropy"]}

    def add_training_data(self, df_train, datasets):
        # [FUNCTION DOC] Builds the training table: the "train" dataset plus the "train_timm"
        # and "train_hf" datasets each repeated three times, concatenated with a fresh index.
        # The df_train argument is ignored (overwritten from datasets). Returns the table.
        # add each dataset to the training data 3 times
        # we really want to make sure that the heuristic performs well on these datasets
        df_timm_train = datasets["train_timm"]
        df_timm_train = df_timm_train.loc[df_timm_train.index.repeat(3)].reset_index(
            drop=True
        )
        df_hf_train = datasets["train_hf"]
        df_hf_train = df_hf_train.loc[df_hf_train.index.repeat(3)].reset_index(
            drop=True
        )
        df_train = datasets["train"]
        df_train = pd.concat(
            [df_train, df_timm_train, df_hf_train],
            ignore_index=True,
        )
        return df_train

    def ranking_always_included_choices(self):
        # [FUNCTION DOC] Returns the choices always kept when ranking: only "extern_mm".
        return ["extern_mm"]


if __name__ == "__main__":
    train = AHTrainDecisionTreeMM()
    train.generate_heuristic()
