# mypy: ignore-errors
# [MODULE DOC]
# Regression tree training script for the pad_mm heuristic: subclasses AHTrainRegressionTree
# adding the pad_mm feature columns. Appends the parent directory to sys.path at import time;
# running it as a script trains and generates the heuristic. Near duplicate of train_pad_mm.py.
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).absolute().parents[1]))

from train_regression import AHTrainRegressionTree

from torch._inductor.fx_passes.pad_mm import pad_mm_operations


class AHTrainPadMM(AHTrainRegressionTree):
    # [CLASS DOC] Regression tree trainer for pad_mm; only overrides feature engineering.
    def __init__(self):
        # [FUNCTION DOC] Calls the base class initializer.
        super().__init__()

    def add_new_features(self, results):
        # [FUNCTION DOC] Adds one column per pad_mm feature operation by applying its function
        # to each row of the results table. Returns the table and the names of the added
        # columns that are categorical. Mutates the given table.
        ops = pad_mm_operations()
        for op in ops:
            results[op.name] = results.apply(op.func, axis=1)
        added_categorical_features = [op.name for op in ops if op.is_categorical]
        return (results, added_categorical_features)


if __name__ == "__main__":
    train = AHTrainPadMM()
    train.generate_heuristic()
