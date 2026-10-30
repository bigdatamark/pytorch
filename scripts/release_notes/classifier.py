# [MODULE DOC] Trains and defines a neural classifier that suggests a release notes category for
# a commit from its title, changed files and author (used by the categorize tool). Contents: a
# frozen pretrained XLM-R text encoder wrapped in the CommitClassifier module with learned
# author and file embeddings, helpers that build and cache the train, validation and test splits
# and the author and file index maps (files under a results directory), batch building, class
# weighting, optional oversampling when imblearn is installed, and a training loop with a
# command line. Import-time side effects: imports torchtext, pandas and tqdm, chooses a global
# device (cuda if available), and looks up the pretrained encoder bundle.
import argparse
import math
import pickle
import random
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Dict, List

import common
import pandas as pd
import torchtext
from torchtext.functional import to_tensor
from tqdm import tqdm

import torch
import torch.nn as nn


XLMR_BASE = torchtext.models.XLMR_BASE_ENCODER
# This should not be here but it works for now
device = "cuda" if torch.cuda.is_available() else "cpu"

HAS_IMBLEARN = False
try:
    import imblearn

    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

# 94% of all files are captured at len 5, good hyperparameter to play around with.
MAX_LEN_FILE = 6

UNKNOWN_TOKEN = "<Unknown>"

# Utilities for working with a truncated file graph


def truncate_file(file: Path, max_len: int = 5):
    # [FUNCTION DOC] Returns the first max_len path components of a path joined with forward
    # slashes, as text.
    return ("/").join(file.parts[:max_len])


def build_file_set(all_files: List[Path], max_len: int):
    # [FUNCTION DOC] Returns the set of distinct truncated path strings for a list of paths.
    truncated_files = [truncate_file(file, max_len) for file in all_files]
    return set(truncated_files)


@dataclass
class CommitClassifierInputs:
    # [CLASS DOC] Batch input for the classifier: three parallel lists with one entry per commit
    # holding the title text, the space separated changed file paths, and the author name.
    title: List[str]
    files: List[str]
    author: List[str]


@dataclass
class CategoryConfig:
    # [CLASS DOC] Hyperparameters for the classifier: the list of category names plus sizes
    # (text feature width 768, inner width 128, dropout 0.1, author embedding size 8, file
    # embedding size 32) and the activation class (rectified linear unit). Plain dataclass;
    # activation is a class attribute, not a configurable field.
    categories: List[str]
    input_dim: int = 768
    inner_dim: int = 128
    dropout: float = 0.1
    activation = nn.ReLU
    embedding_dim: int = 8
    file_embedding_dim: int = 32


class CommitClassifier(nn.Module):
    # [CLASS DOC] Torch module that scores categories for commits. Holds a frozen XLM-R encoder
    # and its text transform, author and file index maps, an author embedding table, a summing
    # embedding bag for files, and for each of title, files and author a dense layer and output
    # projection to the category count. The output logits are the sum of the three branches.
    # Only the new layers train.
    def __init__(
        self,
        encoder_base: torchtext.models.XLMR_BASE_ENCODER,
        author_map: Dict[str, int],
        file_map: [str, int],
        config: CategoryConfig,
    ):
        # [FUNCTION DOC] Builds the layers from the encoder bundle, index maps and config. The
        # encoder is created from the bundle and its gradients are disabled; the size of each
        # embedding follows the map sizes; the number of outputs follows the number of
        # categories.
        super().__init__()
        self.encoder = encoder_base.get_model().requires_grad_(False)
        self.transform = encoder_base.transform()
        self.author_map = author_map
        self.file_map = file_map
        self.categories = config.categories
        self.num_authors = len(author_map)
        self.num_files = len(file_map)
        self.embedding_table = nn.Embedding(self.num_authors, config.embedding_dim)
        self.file_embedding_bag = nn.EmbeddingBag(
            self.num_files, config.file_embedding_dim, mode="sum"
        )
        self.dense_title = nn.Linear(config.input_dim, config.inner_dim)
        self.dense_files = nn.Linear(config.file_embedding_dim, config.inner_dim)
        self.dense_author = nn.Linear(config.embedding_dim, config.inner_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.out_proj_title = nn.Linear(config.inner_dim, len(self.categories))
        self.out_proj_files = nn.Linear(config.inner_dim, len(self.categories))
        self.out_proj_author = nn.Linear(config.inner_dim, len(self.categories))
        self.activation_fn = config.activation()

    def forward(self, input_batch: CommitClassifierInputs):
        # [FUNCTION DOC] Computes category logits for a batch. Title branch: tokenizes and pads,
        # runs the encoder, takes the first token features, then dropout, dense, activation,
        # dropout, projection. File branch: splits each commit's file string on spaces,
        # truncates paths, maps them to indexes (unknown path token if missing), sums embeddings
        # per commit with offsets, then dense, activation, dropout and projection. Author
        # branch: looks up the author index (unknown token if missing), embeds, then the same
        # dense stack. Returns the sum of the three, shape batch by category count. Uses the
        # module global device for new tensors, not the module's own device.
        # Encode input title
        title: List[str] = input_batch.title
        model_input = to_tensor(self.transform(title), padding_value=1).to(device)
        title_features = self.encoder(model_input)
        title_embed = title_features[:, 0, :]
        title_embed = self.dropout(title_embed)
        title_embed = self.dense_title(title_embed)
        title_embed = self.activation_fn(title_embed)
        title_embed = self.dropout(title_embed)
        title_embed = self.out_proj_title(title_embed)

        files: list[str] = input_batch.files
        batch_file_indexes = []
        for file in files:
            paths = [
                truncate_file(Path(file_part), MAX_LEN_FILE)
                for file_part in file.split(" ")
            ]
            batch_file_indexes.append(
                [
                    self.file_map.get(file, self.file_map[UNKNOWN_TOKEN])
                    for file in paths
                ]
            )

        flat_indexes = torch.tensor(
            list(chain.from_iterable(batch_file_indexes)),
            dtype=torch.long,
            device=device,
        )
        offsets = [0]
        offsets.extend(len(files) for files in batch_file_indexes[:-1])
        offsets = torch.tensor(offsets, dtype=torch.long, device=device)
        offsets = offsets.cumsum(dim=0)

        files_embed = self.file_embedding_bag(flat_indexes, offsets)
        files_embed = self.dense_files(files_embed)
        files_embed = self.activation_fn(files_embed)
        files_embed = self.dropout(files_embed)
        files_embed = self.out_proj_files(files_embed)

        # Add author embedding
        authors: List[str] = input_batch.author
        author_ids = [
            self.author_map.get(author, self.author_map[UNKNOWN_TOKEN])
            for author in authors
        ]
        author_ids = torch.tensor(author_ids).to(device)
        author_embed = self.embedding_table(author_ids)
        author_embed = self.dense_author(author_embed)
        author_embed = self.activation_fn(author_embed)
        author_embed = self.dropout(author_embed)
        author_embed = self.out_proj_author(author_embed)

        return title_embed + files_embed + author_embed

    def convert_index_to_category_name(self, most_likely_index):
        # [FUNCTION DOC] Converts a category index (integer) to its name, or a one dimensional
        # tensor of indexes to a list of names. Returns None for any other input type.
        if isinstance(most_likely_index, int):
            return self.categories[most_likely_index]
        elif isinstance(most_likely_index, torch.Tensor):
            return [self.categories[i] for i in most_likely_index]

    def get_most_likely_category_name(self, input):
        # [FUNCTION DOC] Returns the list of the highest scoring category names, one per commit
        # in the batch. Runs forward (no gradient disabling here) and takes the maximum along
        # the category dimension. The input is a classifier input record, not a dictionary as
        # the old comment says.
        # Input will be a dict with title and author keys
        logits = self.forward(input)
        most_likely_index = torch.argmax(logits, dim=1)
        return self.convert_index_to_category_name(most_likely_index)


def get_train_val_data(data_folder: Path, regen_data: bool, train_percentage=0.95):
    # [FUNCTION DOC] Returns the training and validation tables. Reads saved split files from
    # the data folder when they exist and regeneration was not requested. Otherwise reads the
    # commit list file, sets aside Uncategorized rows as the test set, removes skip rows,
    # shuffles the rest randomly (unseeded), splits by the training fraction (default 0.95),
    # writes the three split files into the folder and returns the training and validation
    # parts. Prints progress.
    if (
        not regen_data
        and Path(data_folder / "train_df.csv").exists()
        and Path(data_folder / "val_df.csv").exists()
    ):
        train_data = pd.read_csv(data_folder / "train_df.csv")
        val_data = pd.read_csv(data_folder / "val_df.csv")
        return train_data, val_data
    else:
        print("Train, Val, Test Split not found generating from scratch.")
        commit_list_df = pd.read_csv(data_folder / "commitlist.csv")
        test_df = commit_list_df[commit_list_df["category"] == "Uncategorized"]
        all_train_df = commit_list_df[commit_list_df["category"] != "Uncategorized"]
        # We are going to drop skip from training set since it is so imbalanced
        print(
            "We are removing skip categories, YOU MIGHT WANT TO CHANGE THIS, BUT THIS IS A MORE HELPFUL CLASSIFIER FOR LABELING."
        )
        all_train_df = all_train_df[all_train_df["category"] != "skip"]
        all_train_df = all_train_df.sample(frac=1).reset_index(drop=True)
        split_index = math.floor(train_percentage * len(all_train_df))
        train_df = all_train_df[:split_index]
        val_df = all_train_df[split_index:]
        print("Train data size: ", len(train_df))
        print("Val data size: ", len(val_df))

        test_df.to_csv(data_folder / "test_df.csv", index=False)
        train_df.to_csv(data_folder / "train_df.csv", index=False)
        val_df.to_csv(data_folder / "val_df.csv", index=False)
        return train_df, val_df


def get_author_map(data_folder: Path, regen_data, assert_stored=False):
    # [FUNCTION DOC] Returns the mapping from author name to index. Loads the saved mapping from
    # the data folder if present and regeneration is not requested. Otherwise raises
    # FileNotFoundError when assert_stored is set; else builds the mapping from the unique
    # authors in the commit list file plus the unknown token, saves it and returns it.
    if not regen_data and Path(data_folder / "author_map.pkl").exists():
        with open(data_folder / "author_map.pkl", "rb") as f:
            return pickle.load(f)
    else:
        if assert_stored:
            raise FileNotFoundError(
                "Author map not found, you are loading for inference you need to have an author map!"
            )
        print("Regenerating Author Map")
        all_data = pd.read_csv(data_folder / "commitlist.csv")
        authors = all_data.author.unique().tolist()
        authors.append(UNKNOWN_TOKEN)
        author_map = {author: i for i, author in enumerate(authors)}
        with open(data_folder / "author_map.pkl", "wb") as f:
            pickle.dump(author_map, f)
        return author_map


def get_file_map(data_folder: Path, regen_data, assert_stored=False):
    # [FUNCTION DOC] Returns the mapping from truncated file path to index. Loads the saved
    # mapping if present and regeneration is not requested. Otherwise raises FileNotFoundError
    # when assert_stored is set; else collects every path from the space separated file columns
    # of the commit list file, adds the unknown token, truncates paths, numbers the distinct
    # ones (order of a set), saves and returns the mapping.
    if not regen_data and Path(data_folder / "file_map.pkl").exists():
        with open(data_folder / "file_map.pkl", "rb") as f:
            return pickle.load(f)
    else:
        if assert_stored:
            raise FileNotFoundError(
                "File map not found, you are loading for inference you need to have a file map!"
            )
        print("Regenerating File Map")
        all_data = pd.read_csv(data_folder / "commitlist.csv")
        # Lets explore files
        files = all_data.files_changed.to_list()

        all_files = []
        for file in files:
            paths = [Path(file_part) for file_part in file.split(" ")]
            all_files.extend(paths)
        all_files.append(Path(UNKNOWN_TOKEN))
        file_set = build_file_set(all_files, MAX_LEN_FILE)
        file_map = {file: i for i, file in enumerate(file_set)}
        with open(data_folder / "file_map.pkl", "wb") as f:
            pickle.dump(file_map, f)
        return file_map


#  Generate a dataset for training


def get_title_files_author_categories_zip_list(dataframe: pd.DataFrame):
    # [FUNCTION DOC] Converts a table to a list of tuples of title, files, author and category,
    # with missing authors replaced by the unknown token.
    title = dataframe.title.to_list()
    files_str = dataframe.files_changed.to_list()
    author = dataframe.author.fillna(UNKNOWN_TOKEN).to_list()
    category = dataframe.category.to_list()
    return list(zip(title, files_str, author, category))


def generate_batch(batch):
    # [FUNCTION DOC] Collates a list of tuples into a classifier input record and a target
    # tensor of category indexes on the global device. Raises ValueError if a category is not in
    # the known list.
    title, files, author, category = zip(*batch)
    title = list(title)
    files = list(files)
    author = list(author)
    category = list(category)
    targets = torch.tensor([common.categories.index(cat) for cat in category]).to(
        device
    )
    return CommitClassifierInputs(title, files, author), targets


def train_step(batch, model, optimizer, loss):
    # [FUNCTION DOC] One optimization step: clears gradients, runs the model, computes the loss
    # against the targets, backpropagates and steps the optimizer. Returns the loss tensor.
    input, targets = batch
    optimizer.zero_grad()
    output = model(input)
    l = loss(output, targets)
    l.backward()
    optimizer.step()
    return l


@torch.no_grad()
def eval_step(batch, model, loss):
    # [FUNCTION DOC] Computes and returns the loss of the model on a batch without tracking
    # gradients.
    input, targets = batch
    output = model(input)
    l = loss(output, targets)
    return l


def balance_dataset(dataset: List):
    # [FUNCTION DOC] Randomly oversamples a dataset of tuples when imblearn is installed, then
    # draws a random sample of twice the original size from the result and returns tuples with
    # category names; returns the dataset unchanged when imblearn is missing. Oversampling uses
    # a fixed seed but the final draw does not. Raises ValueError if the oversampled set is
    # smaller than the requested sample.
    if not HAS_IMBLEARN:
        return dataset
    title, files, author, category = zip(*dataset)
    category = [common.categories.index(cat) for cat in category]
    inpt_data = list(zip(title, files, author))
    from imblearn.over_sampling import RandomOverSampler

    # from imblearn.under_sampling import RandomUnderSampler
    rus = RandomOverSampler(random_state=42)
    X, y = rus.fit_resample(inpt_data, category)
    merged = list(zip(X, y))
    merged = random.sample(merged, k=2 * len(dataset))
    X, y = zip(*merged)
    rebuilt_dataset = []
    for i in range(len(X)):
        rebuilt_dataset.append((*X[i], common.categories[y[i]]))
    return rebuilt_dataset


def gen_class_weights(dataset: List):
    # [FUNCTION DOC] Returns a tensor of per-category loss weights on the global device, ordered
    # like the known categories. Counts examples per category, computes a smoothed mean of the
    # most frequent third and a reduced mean of the least frequent third, clamps each count
    # between them and uses the inverse of the clamped count plus a small constant. Raises
    # ZeroDivisionError for datasets with fewer than three examples.
    from collections import Counter

    epsilon = 1e-1
    title, files, author, category = zip(*dataset)
    category = [common.categories.index(cat) for cat in category]
    counter = Counter(category)
    percentile_33 = len(category) // 3
    most_common = counter.most_common(percentile_33)
    least_common = counter.most_common()[-percentile_33:]
    smoothed_top = sum(i[1] + epsilon for i in most_common) / len(most_common)
    smoothed_bottom = sum(i[1] + epsilon for i in least_common) / len(least_common) // 3
    class_weights = torch.tensor(
        [
            1.0 / (min(max(counter[i], smoothed_bottom), smoothed_top) + epsilon)
            for i in range(len(common.categories))
        ],
        device=device,
    )
    return class_weights


def train(save_path: Path, data_folder: Path, regen_data: bool, resample: bool):
    # [FUNCTION DOC] Trains and saves a classifier. Loads splits and maps, builds the model on
    # the global device, uses weighted cross entropy and Adam with learning rate 3e-3, 25
    # epochs, batches of 256 over a reshuffled training list (optionally oversampled). Each
    # epoch evaluates the validation set as one batch and prints losses; finally prints
    # validation accuracy and saves the state dictionary to the given path.
    # KNOWN ISSUE: the comment says the last batch is made bigger, but the final batch is
    #   actually smaller (the end is clipped to the data size).
    train_data, val_data = get_train_val_data(data_folder, regen_data)
    train_zip_list = get_title_files_author_categories_zip_list(train_data)
    val_zip_list = get_title_files_author_categories_zip_list(val_data)

    classifier_config = CategoryConfig(common.categories)
    author_map = get_author_map(data_folder, regen_data)
    file_map = get_file_map(data_folder, regen_data)
    commit_classifier = CommitClassifier(
        XLMR_BASE, author_map, file_map, classifier_config
    ).to(device)

    # Lets train this bag of bits
    class_weights = gen_class_weights(train_zip_list)
    loss = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(commit_classifier.parameters(), lr=3e-3)

    num_epochs = 25
    batch_size = 256

    if resample:
        # Lets not use this
        train_zip_list = balance_dataset(train_zip_list)
    data_size = len(train_zip_list)

    print(f"Training on {data_size} examples.")
    # We can fit all of val into one batch
    val_batch = generate_batch(val_zip_list)

    for i in tqdm(range(num_epochs), desc="Epochs"):
        start = 0
        random.shuffle(train_zip_list)
        while start < data_size:
            end = start + batch_size
            # make the last batch bigger if needed
            if end > data_size:
                end = data_size
            train_batch = train_zip_list[start:end]
            train_batch = generate_batch(train_batch)
            l = train_step(train_batch, commit_classifier, optimizer, loss)
            start = end

        val_l = eval_step(val_batch, commit_classifier, loss)
        tqdm.write(
            f"Finished epoch {i} with a train loss of: {l.item()} and a val_loss of: {val_l.item()}"
        )

    with torch.no_grad():
        commit_classifier.eval()
        val_inpts, val_targets = val_batch
        val_output = commit_classifier(val_inpts)
        val_preds = torch.argmax(val_output, dim=1)
        val_acc = torch.sum(val_preds == val_targets).item() / len(val_preds)
        print(f"Final Validation accuracy is {val_acc}")

    print(f"Jobs done! Saving to {save_path}")
    torch.save(commit_classifier.state_dict(), save_path)


def main():
    # [FUNCTION DOC] Entry point: parses options (train flag, data folder, save path, regenerate
    # flag, resample flag). With the train flag it trains and saves; otherwise it prints a
    # message that only training is supported.
    parser = argparse.ArgumentParser(
        description="Tool to create a classifier for helping to categorize commits"
    )

    parser.add_argument("--train", action="store_true", help="Train a new classifier")
    parser.add_argument("--commit_data_folder", default="results/classifier/")
    parser.add_argument(
        "--save_path", default="results/classifier/commit_classifier.pt"
    )
    parser.add_argument(
        "--regen_data",
        action="store_true",
        help="Regenerate the training data, helps if labeled more examples and want to re-train.",
    )
    parser.add_argument(
        "--resample",
        action="store_true",
        help="Resample the training data to be balanced. (Only works if imblearn is installed.)",
    )
    args = parser.parse_args()

    if args.train:
        train(
            Path(args.save_path),
            Path(args.commit_data_folder),
            args.regen_data,
            args.resample,
        )
        return

    print(
        "Currently this file only trains a new classifier please pass in --train to train a new classifier"
    )


if __name__ == "__main__":
    main()
