# [MODULE DOC] End-to-end script that fine-tunes a BERT sequence classifier on a subset of the
# yelp review dataset, once with native PyTorch and once through a chosen dynamo backend, then
# compares the training loss history and reports the per-epoch time of each. Import-time side
# effects: enables tf32 matmul on cuda, disables tokenizer parallelism through an environment
# variable, and defines the module-level device (cuda when available, else cpu). Requires the
# datasets and transformers packages and downloads data and model weights.
import argparse
import inspect
import os
import sys
import time
from datetime import timedelta

from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import torch
import torch._dynamo
from torch.utils.data import DataLoader


torch.backends.cuda.matmul.allow_tf32 = True

# You will download around 84G dataset if you run this end to end training/evaluation example.

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def data_processing(num_samples, batch_size):
    # [FUNCTION DOC] Loads the yelp review dataset and the cased BERT tokenizer, tokenizes the
    # text field with padding to maximum length and truncation, drops the text column, renames
    # the label column to labels, and sets the torch format. Selects the first num_samples
    # examples of the train and test splits and wraps each in a data loader with the given batch
    # size. Returns the train and evaluation loaders.
    dataset = load_dataset("yelp_review_full")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize_function(examples):
        # [FUNCTION DOC] HELPER: Tokenizes a batch of examples using the text field with padding
        # and truncation, returning the tokenizer output.
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    small_train_dataset = tokenized_datasets["train"].select(range(num_samples))
    small_eval_dataset = tokenized_datasets["test"].select(range(num_samples))

    train_dataloader = DataLoader(small_train_dataset, batch_size=batch_size)
    eval_dataloader = DataLoader(small_eval_dataset, batch_size=batch_size)

    return train_dataloader, eval_dataloader


def training_iter_fn(batch, model, optimizer):
    # [FUNCTION DOC] One training step: runs the model on the batch (keyword arguments), takes
    # the loss from the output, backpropagates, steps the optimizer and clears gradients.
    # Returns the loss tensor.
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss


def model_training_evaluation(
    backend, train_dataloader, eval_dataloader, model, optimizer, num_epochs, evaluation
):
    # [FUNCTION DOC] Trains the model on the module-level device for num_epochs epochs and
    # optionally evaluates it. With no backend the step function runs natively, otherwise it is
    # wrapped by dynamo optimization with that backend. A running loss is accumulated per batch
    # and each 100th batch appends the mean over 100 batches to the loss history and resets the
    # accumulator (it is not reset at epoch boundaries). If evaluation is requested, loads the
    # accuracy metric, switches to eval mode, optionally wraps the model with the backend, and
    # accumulates predictions (arg-max of logits) and references without gradients. Returns the
    # loss history together with the computed metric, or None for the metric if not evaluating.
    # Mutates the model and optimizer (weights are updated in place).
    model.to(device)
    model.train()
    loss_history = []
    if not backend:
        # Run with native Pytorch
        opt_training_iter_fn = training_iter_fn
    else:
        # Support backends: eager, aot_eager, aot_nvfuser and inductor
        opt_training_iter_fn = torch._dynamo.optimize(backend)(training_iter_fn)
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, batch in enumerate(train_dataloader, 0):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = opt_training_iter_fn(batch, model, optimizer)
            running_loss += loss.item()
            if i % 100 == 99:
                loss_history.append(running_loss / 100)
                running_loss = 0.0

    if evaluation:
        metric = load_metric("accuracy")
        model.eval()
        if not backend:
            opt_model = model
        else:
            opt_model = torch._dynamo.optimize(backend)(model)
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = opt_model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])

        return loss_history, metric.compute()
    else:
        return loss_history, None


def check_loss(ref_loss, res_loss):
    # [FUNCTION DOC] Compares two loss histories: raises AssertionError if their lengths differ.
    # Returns true when the sum of the last up to ten values of the result history divided by
    # ten is at most the same quantity of the reference history plus 0.1.
    # KNOWN ISSUE: the sums are divided by ten even when fewer than ten values exist, so short
    #     histories are scaled down inconsistently with the number of values summed.
    if len(ref_loss) != len(res_loss):
        raise AssertionError(
            f"Expected loss lists to have equal length, but got {len(ref_loss)} and {len(res_loss)}"
        )
    length = len(ref_loss)
    x = min(length, 10)
    return sum(res_loss[-x:]) / 10 <= sum(ref_loss[-x:]) / 10 + 0.1


def parse_args():
    # [FUNCTION DOC] Parses command line options: epochs (default 10), num-samples (1000),
    # batch-size (8), learning rate (5e-5), backend (any registered dynamo backend, default
    # inductor), optimizer name (default Adam) and an evaluation flag. Returns the namespace.
    parser = argparse.ArgumentParser(
        description="TorchDynamo end to end training/evaluation benchmark"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs to train (default: 10)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="number of samples to train/eval (default: 1000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="input batch size for training (default: 8)",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5, help="learning rate (default: 5e-5)"
    )
    parser.add_argument(
        "--backend",
        choices=torch._dynamo.list_backends(exclude_tags=None),
        default="inductor",
        help="train/evaluate model with a given backend (default: inductor)",
    )
    parser.add_argument(
        "--optimizer",
        default="Adam",
        help="train model using a given optimizer (default: Adam)",
    )
    parser.add_argument(
        "--evaluation",
        action="store_true",
        help="running evaluation after model training",
    )
    args = parser.parse_args()
    return args


def main():
    # [FUNCTION DOC] Entry point. Parses arguments, builds the data loaders and the pretrained
    # five-label BERT classifier, and creates the chosen torch optimizer by name (enabling its
    # capturable option when supported). Runs training (and optional evaluation) natively and
    # times it, then runs the same model and optimizer again through the chosen backend and
    # times it. Prints a pass or fail line based on the loss comparison, the accuracy if
    # evaluation was requested, and the per-epoch time of both runs.
    # KNOWN ISSUE: both runs reuse the same model and optimizer without resetting weights, so
    #     the second run starts from already-trained weights, and the reported accuracy is that
    #     of the dynamo run only.
    args = parse_args()
    train_dataloader, eval_dataloader = data_processing(
        args.num_samples, args.batch_size
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=5
    )
    optimizer_cls = getattr(sys.modules["torch.optim"], args.optimizer)
    if "capturable" in inspect.signature(optimizer_cls).parameters:
        optimizer = optimizer_cls(model.parameters(), lr=args.lr, capturable=True)
    else:
        optimizer = optimizer_cls(model.parameters(), lr=args.lr)
    native_start = time.time()
    ref_loss, accuracy = model_training_evaluation(
        None,
        train_dataloader,
        eval_dataloader,
        model,
        optimizer,
        args.epochs,
        args.evaluation,
    )
    native_end = time.time()
    res_loss, accuracy = model_training_evaluation(
        args.backend,
        train_dataloader,
        eval_dataloader,
        model,
        optimizer,
        args.epochs,
        args.evaluation,
    )
    dynamo_end = time.time()
    if check_loss(ref_loss, res_loss):
        print(
            "[PASSED] TorchDynamo end to end training loss is less than or equal to native PyTorch"
        )
    else:
        print(
            "[FAILED] TorchDynamo end to end training loss is greater than native Pytorch"
        )
    if args.evaluation:
        print(f"Model accuracy: {accuracy}")
    native_elapsed = native_end - native_start
    dynamo_elapsed = dynamo_end - native_end
    print(
        f"Train model on {args.epochs} epochs with backend {args.backend} and optimizer {args.optimizer}:"
    )
    print(f"PyTorch spent {timedelta(seconds=native_elapsed / args.epochs)} per epoch")
    print(
        f"TorchDynamo spent {timedelta(seconds=dynamo_elapsed / args.epochs)} per epoch"
    )


if __name__ == "__main__":
    main()
