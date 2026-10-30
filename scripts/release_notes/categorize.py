# [MODULE DOC] Interactive command line tool for hand categorizing release note commits. Defines
# the Categorizer class and a main entry that loads the commit list CSV (default under a results
# directory), optionally with a trained classifier that suggests categories, and prompts for a
# category and topic for each commit in the chosen category, saving after every answer. Import
# has no side effects, but the common module it imports needs a GitHub token at import time.
import argparse
import os
import textwrap
from pathlib import Path

import common
from commitlist import CommitList
from common import get_commit_data_cache, topics


class Categorizer:
    # [CLASS DOC] Interactive driver over a CommitList. State: cache of commit features, the
    # commit list, an optional classifier (None when not used) and the category being worked on
    # (Uncategorized by default, or a real category name). categorize walks the matching commits
    # and prompts the user for each, writing the list file after every update.
    def __init__(self, path, category="Uncategorized", use_classifier: bool = False):
        # [FUNCTION DOC] Loads the shared commit data cache and the commit list from the path.
        # With the classifier option, imports torch and the classifier module, picks cuda when
        # available, loads stored author and file maps (asserting they exist) and the saved
        # weights from the results directory, and puts the model in evaluation mode; otherwise
        # the classifier is None. Stores the category filter.
        self.cache = get_commit_data_cache()
        self.commits = CommitList.from_existing(path)
        if use_classifier:
            from classifier import (
                CategoryConfig,
                CommitClassifier,
                get_author_map,
                get_file_map,
                XLMR_BASE,
            )

            import torch

            print("Using a classifier to aid with categorization.")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            classifier_config = CategoryConfig(common.categories)
            author_map = get_author_map(
                Path("results/classifier"), regen_data=False, assert_stored=True
            )
            file_map = get_file_map(
                Path("results/classifier"), regen_data=False, assert_stored=True
            )
            self.classifier = CommitClassifier(
                XLMR_BASE, author_map, file_map, classifier_config
            ).to(device)
            self.classifier.load_state_dict(
                torch.load(Path("results/classifier/commit_classifier.pt"))
            )
            self.classifier.eval()
        else:
            self.classifier = None
        # Special categories: 'Uncategorized'
        # All other categories must be real
        self.category = category

    def categorize(self):
        # [FUNCTION DOC] Prompts through every commit whose category equals the filter. Counts
        # how many commits were already handled, shows each with its position out of the total,
        # and advances to the commit that followed it in the filtered list (or to the jump
        # target if the handler returns one, which it currently never does).
        commits = self.commits.filter(category=self.category)
        total_commits = len(self.commits.commits)
        already_done = total_commits - len(commits)
        i = 0
        while i < len(commits):
            cur_commit = commits[i]
            next_commit = commits[i + 1] if i + 1 < len(commits) else None
            jump_to = self.handle_commit(
                cur_commit, already_done + i + 1, total_commits, commits
            )

            # Increment counter
            if jump_to is not None:
                i = jump_to
            elif next_commit is None:
                i = len(commits)
            else:
                i = commits.index(next_commit)

    def features(self, commit):
        # [FUNCTION DOC] Returns the cached features for a commit, fetching them if needed.
        return self.cache.get(commit.commit_hash)

    def potential_reverts_of(self, commit, commits):
        # [FUNCTION DOC] Finds commits that may revert the given one. Returns an empty list for
        # submodule update titles. Otherwise returns a mapping: an entry from the label name
        # Reverted when the labels have it, plus entries keyed by a one-based display position
        # (offset by 2) for every later commit whose title contains this title with its last ten
        # characters cut off (intended to drop the pull request suffix) and which has a
        # different hash.
        # KNOWN ISSUE: the comment says the cut is eight characters but ten are removed; and the
        #   position keys are approximate by the author's own note.
        submodule_update_str = [
            "Update TensorPipe submodule",
            "Updating submodules",
            "Automated submodule update",
        ]
        if any(a in commit.title for a in submodule_update_str):
            return []

        features = self.features(commit)
        if "Reverted" in features.labels:
            reasons = {"GithubBot": "Reverted"}
        else:
            reasons = {}

        index = commits.index(commit)
        # -8 to remove the (#35011)
        cleaned_title = commit.title[:-10]
        # NB: the index + 2 is sketch
        reasons.update(
            {
                (index + 2 + delta): cand
                for delta, cand in enumerate(commits[index + 1 :])
                if cleaned_title in cand.title
                and commit.commit_hash != cand.commit_hash
            }
        )
        return reasons

    def handle_commit(self, commit, i, total, commits):
        # [FUNCTION DOC] Shows one commit and asks the user to choose its category and topic.
        # Warns about potential reverts, bc-breaking and deprecation labels, and prints title,
        # body, files, labels, current category and the choices after clearing the terminal. The
        # default category is the classifier's top suggestion if a classifier exists, otherwise
        # the current category. Input is accepted by unique prefix of a category or topic; an
        # empty answer takes the default (the current topic for topics); ambiguous input re-
        # prompts. Then updates the commit and returns None.
        potential_reverts = self.potential_reverts_of(commit, commits)
        if potential_reverts:
            potential_reverts = f"!!!POTENTIAL REVERTS!!!: {potential_reverts}"
        else:
            potential_reverts = ""

        features = self.features(commit)
        if self.classifier is not None:
            from classifier import CommitClassifierInputs

            # Some commits don't have authors:
            author = features.author if features.author else "Unknown"
            files = " ".join(features.files_changed)
            classifier_input = CommitClassifierInputs(
                title=[features.title], files=[files], author=[author]
            )
            classifier_category = self.classifier.get_most_likely_category_name(
                classifier_input
            )[0]

        else:
            classifier_category = commit.category

        breaking_alarm = ""
        if "module: bc-breaking" in features.labels:
            breaking_alarm += "\n!!!!!! BC BREAKING !!!!!!"

        if "module: deprecation" in features.labels:
            breaking_alarm += "\n!!!!!! DEPRECATION !!!!!!"

        os.system("clear")
        view = textwrap.dedent(
            f"""\
[{i}/{total}]
================================================================================
{features.title}

{potential_reverts} {breaking_alarm}

{features.body}

Files changed: {features.files_changed}

Labels: {features.labels}

Current category: {commit.category}

Select from: {", ".join(common.categories)}

        """
        )
        print(view)
        cat_choice = None
        while cat_choice is None:
            print("Enter category: ")
            value = input(f"{classifier_category} ").strip()
            if len(value) == 0:
                # The user just pressed enter and likes the default value
                cat_choice = classifier_category
                continue
            choices = [cat for cat in common.categories if cat.startswith(value)]
            if len(choices) != 1:
                print(f"Possible matches: {choices}, try again")
                continue
            cat_choice = choices[0]
        print(f"\nSelected: {cat_choice}")
        print(f"\nCurrent topic: {commit.topic}")
        print(f"""Select from: {", ".join(topics)}""")
        topic_choice = None
        while topic_choice is None:
            value = input("topic> ").strip()
            if len(value) == 0:
                topic_choice = commit.topic
                continue
            choices = [cat for cat in topics if cat.startswith(value)]
            if len(choices) != 1:
                print(f"Possible matches: {choices}, try again")
                continue
            topic_choice = choices[0]
        print(f"\nSelected: {topic_choice}")
        self.update_commit(commit, cat_choice, topic_choice)
        return None

    def update_commit(self, commit, category, topic):
        # [FUNCTION DOC] Sets the commit's category and topic and saves the whole list file.
        # Asserts that both values are known.
        assert category in common.categories
        assert topic in topics
        commit.category = category
        commit.topic = topic
        self.commits.write_result()


def main():
    # [FUNCTION DOC] Entry point: parses the category filter, file path and classifier flag,
    # builds a Categorizer and runs it.
    parser = argparse.ArgumentParser(description="Tool to help categorize commits")
    parser.add_argument(
        "--category",
        type=str,
        default="Uncategorized",
        help='Which category to filter by. "Uncategorized", None, or a category name',
    )
    parser.add_argument(
        "--file",
        help="The location of the commits CSV",
        default="results/commitlist.csv",
    )
    parser.add_argument(
        "--use_classifier",
        action="store_true",
        help="Whether or not to use a classifier to aid in categorization.",
    )

    args = parser.parse_args()
    categorizer = Categorizer(args.file, args.category, args.use_classifier)
    categorizer.categorize()


if __name__ == "__main__":
    main()
