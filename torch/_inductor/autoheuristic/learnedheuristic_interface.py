# [MODULE DOC] Interface classes for learned heuristics: LearnedHeuristic (base hooks), LearnedHeuristicRegression (picks the choice with the best
# predicted feedback when clearly ahead) and LearnedHeuristicDecision (decision-tree style with ranked index probabilities). No import-time side effects.
import operator

from torch._inductor.autoheuristic.autoheuristic_utils import (
    AHContext,
    AHMetadata,
    Choice,
)


class LearnedHeuristic:
    # [CLASS DOC] Base class (interface) for learned heuristics with permissive defaults: always applicable, no decision, confidence threshold 1.0, empty name, no ranking.
    # Generated subclasses override these hooks.
    """
    LearnedHeuristic is a base class for all learned heuristics.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] No-op constructor.
        pass

    def check_precondition(
        self,
        metadata: AHMetadata,
        context: AHContext,
    ) -> bool:
        # [FUNCTION DOC] Default: always returns True (heuristic applies). Subclasses check metadata such as name and GPU.
        return True

    def get_decision(self, context: AHContext, choices: list[Choice]) -> Choice | None:
        # [FUNCTION DOC] Default: returns None (no decision). Subclasses return a choice from choices or None when unsure.
        return None

    def get_confidence_threshold(self) -> float:
        # [FUNCTION DOC] Default: returns 1.0 (the confidence threshold a decision must exceed).
        return 1.0

    def get_name(self) -> str:
        # [FUNCTION DOC] Default: returns an empty string; subclasses return the optimization name they serve.
        return ""

    def get_decisions_ranked(self, context: AHContext) -> list[str] | None:
        # [FUNCTION DOC] Default: returns None (no ranking available). Overridden by decision-tree heuristics to return ranked choices.
        return None


class LearnedHeuristicRegression(LearnedHeuristic):
    # [CLASS DOC] Base for heuristics that predict a numeric feedback per choice; a decision is made if the best prediction sufficiently exceeds the runner-up.
    def get_feedback(self, context: AHContext, choice: Choice) -> float:
        # [FUNCTION DOC] Default: returns a constant predicted feedback of 1.0. Subclasses return a model prediction for the choice.
        return 1.0

    def get_decision(self, context: AHContext, choices: list[Choice]) -> Choice | None:
        # [FUNCTION DOC] Predicts feedback for every choice, sorts ascending, and returns the choice with the highest prediction if the ratio of the highest to second highest
        # prediction exceeds the confidence threshold; otherwise None. Requires at least two choices (raises IndexError otherwise) and nonzero second prediction (ZeroDivisionError).
        # KNOWN ISSUE: ratio comparison assumes positive feedback values where higher is better.
        choice2feedback = {}
        for choice in choices:
            predicted_feedback = self.get_feedback(context, choice)
            choice2feedback[choice] = predicted_feedback
        sorted_choices_feedback = sorted(
            choice2feedback.items(), key=operator.itemgetter(1)
        )
        highest_feedback = sorted_choices_feedback[-1][1]
        second_highest_feedback = sorted_choices_feedback[-2][1]
        if highest_feedback / second_highest_feedback > self.get_confidence_threshold():
            return sorted_choices_feedback[-1][0]
        # We are not sure which choice is the best one
        return None


class LearnedHeuristicDecision(LearnedHeuristic):
    # [CLASS DOC] Base for decision-tree heuristics whose leaves give (probability, choice index) pairs; subclasses implement get_choice (index to choice) and get_best_choices.
    def get_choice(self, idx: int) -> str | None:
        # [FUNCTION DOC] Default: returns None. Subclasses map an index to a choice string.
        return None

    def get_decision(self, context: AHContext, choices: list[Choice]) -> Choice | None:
        # [FUNCTION DOC] Gets ranked (probability, index) pairs; returns None if empty or the top probability does not exceed the confidence threshold; else returns the choice for the top index.
        # The choices argument is not used.
        best_choices = self.get_best_choices(context)
        if not best_choices:
            return None
        (best_choice_proba, best_choice_idx) = best_choices[0]
        if best_choice_proba <= self.get_confidence_threshold():
            return None
        return self.get_choice(best_choice_idx)

    def get_decisions_ranked(self, context: AHContext) -> list[str] | None:
        # [FUNCTION DOC] Returns the choice strings for all ranked indices from get_best_choices, dropping indices that map to None; returns None if there is no ranking.
        feedback_idx_list = self.get_best_choices(context)
        if feedback_idx_list is None:
            return None
        choices = [
            self.get_choice(feedback_idx[1]) for feedback_idx in feedback_idx_list
        ]
        choices = [choice for choice in choices if choice is not None]
        return choices

    def get_best_choices(self, context: AHContext) -> list[tuple[float, int]] | None:
        # [FUNCTION DOC] Default: returns an empty list. Subclasses return (probability, choice index) pairs, best first, or None.
        return []
