# [MODULE DOC] AutoHeuristic data-collection and decision front end for inductor. Defines AutoHeuristic (collect feedback per choice into a
# per-GPU log file, or consult learned heuristics to pick a choice), AutoHeuristicSelectAlgorithm (variant for kernel choice selection hooked
# into select_algorithm timings), LocalFeedback and the InconsistentMetadata error. No import-time side effects.
import json
import os
from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from torch._inductor.autoheuristic.autoheuristic_utils import (
    AHContext,
    AHMetadata,
    AHOperation,
    Choice,
    CHOICE_COL,
    Feedback,
    FEEDBACK_COL,
    get_metadata_str_from_log,
)
from torch._inductor.autoheuristic.learned_heuristic_controller import (
    LearnedHeuristicController,
)
from torch._inductor.ir import ChoiceCaller
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._inductor.utils import get_gpu_shared_memory


class LocalFeedback:
    # [CLASS DOC] Wraps a callable that maps a choice to measured feedback so AutoHeuristic can run it immediately for each choice. Holds only feedback_fn.
    """
    To be able to collect data for a choice, a function providing feedback given a choice has to be provided.
    LocalFeedback can be used when AutoHeuristic should immediately run the function to collect feedback for each choice
    (see pad_mm.py, where the autotuning happens locally, for an example).
    """

    def __init__(self, feedback_fn: Callable[[Choice], Feedback]) -> None:
        # [FUNCTION DOC] Stores the feedback callable.
        self.feedback_fn = feedback_fn

    def __call__(self, choice: Choice) -> Feedback:
        # [FUNCTION DOC] Invokes the stored feedback callable for the choice and returns its feedback value.
        return self.feedback_fn(choice)


class InconsistentMetadata(Exception):
    # [CLASS DOC] Raised when the metadata line in an existing log file differs from the metadata AutoHeuristic would write for the current run.
    """
    Exception that is thrown when AutoHeuristic tries to log data to a file where the metadata stored in the file does
    not match the metadata it would store if the file didn't exist.
    """


class AutoHeuristic:
    # [CLASS DOC] Collects (context, choice, feedback) rows to a text log and/or queries learned heuristics. State: fallback callable, choices, optional LocalFeedback,
    # context, name, collected_feedback mapping, optional context augmentations, metadata (GPU shared memory, capability, choices, name), optional
    # precondition and log_path (only set when the precondition holds). Key methods: get_choice, get_top_k_choices, save_data.
    """
    AutoHeuristic is a framework that allows one to collect data, learn a heuristic (i.e. a regression tree) and
    generate the heuristic to code. This class allows one to collect data. The collected data can then be used to train
    a heuristic (see torchgen/autoheuristic/).
    """

    collected_feedback: dict[Choice, Feedback]

    def __init__(
        self,
        fallback: Callable[[], Choice],
        choices: list[Choice],
        feedback: LocalFeedback | None,
        context: AHContext,
        name: str,
        augment_context: list[AHOperation] | None = None,
        precondition: Callable[[AHMetadata, AHContext], bool] | None = None,
    ) -> None:
        # [FUNCTION DOC] Stores the arguments, builds metadata from GPU shared memory and device capability (0,0 without CUDA) and, if the precondition holds,
        # determines the log path (default or configured). If data collection is enabled for this name and a LocalFeedback exists, runs it for every choice
        # and saves each result (file I/O).
        """
        Initializes an instance of the AutoHeuristic class.

        Args:
            fallback: A callable that returns a Choice when the heuristic is unsure which choice to make, or
            AutoHeuristic is in data collection mode.
            choices: A list of possible choices the heuristic can make.
            feedback: An instance of LocalFeedback that provides feedback for a given choice.
            context: Context to store with each choice and feedback.
            name: A string that identifies the heuristic.
            augment_context: An optional list of AHOperation instances that augment the context.
            precondition: A callable that returns a boolean indicating whether AutoHeuristic should run.
        """
        self.fallback = fallback
        self.choices = choices
        self.feedback = feedback
        self.context = context
        self.name = name
        self.collected_feedback = {}
        self.augment_context = augment_context
        self.metadata = AHMetadata(
            get_gpu_shared_memory(),
            torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0),
            self.choices,
            self.name,
        )
        self.precondition = precondition

        if not self.satisfies_precondition():
            return

        if torch._inductor.config.autoheuristic_log_path == "DEFAULT":
            self.log_path = self.get_default_log_path()
        else:
            self.log_path = torch._inductor.config.autoheuristic_log_path

        if torch._inductor.config.collect_autoheuristic(self.name):
            if self.feedback is not None:
                for choice in self.choices:
                    feedback_val = self.feedback(choice)
                    self.save_data(choice, feedback_val)

    def satisfies_precondition(self) -> bool:
        # [FUNCTION DOC] Returns True when no precondition was given or when the precondition accepts the metadata and context.
        return self.precondition is None or self.precondition(
            self.metadata, self.context
        )

    def get_choice(self) -> Choice:
        # [FUNCTION DOC] Returns the fallback choice unless the precondition holds, learned heuristics are enabled for this name in config, and the controller returns a
        # decision that is among the allowed choices (applying context augmentations first). Otherwise returns fallback(). Mutates the context when augmenting.
        """
        Returns the chosen option based on the value of autoheuristic_use.
        If self.name is one of the comma separated strings in autoheuristic_use,
        it queries a learned heuristic to make a decision. Otherwise, it returns the fallback option.
        """

        if not self.satisfies_precondition():
            return self.fallback()

        if torch._inductor.config.use_autoheuristic(self.name):
            if self.augment_context is not None:
                self.context.apply_operations(self.augment_context)
            controller = LearnedHeuristicController(
                self.metadata,
                self.context,
            )
            decision = controller.get_decision()
            if decision not in self.choices:
                # TODO(AlnisM): We might want to allow this in the future
                return self.fallback()
            if decision is not None:
                return decision
        return self.fallback()

    def get_top_k_choices(
        self, top_k: int, always_included: list[str] | None = None
    ) -> list[Choice] | None:
        # [FUNCTION DOC] Returns up to top_k ranked choices from the learned heuristic (after optional context augmentation), appending any always_included
        # choices not already present. Returns None when the precondition fails, learned heuristics are disabled for this name, or no ranking exists.
        if not self.satisfies_precondition():
            return None
        if torch._inductor.config.use_autoheuristic(self.name):
            if self.augment_context is not None:
                self.context.apply_operations(self.augment_context)
            controller = LearnedHeuristicController(
                self.metadata,
                self.context,
            )
            choices = controller.get_decisions_ranked(top_k)
            if choices is None:
                return None
            if always_included is not None:
                for choice in always_included:
                    if choice not in choices:
                        choices.append(choice)
            return choices
        return None

    def get_collected_feedback(self, choice: Choice) -> Any:
        # [FUNCTION DOC] Returns the feedback recorded for the choice during data collection or None if none.
        return self.collected_feedback.get(choice, None)

    @staticmethod
    def get_device_identifier() -> str:
        # [FUNCTION DOC] Returns the CUDA device name with spaces replaced by underscores, or a fixed string when CUDA is unavailable. Used to separate logs per GPU model.
        # a heuristic might work well for one GPU, but not for another
        # we store the collected data per GPU model and learn a heuristic per GPU model

        # TODO(AlnisM): just using the device name for now, but the same GPU model can have different names
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name().replace(" ", "_")
            return device_name
        return "non_cuda_device"

    def get_default_log_path(self) -> str:
        # [FUNCTION DOC] Returns the default log file path under the inductor cache directory, per device and heuristic name; creates the directory (side effect).
        device_name = self.get_device_identifier()
        path = f"{cache_dir()}/autoheuristic/{device_name}/"
        os.makedirs(path, exist_ok=True)
        path += f"{self.name}.txt"
        return path

    def serialize_metadata(self) -> str:
        # [FUNCTION DOC] Returns a JSON string of the metadata dict extended with the context's numerical and categorical feature name lists.
        metadata_dict = self.metadata.to_dict()
        (
            num_features,
            cat_features,
        ) = self.context.get_numerical_and_categorical_features()
        metadata_dict["numerical_features"] = num_features
        metadata_dict["categorical_features"] = cat_features
        return json.dumps(metadata_dict)

    def save_data(self, choice: Choice, feedback_val: Feedback) -> None:
        # [FUNCTION DOC] Records feedback in collected_feedback and appends a CSV row (context feature values, choice, feedback) to the log file. If the file is new, also
        # writes the metadata JSON line and a header line first. If it exists, checks its first line equals the current metadata and raises InconsistentMetadata otherwise. I/O side effect.
        self.collected_feedback[choice] = feedback_val
        log_path = self.log_path

        lines = []
        log_exists = os.path.exists(log_path)
        if log_exists:
            # if log already exists, make sure it is consistent
            metadata = self.serialize_metadata()
            existing_metadata = get_metadata_str_from_log(self.log_path)
            if existing_metadata != metadata:
                raise InconsistentMetadata(
                    "Given metadata does not match existing metadata"
                )
        else:
            lines.append(self.serialize_metadata())
            feature_header = self.context.get_feature_names_csv()
            header = feature_header + "," + CHOICE_COL + "," + FEEDBACK_COL
            lines.append(header)

        line = ""
        feature_values = self.context.get_feature_values_csv()
        line += feature_values + "," + choice + "," + str(feedback_val)
        lines.append(line)

        with open(log_path, "a") as f:
            f.write("\n".join(lines) + "\n")


class AutoHeuristicSelectAlgorithm(AutoHeuristic):
    # [CLASS DOC] AutoHeuristic specialization for choosing among ChoiceCaller objects: maps each choice's autoheuristic id to the caller (choicestr2choice),
    # keeps the input nodes, and uses a fallback that yields the string unsure when the fallback has no choice. Registers a global feedback saver when collecting.
    """
    AutoHeuristicSelectAlgorithm is a subclass of AutoHeuristic that allows one to collect data and learn a heuristic
    when one wants to use AutoHeuristic for kernel choice selection.
    """

    def __init__(
        self,
        fallback: Callable[[], ChoiceCaller | None],
        choices: list[ChoiceCaller],
        input_nodes: list[Any],
        context: AHContext,
        name: str,
        augment_context: list[AHOperation] | None = None,
        precondition: Callable[[AHMetadata, AHContext], bool] | None = None,
    ) -> None:
        # [FUNCTION DOC] Builds the id-to-caller mapping, wraps the fallback to return an id string, and initializes the base class with string choices and no local
        # feedback. If collecting is enabled for this name and the precondition holds, registers global feedback. Args must match those of the select_algorithm call.
        """
        The arguments choices, input_nodes and name have to match the ones used in the call to
        autotune_select_algorithm(), e.g. if the following call is made
        autotune_select_algorithm(name, choices, input_nodes, layout), the same name, choices and input_nodes
        have to be used here.
        """
        self.input_nodes = input_nodes
        self.choicestr2choice: dict[str, ChoiceCaller] = {}
        for choice in choices:
            self.choicestr2choice[choice.autoheuristic_id()] = choice
        choices_str = list(self.choicestr2choice.keys())

        def fallback_str() -> str:
            # [FUNCTION DOC] HELPER: Calls the fallback; returns the resulting caller's autoheuristic id, or unsure if the fallback returned None.
            fallback_choice = fallback()
            if fallback_choice is None:
                # TODO: Find a nicer way to handle this
                return "unsure"
            return fallback_choice.autoheuristic_id()

        super().__init__(
            fallback_str,
            choices_str,
            None,
            context,
            name,
            augment_context,
            precondition,
        )

        if (
            torch._inductor.config.collect_autoheuristic(self.name)
            and self.satisfies_precondition()
        ):
            self.register_global_feedback(input_nodes, choices)

    def register_global_feedback(
        self, input_nodes: list[Any], choices: list[ChoiceCaller]
    ) -> None:
        # [FUNCTION DOC] Registers a feedback saver with select_algorithm so that each time algorithms are timed with matching inputs and choices, the
        # timings are logged. Builds keys from the input nodes and choices and adds the saver as a partial; mutates select_algorithm's saver registry.
        """
        Registers a callback in select_algorithm, which is called with the timing of each choice.
        """

        from torch._inductor.select_algorithm import (
            add_feedback_saver,
            create_inputs_key,
            create_precompile_key,
        )

        def store_global_feedback(
            ah_inputs_key: str,
            ah_precompile_key: str,
            timings: dict[ChoiceCaller, float],
            name: str,
            input_nodes: list[Any],
            choices: list[ChoiceCaller],
            profiled_time: Callable[[], dict[ChoiceCaller, float]],
            precompile_times: dict[ChoiceCaller, float],
        ) -> None:
            # [FUNCTION DOC] HELPER: Callback invoked by select_algorithm with timings of all choices. Ignores the event unless the inputs key and precompile key match
            # those captured at registration; otherwise saves each choice's timing via save_data keyed by autoheuristic id. Other args are unused.
            current_inputs_key = create_inputs_key(input_nodes)
            if current_inputs_key != ah_inputs_key:
                return
            current_precompile_key = create_precompile_key(
                name, current_inputs_key, choices
            )
            if current_precompile_key != ah_precompile_key:
                return
            for choice, time in timings.items():
                self.save_data(choice.autoheuristic_id(), time)

        inputs_key = create_inputs_key(input_nodes)
        precompile_key = create_precompile_key(self.name, inputs_key, choices)
        feedback_saver = partial(store_global_feedback, inputs_key, precompile_key)
        add_feedback_saver(feedback_saver)

    def get_choice_caller(self) -> ChoiceCaller | None:
        # [FUNCTION DOC] Returns the ChoiceCaller for the chosen id, or None if the chosen id has no caller (for example unsure).
        choice = self.get_choice()
        return self.choicestr2choice.get(choice, None)

    def get_top_k_choices_caller(
        self, top_k: int, always_included: list[str] | None = None
    ) -> list[ChoiceCaller] | None:
        # [FUNCTION DOC] Returns the ChoiceCallers for the top-k ranked choice ids (plus always_included ids), or None if no ranking is available.
        choices = self.get_top_k_choices(top_k, always_included)
        if choices is None:
            return None
        return [self.choicestr2choice[choice] for choice in choices]
