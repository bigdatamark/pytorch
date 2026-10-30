# [MODULE DOC] Registry of HuggingFace LLM-style benchmarks used by the huggingface driver.
# Defines a Benchmark base class, a Whisper speech model benchmark and a text generation
# benchmark, and HF_LLM_MODELS mapping hub model ids to the benchmark class that builds them.
# Import-time side effects: installs transformers with pip when it cannot be imported, then
# imports the needed transformers classes.
import subprocess
import sys

import torch


def pip_install(package):
    # [FUNCTION DOC] HELPER: Install a package into the current interpreter with pip. Raises
    # CalledProcessError if the install fails.
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )
except ModuleNotFoundError:
    print("Installing HuggingFace Transformers...")
    pip_install("git+https://github.com/huggingface/transformers.git#egg=transformers")
finally:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )


class Benchmark:
    # [CLASS DOC] Interface for an LLM benchmark: subclasses provide a static method that builds
    # a model and its example inputs.
    @staticmethod
    def get_model_and_inputs(model_name, device):
        # [FUNCTION DOC] Abstract static method returning a (model, inputs) pair for a hub model
        # name and device. Raises NotImplementedError in the base class.
        raise NotImplementedError("get_model_and_inputs() not implemented")


class WhisperBenchmark(Benchmark):
    # [CLASS DOC] Benchmark for Whisper speech-to-text models. Class constants: audio sample
    # rate of 16000 Hz and a clip duration of 30 seconds.
    SAMPLE_RATE = 16000
    DURATION = 30.0  # seconds

    @staticmethod
    def get_model_and_inputs(model_name, device):
        # [FUNCTION DOC] Load the processor and conditional generation model by name onto the
        # device, clear forced decoder ids and set greedy decoding (no sampling, zero
        # temperature). Inputs: processor features for 30 seconds of low-amplitude random audio
        # moved to the device, plus a one-token decoder input holding the start token. Returns
        # (model, inputs mapping). Downloads from the hub.
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
        model.config.forced_decoder_ids = None

        model.generation_config.do_sample = False
        model.generation_config.temperature = 0.0

        num_samples = int(WhisperBenchmark.DURATION * WhisperBenchmark.SAMPLE_RATE)
        audio = torch.randn(num_samples) * 0.1
        inputs = dict(
            processor(
                audio, sampling_rate=WhisperBenchmark.SAMPLE_RATE, return_tensors="pt"
            )
        )
        inputs["input_features"] = inputs["input_features"].to(device)

        decoder_start_token = model.config.decoder_start_token_id
        inputs["decoder_input_ids"] = torch.tensor(
            [[decoder_start_token]], device=device
        )

        return model, inputs


class TextGenerationBenchmark(Benchmark):
    # [CLASS DOC] Benchmark for causal language models. Class constants: input length 1000 tokens
    # and output length 2000 new tokens.
    INPUT_LENGTH = 1000
    OUTPUT_LENGTH = 2000

    @staticmethod
    def get_model_and_inputs(model_name, device):
        # [FUNCTION DOC] Load tokenizer and causal LM by name with device placement, in eval mode,
        # with greedy decoding, static KV cache, 2000 max new tokens and the end token as pad.
        # Inputs: one row of 1000 random token ids on the device. Returns (model, inputs mapping
        # with input ids). Downloads from the hub.
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
        model.eval()

        model.generation_config.do_sample = False
        model.generation_config.use_cache = True
        model.generation_config.cache_implementation = "static"
        model.generation_config.max_new_tokens = TextGenerationBenchmark.OUTPUT_LENGTH
        model.generation_config.pad_token_id = tokenizer.eos_token_id
        model.generation_config.temperature = 0.0

        vocab_size = tokenizer.vocab_size
        input_ids = torch.randint(
            low=0,
            high=vocab_size,
            size=(1, TextGenerationBenchmark.INPUT_LENGTH),
            device=device,
            dtype=torch.long,
        )
        example_inputs = {"input_ids": input_ids}

        return model, example_inputs


HF_LLM_MODELS: dict[str, Benchmark] = {
    "meta-llama/Llama-3.2-1B": TextGenerationBenchmark,
    "google/gemma-2-2b": TextGenerationBenchmark,
    "google/gemma-3-4b-it": TextGenerationBenchmark,
    "openai/whisper-tiny": WhisperBenchmark,
    "Qwen/Qwen3-0.6B": TextGenerationBenchmark,
    "mistralai/Mistral-7B-Instruct-v0.3": TextGenerationBenchmark,
    "openai/gpt-oss-20b": TextGenerationBenchmark,
}
