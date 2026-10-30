# [MODULE DOC] End-to-end text generation benchmarks for gpt-fast style models (Llama 2 7B in
# bfloat16 and int8, and a reduced Mixtral 8x7B in int8). Provides token sampling, prefill and
# decode loops, random-weight model loading, a model size estimate that counts only activated
# parameters, a driver that measures tokens per second, memory bandwidth and compilation time,
# and three registered experiments returning Experiment records. Relies on modules model,
# quantize and mixtral_moe_quantize that are expected to be importable from the gpt-fast
# checkout, plus the local common and mixtral_moe_model. Import-time side effects: enables
# coordinate descent tuning, unique kernel names, the FX graph cache and disables the indirect
# indexing assertion in the global inductor config, and defines a module flag recording whether
# compilation has happened.
import dataclasses
import itertools
import platform
import time

from common import Experiment, register_experiment
from mixtral_moe_model import ConditionalFeedForward, Transformer as MixtralMoE
from mixtral_moe_quantize import (
    ConditionalFeedForwardInt8,
    WeightOnlyInt8QuantHandler as MixtralMoEWeightOnlyInt8QuantHandler,
)
from model import Transformer as LLaMA
from quantize import WeightOnlyInt8QuantHandler as LLaMAWeightOnlyInt8QuantHandler

import torch
import torch._inductor.config


torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future
torch._inductor.config.assert_indirect_indexing = False

compiled = False


@dataclasses.dataclass
class GPTModelConfig:
    # [CLASS DOC] Dataclass describing one model benchmark target: the model name, the model
    # class to build, a mode string (bfloat16 or int8), the quantizer handler class, target
    # tokens per second, target memory bandwidth in GB/s, target compilation time in seconds,
    # and an optional batch size (default none). It is plain data without methods.
    name: str
    module: type
    mode: str | None
    quantizer: type
    token_per_sec: float
    memory_bandwidth: float
    compilation_time: float
    batch_size: int | None = None


def device_sync(device):
    # [FUNCTION DOC] Block until queued work on the device finishes. For a device string
    # containing cuda the CUDA stream is synchronized; for a cpu device nothing happens; any
    # other device prints a message saying it is unsupported and does nothing.
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif "cpu" in device:
        pass
    else:
        print(f"device={device} is not yet supported")


def get_arch_name() -> str:
    # [FUNCTION DOC] Return a string naming the hardware: the CUDA device name if CUDA is
    # available, otherwise the machine architecture string of the host platform (such as x86_64
    # or arm64). Used as the arch field of experiment records.
    if torch.cuda.is_available():
        return torch.cuda.get_device_name()
    else:
        # This returns x86_64 or arm64 (for aarch64)
        return platform.machine()


def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    # [FUNCTION DOC] Draw one sample index per row of a probability tensor along the last
    # dimension without forcing a host-device synchronization. It divides the probabilities by
    # independent exponential noise of rate one and takes the arg-max, which is equivalent to
    # multinomial sampling. Returns an int32 tensor with the last dimension kept as size one.
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(logits, temperature: float = 1.0, top_k: int | None = None):
    # [FUNCTION DOC] Convert logits to a probability distribution. Logits are divided by the
    # temperature (floored at one hundred-thousandth to avoid division by zero). If top_k is
    # given, every logit smaller than the k-th largest value (k capped at the vocabulary size)
    # is replaced with negative infinity. Returns the softmax over the last dimension.
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(logits, temperature: float = 1.0, top_k: int | None = None):
    # [FUNCTION DOC] Sample the next token from model output. Uses the logits of the first batch
    # element at the last position, converts them to probabilities with the temperature and
    # optional top-k filter, and samples one index without synchronization. Returns a pair of
    # the sampled index tensor and the probability tensor.
    probs = logits_to_probs(logits[0, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def prefill(
    model: torch.nn.Module, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs
) -> torch.Tensor:
    # [FUNCTION DOC] Run the prompt through the model in one pass (the position tensor has batch
    # and sequence dimensions) and return only the sampled next token; the probabilities are
    # discarded. Extra keyword arguments are forwarded to the sampling step. Updates the model
    # key-value cache as a side effect.
    # input_pos: [B, S]
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)[0]


def decode_one_token(
    model: torch.nn.Module, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] Run a single decoding step: feed one token with its position through the
    # model and sample the next token. Raises an AssertionError if the position tensor does not
    # have size one in the last dimension. Returns a pair of the next token and its probability
    # tensor. Updates the key-value cache as a side effect.
    # input_pos: [B, 1]
    if input_pos.shape[-1] != 1:
        raise AssertionError(
            f"input_pos.shape[-1] must be 1, but got {input_pos.shape[-1]}"
        )
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)


def decode_n_tokens(
    model: torch.nn.Module,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    **sampling_kwargs,
):
    # [FUNCTION DOC] Generate a number of tokens one at a time. Each iteration runs inside a
    # context forcing the math attention backend (so Inductor can generate code for attention),
    # decodes one token, increments the position tensor in place, stores clones of the sampled
    # token and its probabilities, and feeds the token back reshaped to one row. Returns two
    # lists: the sampled tokens and their probability tensors. Mutates input_pos in place.
    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        with torch.nn.attention.sdpa_kernel(
            torch.nn.attention.SDPBackend.MATH
        ):  # Actually better for Inductor to codegen attention here
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, **sampling_kwargs
            )
            input_pos += 1
            new_tokens.append(next_token.clone())
            new_probs.append(next_prob.clone())
            cur_token = next_token.view(1, -1)

    return new_tokens, new_probs


@torch.no_grad()
def generate(
    model: torch.nn.Module, prompt: torch.Tensor, max_new_tokens: int, **sampling_kwargs
) -> torch.Tensor:
    # [FUNCTION DOC] Generate text from a one-dimensional prompt of token ids under no gradient
    # mode. Sets up the key-value caches on the model for batch size one and a sequence length
    # capped at the model block size, allocates an output sequence of prompt plus new token
    # length, copies the prompt into it, prefills to get the first new token, then decodes the
    # remaining new tokens one at a time. Returns the full token sequence (prompt followed by
    # the generated tokens). Uses the module level prefill and decode functions, which may have
    # been replaced by compiled versions.
    device, dtype = prompt.device, prompt.dtype
    T = prompt.size(0)
    T_new = T + max_new_tokens
    max_seq_length = min(T_new, model.config.block_size)

    with torch.device(device):
        model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)

    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(T_new, dtype=dtype, device=device)
    empty[:T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)

    next_token = prefill(model, prompt.view(1, -1), input_pos, **sampling_kwargs)
    seq[T] = next_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)

    generated_tokens, _ = decode_n_tokens(
        model, next_token.view(1, -1), input_pos, max_new_tokens - 1, **sampling_kwargs
    )
    seq[T + 1 :] = torch.cat(generated_tokens)
    return seq


def _load_model(x: GPTModelConfig, device="cuda", precision=torch.bfloat16):
    # [FUNCTION DOC] HELPER: Build a benchmark model with random weights. The model is created
    # from the config name on the meta device, cast to the precision, converted by the quantizer
    # when the mode is int8 (printing a message), and then every state dict entry is replaced by
    # a parameter filled with standard normal values on the target device and cast to the
    # original dtype, keeping its requires-grad flag. The state is loaded with assignment and
    # the model is returned in eval mode.
    with torch.device("meta"):
        model = x.module.from_name(x.name)
    model = model.to(dtype=precision)

    if x.mode == "int8":
        print("Using int8 weight-only quantization!")
        model = x.quantizer(model).convert_for_runtime()

    state_dict = model.state_dict()
    for k, v in state_dict.items():
        state_dict[k] = torch.nn.Parameter(
            torch.randn(v.shape, device=device).to(dtype=v.dtype),
            requires_grad=v.requires_grad,
        )
    model.load_state_dict(state_dict, assign=True)
    return model.eval()


# Only count activated parameters and buffers.
def _get_model_size(model):
    # [FUNCTION DOC] HELPER: Estimate the bytes of weights read per generated token. Sums
    # parameter and buffer sizes of all top-level children except embeddings. For
    # mixture-of-experts models (config has an expert count) it subtracts the share of the
    # expert parameters belonging to the inactive experts for each expert feed-forward submodule
    # (plain or int8 variant). Returns a number of bytes.
    # KNOWN ISSUE: in the expert subtraction the buffer term uses the loop variable left over
    # from the first loop, namely the last top-level child, instead of the submodule; this is
    # wrong or at best coincidental for models with buffers.
    model_size = 0
    for name, child in model.named_children():
        if not isinstance(child, torch.nn.Embedding):
            model_size += sum(
                p.numel() * p.dtype.itemsize
                for p in itertools.chain(child.parameters(), child.buffers())
            )

    # Remove the inactivated experts from the model size if this is mixture of experts
    # architecture, since only activated experts are loaded.
    if hasattr(model.config, "num_experts"):
        config = model.config
        for submodule in model.modules():
            if isinstance(
                submodule, (ConditionalFeedForward, ConditionalFeedForwardInt8)
            ):
                model_size -= (
                    sum(
                        p.numel() * p.dtype.itemsize
                        for p in itertools.chain(
                            submodule.parameters(), child.buffers()
                        )
                    )
                    * (config.num_experts - config.num_activated_experts)
                    / config.num_experts
                )

    return model_size


def run_experiment(
    x: GPTModelConfig,
    num_samples: int = 5,
    max_new_tokens: int = 200,
    top_k: int = 200,
    temperature: float = 0.8,
    device: str = "cuda",
) -> None:
    # [FUNCTION DOC] Benchmark one model configuration. Loads the model on the device, builds a
    # fixed six token prompt, seeds the random generator, estimates the model size, and (once
    # per process, guarded by a module flag) replaces the module level decode and prefill
    # functions by compiled versions (decode in reduce-overhead mode, both with full graph). It
    # then runs one warm-up generation, which is timed as the compilation time, followed by the
    # given number of timed generations; for each it records tokens per second and the memory
    # bandwidth implied by the model size. Prints load time, compilation time, averages and the
    # maximum reserved CUDA memory. Returns a tuple of mean tokens per second, mean bandwidth in
    # GB/s and compilation time in seconds. Needs a CUDA device for the final memory report.
    # KNOWN ISSUE: the return annotation claims no value, but a tuple is returned. Compilation
    # time is only measured on the first call in the process; later calls report a short warm-up
    # time or the compiled state of the earlier run.
    print(f"Loading model {x.name}")
    t0 = time.time()
    model = _load_model(x, device=device)
    device_sync(device=device)  # MKG
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    prompt = torch.tensor(
        [1, 15043, 29892, 590, 1024, 338], device=device, dtype=torch.int32
    )
    prompt_length = prompt.size(0)

    torch.manual_seed(1234)
    model_size = _get_model_size(model)

    aggregate_metrics = {"tokens_per_sec": [], "memory_bandwidth": []}
    start = -1
    compilation_time = None

    global decode_one_token, prefill, compiled
    if not compiled:
        compiled = True
        decode_one_token = torch.compile(
            decode_one_token, mode="reduce-overhead", fullgraph=True
        )
        prefill = torch.compile(prefill, fullgraph=True)

    for i in range(start, num_samples):
        device_sync(device=device)  # MKG

        torch.compiler.cudagraph_mark_step_begin()
        t0 = time.perf_counter()
        y = generate(
            model, prompt, max_new_tokens, temperature=temperature, top_k=top_k
        )

        if i == -1:
            compilation_time = time.perf_counter() - t0
            print(f"Compilation time: {compilation_time:.2f} seconds")
            continue

        device_sync(device=device)  # MKG
        t = time.perf_counter() - t0
        tokens_generated = y.size(0) - prompt_length
        tokens_sec = tokens_generated / t
        aggregate_metrics["tokens_per_sec"].append(tokens_sec)
        aggregate_metrics["memory_bandwidth"].append(model_size * tokens_sec / 1e9)

    token_per_sec = torch.mean(torch.tensor(aggregate_metrics["tokens_per_sec"])).item()
    memory_bandwidth = torch.mean(
        torch.tensor(aggregate_metrics["memory_bandwidth"])
    ).item()
    print(f"Average tokens/sec: {token_per_sec:.2f} tokens/sec")
    print(f"Average bandwidth achieved: {memory_bandwidth:.02f} GB/s")
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
    return token_per_sec, memory_bandwidth, compilation_time


# token_per_sec and memory_bandwidth target numbers are for A100-40GB, which are different from the typical A100-80GB.
@register_experiment(name="llama2_7b_bf16")
def run_llama2_7b_bf16(device: str = "cuda"):
    # [FUNCTION DOC] Registered experiment llama2_7b_bf16. Benchmarks Llama 2 7B chat in
    # bfloat16 mode (it still passes the int8 quantizer class, which is only used for int8 mode)
    # with targets 94 tokens/s, 1253 GB/s and 133 s compile time (targets are for an A100-40GB).
    # Runs the benchmark driver on the device and returns a list of three Experiment records,
    # for token rate, memory bandwidth and compilation time, each with the target, the measured
    # value formatted with two decimals, the mode, the device, the arch name and the model flag
    # set to true.
    model = GPTModelConfig(
        "Llama-2-7b-chat-hf",
        LLaMA,
        "bfloat16",
        LLaMAWeightOnlyInt8QuantHandler,
        94,
        1253,
        133,
    )
    token_per_sec, memory_bandwidth, compilation_time = run_experiment(
        model, device=device
    )
    return [
        Experiment(
            model.name,
            "token_per_sec",
            model.token_per_sec,
            f"{token_per_sec:.02f}",
            model.mode,
            device,
            get_arch_name(),
            True,
        ),
        Experiment(
            model.name,
            "memory_bandwidth(GB/s)",
            model.memory_bandwidth,
            f"{memory_bandwidth:.02f}",
            model.mode,
            device,
            get_arch_name(),
            True,
        ),
        Experiment(
            model.name,
            "compilation_time(s)",
            model.compilation_time,
            f"{compilation_time:.02f}",
            model.mode,
            device,
            get_arch_name(),
            True,
        ),
    ]


# token_per_sec and memory_bandwidth target numbers are for A100-40GB, which are different from the typical A100-80GB.
@register_experiment(name="llama2_7b_int8")
def run_llama2_7b_int8(device: str = "cuda"):
    # [FUNCTION DOC] Registered experiment llama2_7b_int8. Benchmarks Llama 2 7B chat with int8
    # weight-only quantization, targets 144 tokens/s, 957 GB/s and 136 s compile time (targets
    # are for an A100-40GB). Runs the benchmark driver on the device and returns a list of three
    # Experiment records, for token rate, memory bandwidth and compilation time, each with the
    # target, the measured value formatted with two decimals, the mode, the device, the arch
    # name and the model flag set to true.
    model = GPTModelConfig(
        "Llama-2-7b-chat-hf",
        LLaMA,
        "int8",
        LLaMAWeightOnlyInt8QuantHandler,
        144,
        957,
        136,
    )
    token_per_sec, memory_bandwidth, compilation_time = run_experiment(
        model, device=device
    )
    return [
        Experiment(
            model.name,
            "token_per_sec",
            model.token_per_sec,
            f"{token_per_sec:.02f}",
            model.mode,
            device,
            get_arch_name(),
            True,
        ),
        Experiment(
            model.name,
            "memory_bandwidth(GB/s)",
            model.memory_bandwidth,
            f"{memory_bandwidth:.02f}",
            model.mode,
            device,
            get_arch_name(),
            True,
        ),
        Experiment(
            model.name,
            "compilation_time(s)",
            model.compilation_time,
            f"{compilation_time:.02f}",
            model.mode,
            device,
            get_arch_name(),
            True,
        ),
    ]


# token_per_sec and memory_bandwidth target numbers are for A100-40GB, which are different from the typical A100-80GB.
@register_experiment(name="mixtral_8x7b_int8")
def run_mixtral_8x7b_int8(device: str = "cuda"):
    # [FUNCTION DOC] Registered experiment mixtral_8x7b_int8. Benchmarks Mixtral 8x7B (reduced
    # layer count to fit CI memory, according to the in-code note) with int8 weight-only
    # quantization, targets 175 tokens/s, 1130 GB/s and 133 s compile time (targets are for an
    # A100-40GB). Runs the benchmark driver on the device and returns a list of three Experiment
    # records, for token rate, memory bandwidth and compilation time, each with the target, the
    # measured value formatted with two decimals, the mode, the device, the arch name and the
    # model flag set to true.
    # We reduced the original number of layers from 32 to 16 to adapt CI memory limitation.
    model = GPTModelConfig(
        "Mixtral-8x7B-v0.1",
        MixtralMoE,
        "int8",
        MixtralMoEWeightOnlyInt8QuantHandler,
        175,
        1130,
        133,
    )
    token_per_sec, memory_bandwidth, compilation_time = run_experiment(
        model, device=device
    )
    return [
        Experiment(
            model.name,
            "token_per_sec",
            model.token_per_sec,
            f"{token_per_sec:.02f}",
            model.mode,
            device,
            get_arch_name(),
            True,
        ),
        Experiment(
            model.name,
            "memory_bandwidth(GB/s)",
            model.memory_bandwidth,
            f"{memory_bandwidth:.02f}",
            model.mode,
            device,
            get_arch_name(),
            True,
        ),
        Experiment(
            model.name,
            "compilation_time(s)",
            model.compilation_time,
            f"{compilation_time:.02f}",
            model.mode,
            device,
            get_arch_name(),
            True,
        ),
    ]
