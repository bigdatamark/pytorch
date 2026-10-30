# [MODULE DOC] Getter functions for the audio and text models of the functional autograd
# benchmark (wav2letter, deepspeech, transformer, multi-head attention). Each getter takes a
# device, builds a model with random data, strips its parameters with extract_weights, and
# returns a pair of a pure loss function of the parameters and the parameter tuple. Import-time
# side effect: records whether functorch is installed in a module flag; imports the sibling
# torchaudio_models module.
import torchaudio_models as models
from utils import check_for_functorch, extract_weights, GetterReturnType, load_weights

import torch
from torch import nn, Tensor


has_functorch = check_for_functorch()


def get_wav2letter(device: torch.device) -> GetterReturnType:
    # [FUNCTION DOC] Build a Wav2Letter model (28 classes) with batch 10, random 700 frame inputs
    # and random integer labels of length 3, plus an NLL loss. Returns (forward, params): forward
    # takes new parameter tensors, loads them into the model, runs it and returns the scalar
    # loss.
    N = 10
    input_frames = 700
    vocab_size = 28
    model = models.Wav2Letter(num_classes=vocab_size)
    criterion = torch.nn.NLLLoss()
    model.to(device)
    params, names = extract_weights(model)

    inputs = torch.rand([N, 1, input_frames], device=device)
    labels = torch.rand(N, 3, device=device).mul(vocab_size).long()

    def forward(*new_params: Tensor) -> Tensor:
        # [FUNCTION DOC] HELPER: Load the given parameters into the model, run it on the fixed
        # inputs and return the NLL loss against the labels.
        load_weights(model, names, new_params)
        out = model(inputs)

        loss = criterion(out, labels)
        return loss

    return forward, params


def get_deepspeech(device: torch.device) -> GetterReturnType:
    # [FUNCTION DOC] Build a DeepSpeech model (five bidirectional LSTM layers, hidden size 1024)
    # with a hamming window audio configuration, random spectrogram inputs, random valid length
    # sizes between 80 and 90 percent of the sequence length and random targets. If functorch is
    # available, batch norm modules are replaced so they work functionally. Uses CTC loss.
    # Returns (forward, params).
    sample_rate = 16000
    window_size = 0.02
    window = "hamming"
    audio_conf = dict(
        sample_rate=sample_rate, window_size=window_size, window=window, noise_dir=None
    )

    N = 10
    num_classes = 10
    spectrogram_size = 161
    # Commented are the original sizes in the code
    seq_length = 500  # 1343
    target_length = 10  # 50
    labels = torch.rand(num_classes, device=device)
    inputs = torch.rand(N, 1, spectrogram_size, seq_length, device=device)
    # Sequence length for each input
    inputs_sizes = (
        torch.rand(N, device=device).mul(seq_length * 0.1).add(seq_length * 0.8)
    )
    targets = torch.rand(N, target_length, device=device)
    targets_sizes = torch.full((N,), target_length, dtype=torch.int, device=device)

    model = models.DeepSpeech(
        rnn_type=nn.LSTM,
        labels=labels,
        rnn_hidden_size=1024,
        nb_layers=5,
        audio_conf=audio_conf,
        bidirectional=True,
    )

    if has_functorch:
        from functorch.experimental import replace_all_batch_norm_modules_

        replace_all_batch_norm_modules_(model)

    model = model.to(device)
    criterion = nn.CTCLoss()
    params, names = extract_weights(model)

    def forward(*new_params: Tensor) -> Tensor:
        # [FUNCTION DOC] HELPER: Load parameters, run the model on inputs and sizes, swap the
        # first two dimensions of the output for CTC loss, and return the CTC loss against
        # targets and sizes.
        load_weights(model, names, new_params)
        out, out_sizes = model(inputs, inputs_sizes)
        out = out.transpose(0, 1)  # For ctc loss

        loss = criterion(out, targets, out_sizes, targets_sizes)
        return loss

    return forward, params


def get_transformer(device: torch.device) -> GetterReturnType:
    # [FUNCTION DOC] Build a small TransformerModel (vocabulary 50, embedding 720, 12 heads, 2
    # layers) with batch 64 and sequence length 128. Random integer data is split into inputs and
    # next-token targets. The model is put in eval mode when functorch is present, to disable
    # dropout for consistency checks. Returns (forward, params).
    # For most SOTA research, you would like to have embed to 720, nhead to 12, bsz to 64, tgt_len/src_len to 128.
    N = 64
    seq_length = 128
    ntoken = 50
    model = models.TransformerModel(
        ntoken=ntoken, ninp=720, nhead=12, nhid=2048, nlayers=2
    )
    model.to(device)

    if has_functorch:
        # disable dropout for consistency checking
        model.eval()

    criterion = nn.NLLLoss()
    params, names = extract_weights(model)

    data = torch.rand(N, seq_length + 1, device=device).mul(ntoken).long()
    inputs = data.narrow(1, 0, seq_length)
    targets = data.narrow(1, 1, seq_length)

    def forward(*new_params: Tensor) -> Tensor:
        # [FUNCTION DOC] HELPER: Load parameters, run the model and return the NLL loss between
        # the flattened outputs and flattened targets.
        load_weights(model, names, new_params)
        out = model(inputs)

        loss = criterion(
            out.reshape(N * seq_length, ntoken), targets.reshape(N * seq_length)
        )
        return loss

    return forward, params


def get_multiheadattn(device: torch.device) -> GetterReturnType:
    # [FUNCTION DOC] Build a multi-head attention container (from the text library tests) with
    # embed size 10, 5 heads, target length 6, source length 10 and batch 64, with random query,
    # key and value, a random boolean attention mask and repeated key and value biases. Returns
    # (forward, params).
    # From https://github.com/pytorch/text/blob/master/test/data/test_modules.py#L10
    embed_dim, nhead, tgt_len, src_len, bsz = 10, 5, 6, 10, 64
    # Build torchtext MultiheadAttention module
    in_proj = models.InProjContainer(
        torch.nn.Linear(embed_dim, embed_dim, bias=False),
        torch.nn.Linear(embed_dim, embed_dim, bias=False),
        torch.nn.Linear(embed_dim, embed_dim, bias=False),
    )

    model = models.MultiheadAttentionContainer(
        nhead,
        in_proj,
        models.ScaledDotProduct(),
        torch.nn.Linear(embed_dim, embed_dim, bias=False),
    )
    model.to(device)
    params, names = extract_weights(model)

    query = torch.rand((tgt_len, bsz, embed_dim), device=device)
    key = value = torch.rand((src_len, bsz, embed_dim), device=device)
    attn_mask_2D = torch.randint(0, 2, (tgt_len, src_len), device=device).to(torch.bool)
    bias_k = bias_v = torch.rand((1, 1, embed_dim), device=device)

    attn_mask = torch.stack([attn_mask_2D] * (bsz * nhead))
    bias_k = bias_k.repeat(1, bsz, 1).reshape(1, bsz * nhead, -1)
    bias_v = bias_v.repeat(1, bsz, 1).reshape(1, bsz * nhead, -1)

    def forward(*new_params: Tensor) -> Tensor:
        # [FUNCTION DOC] HELPER: Load parameters, run attention and return the sum of the
        # attention output and the sum of the attention weights (no specific loss, just a
        # backprop target for both).
        load_weights(model, names, new_params)
        mha_output, attn_weights = model(
            query, key, value, attn_mask=attn_mask, bias_k=bias_k, bias_v=bias_v
        )

        # Don't test any specific loss, just backprop ones for both outputs
        loss = mha_output.sum() + attn_weights.sum()

        return loss

    return forward, params
