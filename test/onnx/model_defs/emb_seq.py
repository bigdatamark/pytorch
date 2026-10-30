# [MODULE DOC] Two small embedding-plus-linear models used as ONNX test inputs.
import torch.nn as nn


class EmbeddingNetwork1(nn.Module):
    # [CLASS DOC] Embedding of 10 indices into dim features followed by a linear
    # layer to one output, chained in a Sequential that shares the same embedding
    # and linear submodules.
    def __init__(self, dim=5):
        # [FUNCTION DOC] Creates an embedding with 10 rows of size dim, a linear layer from
        # dim to 1, and the sequential container wrapping both.
        super().__init__()
        self.emb = nn.Embedding(10, dim)
        self.lin1 = nn.Linear(dim, 1)
        self.seq = nn.Sequential(
            self.emb,
            self.lin1,
        )

    def forward(self, input):
    # [FUNCTION DOC] Returns the sequential applied to an index tensor.
        return self.seq(input)


class EmbeddingNetwork2(nn.Module):
    # [CLASS DOC] Embedding of in_space indices into dim features, then a linear layer to
    # one output and a sigmoid, all held in a Sequential.
    def __init__(self, in_space=10, dim=3):
        # [FUNCTION DOC] Creates the embedding (in_space rows, dim columns) and a sequential
        # of the embedding, a linear layer from dim to 1, and a sigmoid.
        super().__init__()
        self.embedding = nn.Embedding(in_space, dim)
        self.seq = nn.Sequential(self.embedding, nn.Linear(dim, 1), nn.Sigmoid())

    def forward(self, indices):
    # [FUNCTION DOC] Returns the sequential applied to an index tensor.
        return self.seq(indices)
