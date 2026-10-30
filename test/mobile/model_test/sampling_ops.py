# [MODULE DOC] Mobile model test fixture covering the random-sampling operators listed in the
# torch random sampling docs. Defines one nn.Module whose forward exercises each sampling op.
# No import-time side effects.
import torch


# https://pytorch.org/docs/stable/torch.html#random-sampling


class SamplingOpsModule(torch.nn.Module):
    # [CLASS DOC] Stateless nn.Module used by the mobile model test generator. Its forward
    # calls many random-sampling ops (bernoulli, multinomial, normal, poisson, rand*, randint*,
    # randperm, and in-place distribution fillers) so they appear in the traced op set.
    def forward(self):
        # [FUNCTION DOC] Takes no inputs. Builds a 3x3 uniform tensor, a size tuple and a
        # weights tensor, then calls each sampling op (some in-place ops mutate the local
        # tensor) and passes all results to the len builtin. Seed-related ops are left
        # commented out.
        # KNOWN ISSUE: len is given many positional arguments, which plain Python rejects
        # with a type error; the body exists only so the ops appear in a scripted graph.
        a = torch.empty(3, 3).uniform_(0.0, 1.0)
        size = (1, 4)
        weights = torch.tensor([0, 10, 3, 0], dtype=torch.float)
        return len(
            # torch.seed(),
            # torch.manual_seed(0),
            torch.bernoulli(a),
            # torch.initial_seed(),
            torch.multinomial(weights, 2),
            torch.normal(2.0, 3.0, size),
            torch.poisson(a),
            torch.rand(2, 3),
            torch.rand_like(a),
            torch.randint(10, size),
            torch.randint_like(a, 4),
            torch.rand(4),
            torch.randn_like(a),
            torch.randperm(4),
            a.bernoulli_(),
            a.cauchy_(),
            a.exponential_(),
            a.geometric_(0.5),
            a.log_normal_(),
            a.normal_(),
            a.random_(),
            a.uniform_(),
        )
