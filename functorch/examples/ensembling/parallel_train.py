# [MODULE DOC] Example script: shows that eager-mode vmap can train an ensemble of models in
# parallel (adapted from a JAX tutorial). Trains a two-layer MLP on a synthetic two-spiral
# dataset, first a single model then two models via vmap over a functional SGD step.
# Import-time side effects: parses command line arguments (--device), builds the dataset,
# the global model and loss, and runs the whole training demo (step4, step6) printing losses.
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad_and_value, stack_module_state, vmap


# Adapted from http://willwhitney.com/parallel-training-jax.html , which is a
# tutorial on Model Ensembling with JAX by Will Whitney.
#
# The original code comes with the following citation:
# @misc{Whitney2021Parallelizing,
#     author = {William F. Whitney},
#     title = { {Parallelizing neural networks on one GPU with JAX} },
#     year = {2021},
#     url = {http://willwhitney.com/parallel-training-jax.html},
# }

# GOAL: Demonstrate that it is possible to use eager-mode vmap
# to parallelize training over models.

parser = argparse.ArgumentParser(description="Functorch Ensembled Models")
parser.add_argument(
    "--device",
    type=str,
    default="cpu",
    help="CPU or GPU ID for this process (default: 'cpu')",
)
args = parser.parse_args()

DEVICE = args.device

# Step 1: Make some spirals


def make_spirals(n_samples, noise_std=0.0, rotations=1.0):
    # [FUNCTION DOC] Generates a two-class spiral dataset on the global DEVICE. Each sample
    # gets a random sign (class) and a radius/angle along the spiral, plus Gaussian noise
    # scaled by noise_std. Returns (points of shape n_samples by 2, long labels of n_samples).
    # Uses the global random number generator.
    ts = torch.linspace(0, 1, n_samples, device=DEVICE)
    rs = ts**0.5
    thetas = rs * rotations * 2 * math.pi
    signs = torch.randint(0, 2, (n_samples,), device=DEVICE) * 2 - 1
    labels = (signs > 0).to(torch.long).to(DEVICE)

    xs = (
        rs * signs * torch.cos(thetas)
        + torch.randn(n_samples, device=DEVICE) * noise_std
    )
    ys = (
        rs * signs * torch.sin(thetas)
        + torch.randn(n_samples, device=DEVICE) * noise_std
    )
    points = torch.stack([xs, ys], dim=1)
    return points, labels


points, labels = make_spirals(100, noise_std=0.05)


# Step 2: Define two-layer MLP and loss function
class MLPClassifier(nn.Module):
    # [CLASS DOC] Two-layer MLP classifier for 2-D inputs: a linear layer to hidden_dim, ReLU,
    # a linear layer to n_classes, and log-softmax output (suitable for NLL loss).
    # State: hidden_dim, n_classes, and submodules fc1, fc2.
    def __init__(self, hidden_dim=32, n_classes=2):
        # [FUNCTION DOC] Stores the sizes and creates the fc1 (2 to hidden_dim) and fc2
        # (hidden_dim to n_classes) linear layers.
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        self.fc1 = nn.Linear(2, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.n_classes)

    def forward(self, x):
        # [FUNCTION DOC] Maps a batch of 2-D points to per-class log probabilities
        # (last dimension is the class dimension).
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, -1)
        return x


loss_fn = nn.NLLLoss()
model = MLPClassifier().to(DEVICE)


def train_step_fn(weights, batch, targets, lr=0.2):
    # [FUNCTION DOC] One functional SGD step. Computes loss and gradients of the global model
    # (run via functional_call with the given weights mapping) on the batch, and returns
    # (loss, new weights mapping with each weight reduced by lr times its gradient). Does not
    # mutate the inputs; vmap-compatible so it can train many models at once.
    def compute_loss(weights, batch, targets):
        # [FUNCTION DOC] HELPER: Runs the global model with the given weights on the batch
        # and returns the global loss_fn value against targets.
        output = functional_call(model, weights, batch)
        loss = loss_fn(output, targets)
        return loss

    grad_weights, loss = grad_and_value(compute_loss)(weights, batch, targets)

    # NB: PyTorch is missing a "functional optimizer API" (possibly coming soon)
    # so we are going to re-implement SGD here.
    new_weights = {}
    with torch.no_grad():
        for key in grad_weights:
            new_weights[key] = weights[key] - grad_weights[key] * lr

    return loss, new_weights


# Step 4: Let's verify this actually trains.
# We should see the loss decrease.
def step4():
    # [FUNCTION DOC] HELPER: Trains the single global model for 2000 steps on the global
    # points and labels, printing the loss every 100 steps. Sets the module-global weights
    # to the latest weights mapping. The model's own parameters are not updated.
    global weights
    for i in range(2000):
        loss, weights = train_step_fn(dict(model.named_parameters()), points, labels)
        if i % 100 == 0:
            print(loss)


step4()

# Step 5: We're ready for multiple models. Let's define an init_fn
# that, given a number of models, returns to us all of the weights.


def init_fn(num_models):
    # [FUNCTION DOC] Creates num_models freshly initialized MLPClassifier instances on DEVICE
    # and returns their parameters stacked along a new leading dimension as a mapping.
    models = [MLPClassifier().to(DEVICE) for _ in range(num_models)]
    params, _ = stack_module_state(models)
    return params


# Step 6: Now, can we try multiple models at the same time?
# The answer is: yes! `loss` is a 2-tuple, and we can see that the value keeps
# on decreasing


def step6():
    # [FUNCTION DOC] HELPER: Trains two models at once for 2000 steps by mapping the train step
    # over the stacked weights (shared data), printing the per-model losses every 200 steps.
    parallel_train_step_fn = vmap(train_step_fn, in_dims=(0, None, None))
    batched_weights = init_fn(num_models=2)
    for i in range(2000):
        loss, batched_weights = parallel_train_step_fn(batched_weights, points, labels)
        if i % 200 == 0:
            print(loss)


step6()

# Step 7: Now, the flaw with step 6 is that we were training on the same exact
# data. This can lead to all of the models in the ensemble overfitting in the
# same way. The solution that http://willwhitney.com/parallel-training-jax.html
# applies is to randomly subset the data in a way that the models do not receive
# exactly the same data in each training step!
# Because the goal of this doc is to show that we can use eager-mode vmap to
# achieve similar things as JAX, the rest of this is left as an exercise to the reader.

# In conclusion, to achieve what http://willwhitney.com/parallel-training-jax.html
# does, we used the following additional items that PyTorch does not have:
# 1. NN module functional API that turns a module into a (state, state_less_fn) pair
# 2. Functional optimizers
# 3. A "functional" grad API (that effectively wraps autograd.grad)
# 4. Composability between the functional grad API and torch.vmap.
