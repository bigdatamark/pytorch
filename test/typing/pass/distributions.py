# [MODULE DOC] Static typing check that distribution attributes are typed as Tensor. It builds a
# Normal distribution from two Python numbers and a MultivariateNormal from a zero vector and
# identity matrix, and asserts the mean and covariance_matrix attributes are typed as Tensor.
# Constructing the distributions runs at import time.
from typing_extensions import assert_type

import torch
from torch import distributions, Tensor


dist = distributions.Normal(0, 1)
assert_type(dist.mean, Tensor)

dist = distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
assert_type(dist.covariance_matrix, Tensor)
