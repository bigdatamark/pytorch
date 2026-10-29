# Copyright (c) Meta Platforms, Inc. and affiliates
# [PACKAGE DOC]
# Package of DTensor sharding strategies and propagation rules for ATen ops.
# Importing it star-imports the conv, embedding, math, matrix, pointwise, random,
# tensor and view op modules, each of which registers its strategies with the
# sharding propagator as an import-time side effect. It then calls
# auto_register_op_variants to register the remaining op variants (autogen).
# It defines no names of its own.
from ._conv_ops import *  # noqa: F403
from ._embedding_ops import *  # noqa: F403
from ._math_ops import *  # noqa: F403
from ._matrix_ops import *  # noqa: F403
from ._pointwise_ops import *  # noqa: F403
from ._random_ops import *  # noqa: F403
from ._tensor_ops import *  # noqa: F403
from ._view_ops import *  # noqa: F403
from .autogen import auto_register_op_variants


auto_register_op_variants()
