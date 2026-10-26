# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# [PACKAGE DOC] Package for activation offloading in the functorch AOT pipeline.
# Contains activation_offloading (a graph pass that moves saved activations to
# CPU between forward and backward) and offload_ops (custom ops used by that
# pass). This __init__ defines no exports and has no import-time side effects.
