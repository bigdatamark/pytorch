# [MODULE DOC] Predefined QConfig objects for experimental APoT quantization at 8 and 4 bits (b with k=2 per comment):
# uniform, APoT-weight and APoT-activation combinations built from FakeQuantize and APoTFakeQuantize factories, plus the default
# symmetric fake-quant factories they use. Import-time side effect: constructs these factory/QConfig objects.
# KNOWN ISSUE: some configs pass a with_args bound method (e.g. uniform_qconfig_8bit weight) where others pass a factory instance, so calling the weight entry returns another factory rather than a module.
import torch
from torch.ao.quantization import MinMaxObserver
from torch.ao.quantization.experimental.fake_quantize import APoTFakeQuantize
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.qconfig import QConfig


"""
Default symmetric fake_quant for activations.
"""
default_symmetric_fake_quant = FakeQuantize.with_args(
    observer=MinMaxObserver, qscheme=torch.per_tensor_symmetric, dtype=torch.quint8
)

"""
Default symmetric fake_quant for weights.
"""
default_weight_symmetric_fake_quant = FakeQuantize.with_args(
    observer=MinMaxObserver, qscheme=torch.per_tensor_symmetric, dtype=torch.qint8
)

# uniform activation and weight, b=8 k=2
uniform_qconfig_8bit = QConfig(
    activation=default_symmetric_fake_quant,
    weight=default_weight_symmetric_fake_quant.with_args,
)

# uniform activation, APoT weight, b=8 k=2
apot_weight_qconfig_8bit = QConfig(
    activation=default_symmetric_fake_quant.with_args,
    weight=APoTFakeQuantize.with_args(b=8, k=2, dtype=torch.qint8),
)

# APoT activation and uniform weight, b=8 k=2
apot_qconfig_8bit = QConfig(
    activation=APoTFakeQuantize.with_args(b=8, k=2, dtype=torch.quint8),
    weight=APoTFakeQuantize.with_args(b=8, k=2, dtype=torch.qint8),
)

# uniform activation and weight, b=4 k=2
uniform_qconfig_4bit = QConfig(
    activation=default_symmetric_fake_quant.with_args(quant_min=0, quant_max=15),
    weight=default_weight_symmetric_fake_quant.with_args(quant_min=0, quant_max=15),
)

# uniform activation, APoT weight, b=4 k=2
apot_weight_qconfig_4bit = QConfig(
    activation=default_symmetric_fake_quant.with_args(quant_min=0, quant_max=15),
    weight=APoTFakeQuantize.with_args(b=4, k=2, dtype=torch.qint8),
)

# APoT activation and uniform weight, b=4 k=2
apot_qconfig_4bit = QConfig(
    activation=APoTFakeQuantize.with_args(b=4, k=2, dtype=torch.quint8),
    weight=APoTFakeQuantize.with_args(b=4, k=2, dtype=torch.qint8),
)
