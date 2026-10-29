# [MODULE DOC] Constants for the ONNX exporter: the ONNX IR version written to files, the opset
# version and domain name that torchlib is implemented with, and the domain name used for
# functions translated from subgraphs. Pure data; no import-time side effects.
# ir_version used for the ONNX file. See https://github.com/onnx/onnx/blob/main/docs/IR.md#onnx-versioning
ONNX_IR_VERSION = 10
# The opset version torchlib is implemented with. Update this number when updating torchlib
TORCHLIB_OPSET = 18
TORCHLIB_DOMAIN = "pkg.torch.onnx"
# Domain used for functions translated from subgraphs
LOCAL_FUNCTION_DOMAIN = "pkg.torch.__subgraph__"
