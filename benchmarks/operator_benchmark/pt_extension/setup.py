# [MODULE DOC] Build script for the benchmark_cpp_extension C++ extension (from extension.cpp).
# Calling setup at import/run time compiles it with the PyTorch BuildExtension build command.
from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CppExtension


setup(
    name="benchmark_cpp_extension",
    ext_modules=[CppExtension("benchmark_cpp_extension", ["extension.cpp"])],
    cmdclass={"build_ext": BuildExtension},
)
