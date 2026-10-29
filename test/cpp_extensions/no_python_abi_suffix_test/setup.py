# [MODULE DOC] Build script for the no_python_abi_suffix_test C++ extension, used to test that
# BuildExtension with the no-python-ABI-suffix option produces a shared library without the
# interpreter tag in its filename. Calls setup at import/run time, compiling the single cpp
# source next to it.
from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CppExtension


setup(
    name="no_python_abi_suffix_test",
    ext_modules=[
        CppExtension("no_python_abi_suffix_test", ["no_python_abi_suffix_test.cpp"])
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
