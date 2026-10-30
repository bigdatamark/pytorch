# [MODULE DOC] Command-line shim that runs the composable_kernel fmha generate script on
# Python versions where the old module loader method was removed. It re-adds an equivalent
# load-module method to the source file loader class, then runs the given generate script as
# the main program, forwarding the remaining arguments. No import-time side effects beyond
# imports; the patching only happens when main runs.
"""Compatibility shim for composable_kernel's ck_tile/01_fmha/generate.py.

That script registers its codegen ops via ``loader.load_module()``, which was
removed in Python 3.15 (deprecated since 3.4). We re-add an equivalent
``load_module`` to importlib's source file loader, then run generate.py
unchanged. This unblocks ROCm wheel builds on Python 3.15 without bumping the
CK submodule. See https://github.com/pytorch/pytorch/issues/184900.

Remove this shim and inline the direct ``python3 .../generate.py`` calls in
CMakeLists.txt once third_party/composable_kernel is bumped past the upstream
load_module fix.

Usage: python3 generate_compat.py <path-to-generate.py> [generate.py args...]
"""

import importlib.machinery
import importlib.util
import os
import runpy
import sys


def _install_load_module_shim() -> None:
    # [FUNCTION DOC] HELPER: Add a load-module method to the importlib source file loader class
    # if it is missing (does nothing if present). This patches a standard library class for
    # the whole process. Returns None.
    loader = importlib.machinery.SourceFileLoader
    if hasattr(loader, "load_module"):
        return

    def load_module(self, fullname):
        # [FUNCTION DOC] HELPER: Replacement load-module method: build a spec from the loader,
        # create a module from it, execute the module with the loader and return the module.
        # Unlike the old method, it does not register the module in the loaded-modules table.
        spec = importlib.util.spec_from_loader(fullname, self)
        module = importlib.util.module_from_spec(spec)
        self.exec_module(module)
        return module

    loader.load_module = load_module


def main() -> None:
    # [FUNCTION DOC] Entry point: take the generate script path from the first command-line
    # argument, install the loader shim, rewrite the argument list so the script is the program
    # and the remaining arguments follow, put the script folder first on the module search
    # path and run the script as the main program. Mutates global argument and path state.
    generate_py = os.path.abspath(sys.argv[1])
    _install_load_module_shim()
    # Run generate.py as if invoked directly: argv[0] is the script and its
    # directory leads sys.path so its 'codegen' package imports resolve.
    sys.argv = [generate_py, *sys.argv[2:]]
    sys.path.insert(0, os.path.dirname(generate_py))
    runpy.run_path(generate_py, run_name="__main__")


if __name__ == "__main__":
    main()
