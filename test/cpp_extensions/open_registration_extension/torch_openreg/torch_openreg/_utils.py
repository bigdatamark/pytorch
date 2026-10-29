# [MODULE DOC] Windows-only helper that preloads the extension's DLLs from the package lib
# directory so the native module can import. No import-time side effects beyond imports.
import ctypes
import glob
import os


def _load_dll_libraries() -> None:
    # [FUNCTION DOC] HELPER: Loads every DLL under the package "lib" directory via kernel32.
    # Adds the directory to the DLL search path, tries LoadLibraryEx with safe flags first and
    # falls back to prepending the directory to PATH (once) and plain LoadLibrary. Takes no
    # arguments, returns nothing; raises an OS error naming the DLL on load failure (except
    # error 126 on the first attempt, which triggers the fallback). Mutates the PATH
    # environment variable, the process error mode (restored at the end) and DLL search dirs.
    # KNOWN ISSUE: the error mode is not restored if an error is raised mid-loop.
    openreg_dll_path = os.path.join(os.path.dirname(__file__), "lib")

    kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
    with_load_library_flags = hasattr(kernel32, "AddDllDirectory")
    prev_error_mode = kernel32.SetErrorMode(0x0001)

    kernel32.LoadLibraryW.restype = ctypes.c_void_p
    if with_load_library_flags:
        kernel32.LoadLibraryExW.restype = ctypes.c_void_p

    os.add_dll_directory(openreg_dll_path)

    dlls = glob.glob(os.path.join(openreg_dll_path, "*.dll"))
    path_patched = False
    for dll in dlls:
        is_loaded = False
        if with_load_library_flags:
            res = kernel32.LoadLibraryExW(dll, None, 0x00001100)
            last_error = ctypes.get_last_error()
            if res is None and last_error != 126:
                err = ctypes.WinError(last_error)
                err.strerror += f' Error loading "{dll}" or one of its dependencies.'
                raise err
            elif res is not None:
                is_loaded = True
        if not is_loaded:
            if not path_patched:
                os.environ["PATH"] = ";".join([openreg_dll_path] + [os.environ["PATH"]])
                path_patched = True
            res = kernel32.LoadLibraryW(dll)
            if res is None:
                err = ctypes.WinError(ctypes.get_last_error())
                err.strerror += f' Error loading "{dll}" or one of its dependencies.'
                raise err

    kernel32.SetErrorMode(prev_error_mode)
