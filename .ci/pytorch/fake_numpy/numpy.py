# [MODULE DOC] HELPER: Shadow package named numpy placed on the module search path of CI
# jobs so that importing numpy fails. Import-time side effect: always raises a module-not-found
# error with a message saying the real NumPy lives elsewhere.
raise ModuleNotFoundError("Sorry PyTorch, but our NumPy is in the other folder")
