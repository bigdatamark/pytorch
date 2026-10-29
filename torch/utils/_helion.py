# [MODULE DOC] Cached availability checks for the optional helion package. No import-time
# side effects (helion is imported lazily inside the check).
import functools


@functools.cache
def has_helion_package() -> bool:
    # [FUNCTION DOC] Returns True if the helion package can be imported, else False (only an
    # import error is caught). Cached after the first call; importing helion is a side effect.
    try:
        import helion  # type: ignore[import-untyped, import-not-found]  # noqa: F401
    except ImportError:
        return False
    return True


@functools.cache
def has_helion() -> bool:
    # [FUNCTION DOC] Returns whether helion is usable; currently identical to the package check.
    # Cached after the first call.
    return has_helion_package()
