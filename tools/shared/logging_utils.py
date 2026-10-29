# [MODULE DOC] Tiny string-formatting helpers for log output: pluralizing a count with a noun
# and rendering a duration in seconds as a short human-readable string. Pure functions; no
# import-time side effects.
def pluralize(count: int, singular_word: str, plural_word: str = "") -> str:
    # [FUNCTION DOC] Format a count with its noun. Args: count, singular_word, plural_word
    # (optional; when empty the singular word with an s appended is used). Returns the
    # string 'N word', using the singular form only when count equals one (so zero gives the
    # plural).
    if count == 1:
        return f"{count} {singular_word}"

    if not plural_word:
        plural_word = f"{singular_word}s"

    return f"{count} {plural_word}"


def duration_to_str(seconds: float) -> str:
    # [FUNCTION DOC] Render a duration in seconds as a short string with one decimal:
    # seconds with suffix s below a minute, minutes with suffix m below an hour, otherwise
    # hours with suffix h. Durations under a very small threshold (0.00001) return exactly
    # 0s. Negative values also return 0s.
    if seconds < 0.00001:
        return "0s"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"
