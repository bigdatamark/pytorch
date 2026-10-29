# [MODULE DOC] Defines the shared persistent memoizer instance used by pad_mm for its should-pad
# decision, storing results under the should_pad sub-directory. Creating it at import time
# constructs a PersistentMemoizer.
from pathlib import Path

from .interfaces import PersistentMemoizer


# Memoizer for _should_pad in pad_mm
should_pad_memoizer = PersistentMemoizer(sub_dir=Path("should_pad"))
