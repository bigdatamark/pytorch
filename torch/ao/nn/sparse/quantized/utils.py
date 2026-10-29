# [MODULE DOC] Global-context helper that conveys the block sparse pattern (row/col block
# size) for quantized sparse Linear. Exports LinearBlockSparsePattern. Only (1, 4) and
# (8, 1) patterns are valid. No import-time side effects besides creating a class-level lock.
import threading


__all__ = ["LinearBlockSparsePattern"]


def _is_valid_linear_block_sparse_pattern(
    row_block_size: int, col_block_size: int
) -> bool:
    # [FUNCTION DOC] HELPER: Returns True only if the block sizes are (1, 4) or (8, 1).
    return (row_block_size == 1 and col_block_size == 4) or (
        row_block_size == 8 and col_block_size == 1
    )


# This is a stop-gap measure as current flow does not allow module
# specific block sparse pattern.
# In fact there is no way to convey sparse pattern via module config
# of quantization flow. Thus using the global context to convey
# sparsity pattern.
# Once the flow supports it, this should be removed.
class LinearBlockSparsePattern:
    # [CLASS DOC] Context manager that temporarily sets a process-global block sparse
    # pattern. Class attributes hold the current and previous row/col block sizes plus a
    # reentrant lock. Constructing an instance already acquires the lock and sets the
    # pattern; leaving the with block restores the previous values and releases the lock.
    # KNOWN ISSUE: the lock is acquired in __init__, so an instance that is never entered
    # as a context manager never releases it; nested use only remembers one previous level.
    rlock = threading.RLock()
    row_block_size: int = 1
    col_block_size: int = 4
    prev_row_block_size: int = 1
    prev_col_block_size: int = 4

    def __init__(self, row_block_size: int = 1, col_block_size: int = 4):
        # [FUNCTION DOC] Validates the pattern (AssertionError if not (1,4) or (8,1)),
        # acquires the class lock, saves the current sizes as previous, and sets the new sizes.
        if not _is_valid_linear_block_sparse_pattern(row_block_size, col_block_size):
            raise AssertionError(
                f"Invalid linear block sparse pattern: "
                f"row_block_size={row_block_size}, col_block_size={col_block_size}"
            )
        LinearBlockSparsePattern.rlock.acquire()
        LinearBlockSparsePattern.prev_row_block_size = (
            LinearBlockSparsePattern.row_block_size
        )
        LinearBlockSparsePattern.prev_col_block_size = (
            LinearBlockSparsePattern.col_block_size
        )
        LinearBlockSparsePattern.row_block_size = row_block_size
        LinearBlockSparsePattern.col_block_size = col_block_size

    def __enter__(self) -> None:
        # [FUNCTION DOC] No-op; the pattern was already applied in the constructor.
        pass

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        backtrace: object | None,
    ) -> None:
        # [FUNCTION DOC] Restores the previous block sizes and releases the class lock.
        # Exceptions are not suppressed.
        LinearBlockSparsePattern.row_block_size = (
            LinearBlockSparsePattern.prev_row_block_size
        )
        LinearBlockSparsePattern.col_block_size = (
            LinearBlockSparsePattern.prev_col_block_size
        )
        LinearBlockSparsePattern.rlock.release()

    @staticmethod
    def block_size() -> tuple[int, int]:
        # [FUNCTION DOC] Returns the current global (row_block_size, col_block_size).
        return (
            LinearBlockSparsePattern.row_block_size,
            LinearBlockSparsePattern.col_block_size,
        )
