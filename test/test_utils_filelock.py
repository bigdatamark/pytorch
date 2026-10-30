# Owner(s): ["module: unknown"]
# [MODULE DOC] Tests for the monitored FileLock in torch.utils._filelock: acquiring the lock in a
# simple case and serializing concurrent access to a shared file from many threads.
import concurrent.futures
import tempfile
import time

from torch.testing._internal.common_utils import run_tests, skipIfWindows, TestCase
from torch.utils._filelock import FileLock


class TestFileLock(TestCase):
    # [CLASS DOC] Test case for FileLock behavior.
    def test_no_crash(self):
        # [FUNCTION DOC] Creates a temporary file path and enters and leaves the lock on it,
        # checking only that no exception is raised. The temporary file is never removed.
        _, p = tempfile.mkstemp()
        with FileLock(p):
            pass

    @skipIfWindows(
        msg="Windows doesn't support multiple files being opened at once easily"
    )
    def test_sequencing(self):
        # [FUNCTION DOC] Skipped on Windows. Ten threads each take a lock on a sidecar lock file
        # and then append their index to a shared temporary file, recording start and end times;
        # the test checks all ten indices were written and that no two recorded intervals overlap
        # (mutual exclusion).
        with tempfile.NamedTemporaryFile() as ofd:
            p = ofd.name

            def test_thread(i):
                # [FUNCTION DOC] HELPER: Thread body. Holds the lock file for the shared path,
                # appends the given index as text to the shared file, and returns the pair of
                # start and end timestamps.
                with FileLock(p + ".lock"):
                    start = time.time()
                    with open(p, "a") as fd:
                        fd.write(str(i))
                    end = time.time()
                    return (start, end)

            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(test_thread, i) for i in range(10)]
                times = []
                for f in futures:
                    times.append(f.result(60))

            with open(p) as fd:
                self.assertEqual(
                    set(fd.read()), {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}
                )

            for i, (start, end) in enumerate(times):
                for j, (newstart, newend) in enumerate(times):
                    if i == j:
                        continue

                    # Times should never intersect
                    self.assertFalse(newstart > start and newstart < end)
                    self.assertFalse(newend > start and newstart < end)


if __name__ == "__main__":
    run_tests()
