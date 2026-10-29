# Owner(s): ["module: intel"]

# [MODULE DOC] Tests for the Xeon CPU launcher (torch.backends.xeon.run_cpu): parsing of lscpu
# topology output into core and NUMA node lists, and launching multiple instances through a
# subprocess. Tests are skipped on non-Linux platforms. No import-time side effects.

import shutil
import subprocess
import tempfile
import unittest

from torch.testing._internal.common_utils import IS_LINUX, run_tests, TestCase


@unittest.skipIf(not IS_LINUX, "Only works on linux")
class TestTorchrun(TestCase):
    # [CLASS DOC] Tests for the run_cpu launcher. Holds _test_dir, a per-test temporary directory
    # used as the launcher log path. Skipped entirely when not running on Linux.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Runs base setup and creates a temporary directory named with the
        # class name prefix, stored in _test_dir.
        super().setUp()
        self._test_dir = tempfile.mkdtemp(prefix=self.__class__.__name__)

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Recursively deletes the temporary directory (raises if missing).
        # KNOWN ISSUE: does not call the base tearDown.
        shutil.rmtree(self._test_dir)

    def test_cpu_info(self):
        # [FUNCTION DOC] Feeds a fixed 16-logical-CPU, 2-socket lscpu parsable text to _CPUinfo and
        # checks physical/logical core counts, per-node physical and logical core lists, all-core
        # lists (logical ordered node by node), and numa_aware_check results (which NUMA nodes a
        # given core list spans). Raises AssertionError on any mismatch.
        lscpu_info = """# The following is the parsable format, which can be fed to other
# programs. Each different item in every column has an unique ID
# starting from zero.
# CPU,Core,Socket,Node
0,0,0,0
1,1,0,0
2,2,0,0
3,3,0,0
4,4,1,1
5,5,1,1
6,6,1,1
7,7,1,1
8,0,0,0
9,1,0,0
10,2,0,0
11,3,0,0
12,4,1,1
13,5,1,1
14,6,1,1
15,7,1,1
"""
        from torch.backends.xeon.run_cpu import _CPUinfo

        cpuinfo = _CPUinfo(lscpu_info)
        if cpuinfo._physical_core_nums() != 8:
            raise AssertionError(
                f"Expected 8 physical cores, got {cpuinfo._physical_core_nums()}"
            )
        if cpuinfo._logical_core_nums() != 16:
            raise AssertionError(
                f"Expected 16 logical cores, got {cpuinfo._logical_core_nums()}"
            )
        if cpuinfo.get_node_physical_cores(0) != [0, 1, 2, 3]:
            raise AssertionError(
                f"Expected [0, 1, 2, 3], got {cpuinfo.get_node_physical_cores(0)}"
            )
        if cpuinfo.get_node_physical_cores(1) != [4, 5, 6, 7]:
            raise AssertionError(
                f"Expected [4, 5, 6, 7], got {cpuinfo.get_node_physical_cores(1)}"
            )
        if cpuinfo.get_node_logical_cores(0) != [0, 1, 2, 3, 8, 9, 10, 11]:
            raise AssertionError(
                f"Expected [0, 1, 2, 3, 8, 9, 10, 11], got {cpuinfo.get_node_logical_cores(0)}"
            )
        if cpuinfo.get_node_logical_cores(1) != [4, 5, 6, 7, 12, 13, 14, 15]:
            raise AssertionError(
                f"Expected [4, 5, 6, 7, 12, 13, 14, 15], got {cpuinfo.get_node_logical_cores(1)}"
            )
        if cpuinfo.get_all_physical_cores() != [0, 1, 2, 3, 4, 5, 6, 7]:
            raise AssertionError(
                f"Expected [0, 1, 2, 3, 4, 5, 6, 7], got {cpuinfo.get_all_physical_cores()}"
            )
        expected_logical = [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15]
        if cpuinfo.get_all_logical_cores() != expected_logical:
            raise AssertionError(
                f"Expected {expected_logical}, got {cpuinfo.get_all_logical_cores()}"
            )
        if cpuinfo.numa_aware_check([0, 1, 2, 3]) != [0]:
            raise AssertionError(
                f"Expected [0], got {cpuinfo.numa_aware_check([0, 1, 2, 3])}"
            )
        if cpuinfo.numa_aware_check([4, 5, 6, 7]) != [1]:
            raise AssertionError(
                f"Expected [1], got {cpuinfo.numa_aware_check([4, 5, 6, 7])}"
            )
        if cpuinfo.numa_aware_check([2, 3, 4, 5]) != [0, 1]:
            raise AssertionError(
                f"Expected [0, 1], got {cpuinfo.numa_aware_check([2, 3, 4, 5])}"
            )

    def test_multi_threads(self):
        # [FUNCTION DOC] Runs the launcher in a shell subprocess with 4 instances, all
        # optimizations disabled, no-python mode and a print-working-directory command, logging
        # into _test_dir. Counts output lines whose last dash-separated segment is the command and
        # raises AssertionError unless exactly 4 were seen.
        num = 0
        with subprocess.Popen(
            f"python -m torch.backends.xeon.run_cpu --ninstances 4 --use-default-allocator \
            --disable-iomp --disable-numactl --disable-taskset --log-path {self._test_dir} --no-python pwd",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ) as p:
            for line in p.stdout.readlines():
                segs = str(line, "utf-8").strip().split("-")
                if segs[-1].strip() == "pwd":
                    num += 1
        if num != 4:
            raise AssertionError(
                f"Failed to launch multiple instances for inference, got {num}"
            )

    def test_multi_ncores_per_instance_setting(self):
        # [FUNCTION DOC] Same as the multi-thread test but with 3 instances and explicit per-
        # instance core counts (5, 5, 6). Raises AssertionError unless exactly 3 launched
        # instances are observed in the output.
        # KNOWN ISSUE: the core counts total 16, so the test may depend on the host CPU count.
        num = 0
        with subprocess.Popen(
            f"python -m torch.backends.xeon.run_cpu --ninstances 3 --ncores-per-instance 5 5 6 --use-default-allocator \
            --disable-iomp --disable-numactl --disable-taskset --log-path {self._test_dir} --no-python pwd",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ) as p:
            for line in p.stdout.readlines():
                segs = str(line, "utf-8").strip().split("-")
                if segs[-1].strip() == "pwd":
                    num += 1
        if num != 3:
            raise AssertionError(
                f"Failed to launch multiple instances for inference, got {num}"
            )


if __name__ == "__main__":
    run_tests()
