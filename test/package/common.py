# [MODULE DOC] Shared base test case for the torch.package tests. Defines PackageTestCase, which
# manages temporary files and makes the fake fixture packages in this directory importable.
# Importing the module also imports the package exporter module; it has no other
# import-time effects.
import os
import sys
from tempfile import NamedTemporaryFile

import torch.package.package_exporter
from torch.testing._internal.common_utils import IS_WINDOWS, TestCase


class PackageTestCase(TestCase):
    # [CLASS DOC] Base class for torch.package tests. Keeps a list of open NamedTemporaryFile
    # objects (the private temporary files list) that are closed in tearDown, saves and restores
    # the module search path around each test, and turns off the TorchScript serialization gate
    # of the exporter.
    def __init__(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Forwards all arguments to the TestCase constructor and then
        # creates an empty list that will hold the temporary file objects created by the
        # temp method.
        super().__init__(*args, **kwargs)
        self._temporary_files = []

    def temp(self):
        # [FUNCTION DOC] Creates a named temporary file and returns its path string. On Windows
        # the file is closed immediately so that it can be reopened by name; elsewhere the open
        # file object is kept in the temporary files list so that it stays alive until tearDown
        # closes (and deletes) it.
        t = NamedTemporaryFile()  # noqa: SIM115
        name = t.name
        if IS_WINDOWS:
            t.close()  # can't read an open file in windows
        else:
            self._temporary_files.append(t)
        return name

    def setUp(self):
        # [FUNCTION DOC] Runs the base setUp, records the directory containing this file, saves
        # a copy of the current module search path and appends the test directory to it so the
        # fake packages can be imported by name. Side effect: sets the exporter module flag that
        # gates TorchScript serialization to False globally (it is never restored).
        """Add test/package/ to module search path. This ensures that
        importing our fake packages via, e.g. `import package_a` will always
        work regardless of how we invoke the test.
        """
        super().setUp()
        self.package_test_dir = os.path.dirname(os.path.realpath(__file__))
        self.orig_sys_path = sys.path.copy()
        sys.path.append(self.package_test_dir)
        torch.package.package_exporter._gate_torchscript_serialization = False

    def tearDown(self):
        # [FUNCTION DOC] Runs the base tearDown, restores the module search path saved in setUp,
        # then closes every temporary file created through temp (which removes them) and empties
        # the tracking list.
        super().tearDown()
        sys.path = self.orig_sys_path

        # remove any temporary files
        for t in self._temporary_files:
            t.close()
        self._temporary_files = []
