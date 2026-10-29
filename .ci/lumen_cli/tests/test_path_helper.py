# test_path_utils.py
# Run: pytest -q

# [MODULE DOC] Unit tests for path_helper using temporary directories: path conversion
# and resolution, directory creation, removal and forced recreation, file and directory
# copy including merging, and existence checks.

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cli.lib.common.path_helper import (
    copy,
    ensure_dir_exists,
    force_create_dir,
    get_path,
    is_path_exist,
    remove_dir,
)


class TestPathHelper(unittest.TestCase):
    # [CLASS DOC] Test case for path helpers; each test gets a fresh temporary directory
    # exposed as tmpdir and tmp_path and removed afterwards.
    def setUp(self):
        # [FUNCTION DOC] Create the temporary directory and store its Path.
        self.tmpdir = TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)

    def tearDown(self):
        # [FUNCTION DOC] Delete the temporary directory.
        self.tmpdir.cleanup()

    # -------- get_path --------
    def test_get_path_returns_path_for_str(self):
        # [FUNCTION DOC] A relative string stays a relative Path equal to the input when
        # resolution is off. KNOWN ISSUE: changes the process working directory into the
        # temporary directory and never restores it.
        # Use relative path to avoid absolute-ness
        rel_str = "sub/f.txt"
        os.chdir(self.tmp_path)
        p = get_path(rel_str, resolve=False)
        self.assertIsInstance(p, Path)
        self.assertFalse(p.is_absolute())
        self.assertEqual(str(p), rel_str)

    def test_get_path_resolves(self):
        # [FUNCTION DOC] With resolution on the result is absolute and keeps the tail.
        rel_str = "sub/f.txt"
        p = get_path(str(self.tmp_path / rel_str), resolve=True)
        self.assertTrue(p.is_absolute())
        self.assertTrue(str(p).endswith(rel_str))

    def test_get_path_with_path_input(self):
        # [FUNCTION DOC] A Path input comes back with the same text form.
        p_in = self.tmp_path / "sub/f.txt"
        p_out = get_path(p_in, resolve=False)
        self.assertTrue(str(p_out) == str(p_in))

    def test_get_path_with_none_raises(self):
        # [FUNCTION DOC] A None input raises ValueError.
        with self.assertRaises(ValueError):
            get_path(None)  # type: ignore[arg-type]

    def test_get_path_invalid_type_raises(self):
        # [FUNCTION DOC] A non-path, non-string input (an integer) raises TypeError.
        with self.assertRaises(TypeError):
            get_path(123)  # type: ignore[arg-type]

    # -------- ensure_dir_exists / force_create_dir / remove_dir --------
    def test_ensure_dir_exists_creates_and_is_idempotent(self):
        # [FUNCTION DOC] Creating a directory works and repeating the call is harmless.
        d = self.tmp_path / "made"
        ensure_dir_exists(d)
        self.assertTrue(d.exists() and d.is_dir())
        ensure_dir_exists(d)

    def test_force_create_dir_clears_existing(self):
        # [FUNCTION DOC] Forced creation on a populated directory leaves it existing and
        # empty.
        d = self.tmp_path / "fresh"
        (d / "inner").mkdir(parents=True)
        (d / "inner" / "f.txt").write_text("x")
        force_create_dir(d)
        self.assertTrue(d.exists())
        self.assertEqual(list(d.iterdir()), [])

    def test_remove_dir_none_is_noop(self):
        # [FUNCTION DOC] Removing a None path does nothing and raises nothing.
        remove_dir(None)  # type: ignore[arg-type]

    def test_remove_dir_nonexistent_is_noop(self):
        # [FUNCTION DOC] Removing a path that does not exist does nothing.
        ghost = self.tmp_path / "ghost"
        remove_dir(ghost)

    def test_remove_dir_accepts_str(self):
        # [FUNCTION DOC] A directory given as a string is removed.
        d = self.tmp_path / "to_rm"
        d.mkdir()
        remove_dir(str(d))
        self.assertFalse(d.exists())

    # -------- copy --------
    def test_copy_file_to_file(self):
        # [FUNCTION DOC] Copying a file into a not-yet-existing subdirectory creates the
        # parent and preserves the content.
        src = self.tmp_path / "src.txt"
        dst = self.tmp_path / "out" / "dst.txt"
        src.write_text("hello")
        copy(src, dst)
        self.assertEqual(dst.read_text(), "hello")

    def test_copy_dir_to_new_dir(self):
        # [FUNCTION DOC] Copying a directory tree to a new location reproduces nested files.
        src = self.tmp_path / "srcdir"
        (src / "a").mkdir(parents=True)
        (src / "a" / "f.txt").write_text("content")
        dst = self.tmp_path / "destdir"
        copy(src, dst)
        self.assertEqual((dst / "a" / "f.txt").read_text(), "content")

    def test_copy_dir_into_existing_dir_overwrite_true_merges(self):
        # [FUNCTION DOC] Copying a tree onto an existing directory merges: old files are
        # kept and new files added. There is no overwrite option despite the name.
        src = self.tmp_path / "srcdir"
        dst = self.tmp_path / "destdir"
        (src / "x").mkdir(parents=True)
        (src / "x" / "new.txt").write_text("new")
        dst.mkdir()
        (dst / "existing.txt").write_text("old")
        copy(src, dst)
        self.assertEqual((dst / "existing.txt").read_text(), "old")
        self.assertEqual((dst / "x" / "new.txt").read_text(), "new")

    def test_is_str_path_exist(self):
        # [FUNCTION DOC] Existence check is true for an existing file given as string or
        # Path and false for missing paths and the empty string.
        p = self.tmp_path / "x.txt"
        p.write_text("1")
        self.assertTrue(is_path_exist(str(p)))
        self.assertTrue(is_path_exist(p))
        self.assertFalse(is_path_exist(str(self.tmp_path / "missing")))
        self.assertFalse(is_path_exist(self.tmp_path / "missing"))
        self.assertFalse(is_path_exist(""))


if __name__ == "__main__":
    unittest.main()
