# Owner(s): ["module: hub"]
# [MODULE DOC] Tests for torch.hub. Network-backed tests load models, state dicts and entrypoint
# lists from the ailzhang torchhub_example GitHub repo and release assets, and check the repo-trust
# prompt and trusted-list file behavior. Offline tests check that zip extraction rejects directory
# traversal and absolute paths. Defines a state-dict summing helper and the expected sum and
# release URL constants. No import-time side effects beyond defining them.

import os
import tempfile
import unittest
import warnings
import zipfile
from unittest.mock import patch

import torch
import torch.hub as hub
from torch.testing._internal.common_utils import (
    IS_SANDCASTLE,
    retry,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)


def sum_of_state_dict(state_dict):
    # [FUNCTION DOC] HELPER: Adds up the sum of every tensor value in a state dict mapping and
    # returns the total as a tensor (starting from the integer zero). Used to fingerprint
    # downloaded weights against the expected constant.
    s = 0
    for v in state_dict.values():
        s += v.sum()
    return s


SUM_OF_HUB_EXAMPLE = 431080
TORCHHUB_EXAMPLE_RELEASE_URL = (
    "https://github.com/ailzhang/torchhub_example/releases/download/0.1/mnist_init_ones"
)


@unittest.skipIf(IS_SANDCASTLE, "Sandcastle cannot ping external")
class TestHub(TestCase):
    # [CLASS DOC] Test case for hub loading, caching and trust handling; skipped entirely on
    # Sandcastle because it needs external network. Each test gets a private temporary hub
    # directory and a path to the trusted list file inside it, stored in instance attributes.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Fixture setup. Remembers the current hub directory, creates a
        # temporary directory, points torch.hub at it, and records where the trusted list file
        # would be. Mutates the global hub directory setting.
        super().setUp()
        self.previous_hub_dir = torch.hub.get_dir()
        self.tmpdir = tempfile.TemporaryDirectory("hub_dir")
        torch.hub.set_dir(self.tmpdir.name)
        self.trusted_list_path = os.path.join(torch.hub.get_dir(), "trusted_list")

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Fixture teardown. Restores the previously saved hub directory in
        # the global hub setting and removes the temporary directory.
        super().tearDown()
        torch.hub.set_dir(self.previous_hub_dir)  # probably not needed, but can't hurt
        self.tmpdir.cleanup()

    def _assert_trusted_list_is_empty(self):
        # [FUNCTION DOC] HELPER: Reads the trusted list file and raises AssertionError listing the
        # lines if it contains any line. Raises a file-not-found error if the file does not exist.
        with open(self.trusted_list_path) as f:
            lines = f.readlines()
            if lines:
                raise AssertionError(f"Expected empty trusted list, got {lines}")

    def _assert_in_trusted_list(self, line):
        # [FUNCTION DOC] HELPER: Opens the trusted list file and raises AssertionError unless some
        # stripped line equals the given repo entry string.
        with open(self.trusted_list_path) as f:
            if line not in (l.strip() for l in f):
                raise AssertionError(f"Expected {line} in trusted list")

    @retry(Exception, tries=3)
    def test_load_from_github(self):
        # [FUNCTION DOC] Loads the mnist model with pretrained weights from the GitHub example repo
        # (network access, retried up to three times) and checks the sum of its state dict equals
        # the expected constant.
        hub_model = hub.load(
            "ailzhang/torchhub_example",
            "mnist",
            source="github",
            pretrained=True,
            verbose=False,
        )
        self.assertEqual(sum_of_state_dict(hub_model.state_dict()), SUM_OF_HUB_EXAMPLE)

    @retry(Exception, tries=3)
    def test_load_from_local_dir(self):
        # [FUNCTION DOC] Fetches the example repo into the cache with trust enabled, then loads
        # mnist pretrained weights from that local directory using the local source, and checks the
        # state dict sum against the expected constant. Retried on failure.
        local_dir = hub._get_cache_or_reload(
            "ailzhang/torchhub_example",
            force_reload=False,
            trust_repo=True,
        )
        hub_model = hub.load(
            local_dir, "mnist", source="local", pretrained=True, verbose=False
        )
        self.assertEqual(sum_of_state_dict(hub_model.state_dict()), SUM_OF_HUB_EXAMPLE)

    @retry(Exception, tries=3)
    def test_load_from_branch(self):
        # [FUNCTION DOC] Loads mnist with pretrained weights from a repo spec that names a branch
        # containing a slash, and checks the state dict sum equals the expected constant. Retried
        # on failure.
        hub_model = hub.load(
            "ailzhang/torchhub_example:ci/test_slash",
            "mnist",
            pretrained=True,
            verbose=False,
        )
        self.assertEqual(sum_of_state_dict(hub_model.state_dict()), SUM_OF_HUB_EXAMPLE)

    @retry(Exception, tries=3)
    def test_get_set_dir(self):
        # [FUNCTION DOC] Sets the hub directory to a fresh temporary directory and checks the
        # getter returns it, loads a model and verifies the repo checkout directory was created
        # inside it. Afterwards checks that setting a directory with a home-user tilde stores the
        # expanded path (regression for an upstream issue). Raises AssertionError if the checkout
        # directory is missing.
        previous_hub_dir = torch.hub.get_dir()
        with tempfile.TemporaryDirectory("hub_dir") as tmpdir:
            torch.hub.set_dir(tmpdir)
            self.assertEqual(torch.hub.get_dir(), tmpdir)
            self.assertNotEqual(previous_hub_dir, tmpdir)

            hub_model = hub.load(
                "ailzhang/torchhub_example", "mnist", pretrained=True, verbose=False
            )
            self.assertEqual(
                sum_of_state_dict(hub_model.state_dict()), SUM_OF_HUB_EXAMPLE
            )
            if not os.path.exists(
                os.path.join(tmpdir, "ailzhang_torchhub_example_master")
            ):
                raise AssertionError(
                    "Expected torchhub_example_master directory to exist"
                )

        # Test that set_dir properly calls expanduser()
        # non-regression test for https://github.com/pytorch/pytorch/issues/69761
        new_dir = os.path.join("~", "hub")
        torch.hub.set_dir(new_dir)
        self.assertEqual(torch.hub.get_dir(), os.path.expanduser(new_dir))

    @retry(Exception, tries=3)
    def test_list_entrypoints(self):
        # [FUNCTION DOC] Lists entrypoints of the example repo with trust enabled and asserts that
        # the entry named mnist is present. Retried on failure.
        entry_lists = hub.list("ailzhang/torchhub_example", trust_repo=True)
        self.assertObjectIn("mnist", entry_lists)

    @retry(Exception, tries=3)
    def test_download_url_to_file(self):
        # [FUNCTION DOC] Downloads the release asset to a temporary file without a progress bar,
        # loads it as a saved state and checks its sum. Also creates an empty reference file and
        # raises AssertionError if the downloaded file's permission bits differ from the reference
        # (so downloads get default permissions).
        with tempfile.TemporaryDirectory() as tmpdir:
            f = os.path.join(tmpdir, "temp")
            hub.download_url_to_file(TORCHHUB_EXAMPLE_RELEASE_URL, f, progress=False)
            loaded_state = torch.load(f)
            self.assertEqual(sum_of_state_dict(loaded_state), SUM_OF_HUB_EXAMPLE)
            # Check that the downloaded file has default file permissions
            f_ref = os.path.join(tmpdir, "reference")
            open(f_ref, "w").close()
            expected_permissions = oct(os.stat(f_ref).st_mode & 0o777)
            actual_permissions = oct(os.stat(f).st_mode & 0o777)
            if actual_permissions != expected_permissions:
                raise AssertionError(
                    f"Expected permissions {expected_permissions}, got {actual_permissions}"
                )

    @retry(Exception, tries=3)
    def test_load_state_dict_from_url(self):
        # [FUNCTION DOC] Loads a state dict from the release URL and checks its sum; then again
        # with an explicit file name and checks that the file exists under the checkpoints folder
        # of the hub directory; then again with weights-only loading. All three results must match
        # the expected sum.
        loaded_state = hub.load_state_dict_from_url(TORCHHUB_EXAMPLE_RELEASE_URL)
        self.assertEqual(sum_of_state_dict(loaded_state), SUM_OF_HUB_EXAMPLE)

        # with name
        file_name = "the_file_name"
        loaded_state = hub.load_state_dict_from_url(
            TORCHHUB_EXAMPLE_RELEASE_URL, file_name=file_name
        )
        expected_file_path = os.path.join(torch.hub.get_dir(), "checkpoints", file_name)
        self.assertTrue(os.path.exists(expected_file_path))
        self.assertEqual(sum_of_state_dict(loaded_state), SUM_OF_HUB_EXAMPLE)

        # with safe weight_only
        loaded_state = hub.load_state_dict_from_url(
            TORCHHUB_EXAMPLE_RELEASE_URL, weights_only=True
        )
        self.assertEqual(sum_of_state_dict(loaded_state), SUM_OF_HUB_EXAMPLE)

    @retry(Exception, tries=3)
    def test_load_legacy_zip_checkpoint(self):
        # [FUNCTION DOC] Loads the mnist_zip entrypoint (a legacy non-zipfile checkpoint format)
        # while recording all warnings, checks the weights sum, and raises AssertionError unless a
        # warning mentioning deprecation in favor of the default zipfile format was emitted.
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            hub_model = hub.load(
                "ailzhang/torchhub_example", "mnist_zip", pretrained=True, verbose=False
            )
            self.assertEqual(
                sum_of_state_dict(hub_model.state_dict()), SUM_OF_HUB_EXAMPLE
            )
            if not any(
                "will be deprecated in favor of default zipfile" in str(w) for w in ws
            ):
                raise AssertionError("Expected deprecation warning for legacy zip")

    # Test the default zipfile serialization format produced by >=1.6 release.
    @retry(Exception, tries=3)
    def test_load_zip_1_6_checkpoint(self):
        # [FUNCTION DOC] Loads the mnist_zip_1_6 entrypoint (the zipfile serialization used by
        # releases 1.6 and later) with trust enabled and checks the state dict sum equals the
        # expected constant.
        hub_model = hub.load(
            "ailzhang/torchhub_example",
            "mnist_zip_1_6",
            pretrained=True,
            verbose=False,
            trust_repo=True,
        )
        self.assertEqual(sum_of_state_dict(hub_model.state_dict()), SUM_OF_HUB_EXAMPLE)

    @retry(Exception, tries=3)
    def test_hub_parse_repo_info(self):
        # [FUNCTION DOC] Checks the repo spec parser: an explicit branch is returned as given, the
        # vision repo defaults to the main branch, and the example repo defaults to master. Needs
        # network for the default branch lookups.
        # If the branch is specified we just parse the input and return
        self.assertEqual(torch.hub._parse_repo_info("a/b:c"), ("a", "b", "c"))
        # For torchvision, the default branch is main
        self.assertEqual(
            torch.hub._parse_repo_info("pytorch/vision"), ("pytorch", "vision", "main")
        )
        # For the torchhub_example repo, the default branch is still master
        self.assertEqual(
            torch.hub._parse_repo_info("ailzhang/torchhub_example"),
            ("ailzhang", "torchhub_example", "master"),
        )

    @retry(Exception, tries=3)
    def test_load_commit_from_forked_repo(self):
        # [FUNCTION DOC] Loading a vision repo spec that names a commit hash (as from a forked
        # repo) must raise ValueError whose message mentions a commit from a forked repo.
        with self.assertRaisesRegex(ValueError, "If it's a commit from a forked repo"):
            torch.hub.load("pytorch/vision:4e2c216", "resnet18")

    @retry(Exception, tries=3)
    @patch("builtins.input", return_value="")
    def test_trust_repo_false_emptystring(self, patched_input):
        # [FUNCTION DOC] With the input prompt mocked to answer an empty string, loading with trust
        # disabled raises an exception saying the repository is untrusted. Verified twice: each
        # time the trusted list stays empty and the prompt was asked exactly once (mock reset in
        # between).
        with self.assertRaisesRegex(Exception, "Untrusted repository."):
            torch.hub.load(
                "ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo=False
            )
        self._assert_trusted_list_is_empty()
        patched_input.assert_called_once()

        patched_input.reset_mock()
        with self.assertRaisesRegex(Exception, "Untrusted repository."):
            torch.hub.load(
                "ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo=False
            )
        self._assert_trusted_list_is_empty()
        patched_input.assert_called_once()

    @retry(Exception, tries=3)
    @patch("builtins.input", return_value="no")
    def test_trust_repo_false_no(self, patched_input):
        # [FUNCTION DOC] With the input prompt mocked to answer no, loading with trust disabled
        # raises an untrusted repository exception, leaves the trusted list empty, and asks once.
        # Repeated a second time after resetting the mock, with the same expectations.
        with self.assertRaisesRegex(Exception, "Untrusted repository."):
            torch.hub.load(
                "ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo=False
            )
        self._assert_trusted_list_is_empty()
        patched_input.assert_called_once()

        patched_input.reset_mock()
        with self.assertRaisesRegex(Exception, "Untrusted repository."):
            torch.hub.load(
                "ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo=False
            )
        self._assert_trusted_list_is_empty()
        patched_input.assert_called_once()

    @skipIfTorchDynamo(msg="https://github.com/pytorch/pytorch/issues/159627")
    @retry(Exception, tries=3)
    @patch("builtins.input", return_value="y")
    def test_trusted_repo_false_yes(self, patched_input):
        # [FUNCTION DOC] With the prompt mocked to answer y (skipped under TorchDynamo), loading
        # with trust disabled adds the repo to the trusted list and prompts once. Then loading with
        # check mode must not prompt, and loading with trust disabled again must prompt once more.
        torch.hub.load("ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo=False)
        self._assert_in_trusted_list("ailzhang_torchhub_example")
        patched_input.assert_called_once()

        # Loading a second time with "check", we don't ask for user input
        patched_input.reset_mock()
        torch.hub.load("ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo="check")
        patched_input.assert_not_called()

        # Loading again with False, we still ask for user input
        patched_input.reset_mock()
        torch.hub.load("ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo=False)
        patched_input.assert_called_once()

    @retry(Exception, tries=3)
    @patch("builtins.input", return_value="no")
    def test_trust_repo_check_no(self, patched_input):
        # [FUNCTION DOC] With the prompt mocked to answer no, loading in check mode raises an
        # untrusted repository exception and prompts once. On a second attempt after resetting the
        # mock it again raises and prompts once. The first attempt must leave the trusted list
        # empty.
        with self.assertRaisesRegex(Exception, "Untrusted repository."):
            torch.hub.load(
                "ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo="check"
            )
        self._assert_trusted_list_is_empty()
        patched_input.assert_called_once()

        patched_input.reset_mock()
        with self.assertRaisesRegex(Exception, "Untrusted repository."):
            torch.hub.load(
                "ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo="check"
            )
        patched_input.assert_called_once()

    @skipIfTorchDynamo(msg="https://github.com/pytorch/pytorch/issues/155617")
    @retry(Exception, tries=3)
    @patch("builtins.input", return_value="y")
    def test_trust_repo_check_yes(self, patched_input):
        # [FUNCTION DOC] With the prompt mocked to answer y (skipped under TorchDynamo), loading in
        # check mode records the repo in the trusted list and prompts once; a second load in check
        # mode does not prompt.
        torch.hub.load("ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo="check")
        self._assert_in_trusted_list("ailzhang_torchhub_example")
        patched_input.assert_called_once()

        # Loading a second time with "check", we don't ask for user input
        patched_input.reset_mock()
        torch.hub.load("ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo="check")
        patched_input.assert_not_called()

    @retry(Exception, tries=3)
    def test_trust_repo_true(self):
        # [FUNCTION DOC] Loading with trust explicitly enabled records the example repo name in the
        # trusted list file.
        torch.hub.load("ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo=True)
        self._assert_in_trusted_list("ailzhang_torchhub_example")

    @retry(Exception, tries=3)
    def test_trust_repo_builtin_trusted_owners(self):
        # [FUNCTION DOC] Loading the vision repo in check mode succeeds without a prompt because
        # its owner is built in trusted, and the trusted list stays empty.
        torch.hub.load("pytorch/vision", "resnet18", trust_repo="check")
        self._assert_trusted_list_is_empty()

    @retry(Exception, tries=3)
    def test_trust_repo_legacy(self):
        # [FUNCTION DOC] Downloads the repo with trust enabled, deletes the trusted list file, then
        # loads again in check mode. Passes without a prompt since the repo already exists in the
        # cache (legacy behavior), and the trusted list must be empty afterwards.
        # We first download a repo and then delete the allowlist file
        # Then we check that the repo is indeed trusted without a prompt,
        # because it was already downloaded in the past.
        torch.hub.load("ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo=True)
        os.remove(self.trusted_list_path)

        torch.hub.load("ailzhang/torchhub_example", "mnist_zip_1_6", trust_repo="check")

        self._assert_trusted_list_is_empty()

    def _create_malicious_zip(self, filename, malicious_path):
        # [FUNCTION DOC] HELPER: Writes a new zip archive at the given file name containing a
        # single entry whose path is the given malicious path and whose content is fixed bytes.
        """Helper to create a zip file with a malicious path entry."""
        with zipfile.ZipFile(filename, "w") as zf:
            # Create an entry with malicious path
            zf.writestr(malicious_path, b"malicious content")

    def _create_legacy_malicious_zip(self, filename, malicious_path):
        # [FUNCTION DOC] HELPER: Behaves the same as the other malicious zip builder: writes a zip
        # with exactly one entry at the given path, to mimic the legacy single entry layout.
        """Helper to create a legacy-style zip file (single entry) with malicious path."""
        with zipfile.ZipFile(filename, "w") as zf:
            # Create a single entry with malicious path for legacy format
            zf.writestr(malicious_path, b"malicious content")

    def test_safe_extract_zip_blocks_directory_traversal(self):
        # [FUNCTION DOC] For a list of parent-directory traversal paths, builds a zip for each,
        # then asserts the safe extraction helper raises ValueError whose message mentions
        # directory traversal. Each path is its own subtest and the zip is removed afterwards. Runs
        # in a temp directory.
        """Test that _safe_extract_zip blocks directory traversal attacks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            malicious_zip = os.path.join(tmpdir, "malicious.zip")
            extract_dir = os.path.join(tmpdir, "extract")
            os.makedirs(extract_dir)

            # Test various directory traversal patterns
            malicious_paths = [
                "../../../etc/passwd",
                "..\\..\\..\\windows\\system32\\config\\sam",
                "../../tmp/malicious_script.py",
                "../legitimate_looking_file.txt",
                "subdir/../../../etc/shadow",
            ]

            for malicious_path in malicious_paths:
                with self.subTest(path=malicious_path):
                    self._create_malicious_zip(malicious_zip, malicious_path)

                    with zipfile.ZipFile(malicious_zip) as zf:
                        with self.assertRaises(ValueError) as cm:
                            hub._safe_extract_zip(zf, extract_dir)

                        # Verify the error message indicates directory traversal
                        self.assertIn("directory traversal", str(cm.exception))

                    os.remove(malicious_zip)

    def test_safe_extract_zip_blocks_absolute_paths(self):
        # [FUNCTION DOC] For a list of absolute paths in Unix and Windows styles, builds a zip for
        # each and asserts the safe extraction helper raises ValueError mentioning an absolute
        # path. Each path is a subtest; the zip is removed after each.
        """Test that _safe_extract_zip blocks absolute path attacks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            malicious_zip = os.path.join(tmpdir, "malicious.zip")
            extract_dir = os.path.join(tmpdir, "extract")
            os.makedirs(extract_dir)

            # Test absolute paths (both Unix and Windows style)
            malicious_paths = [
                "/etc/passwd",
                "/tmp/malicious_file",
                "C:\\Windows\\System32\\config\\sam",
                "\\etc\\passwd",
                "/home/user/.ssh/authorized_keys",
            ]

            for malicious_path in malicious_paths:
                with self.subTest(path=malicious_path):
                    self._create_malicious_zip(malicious_zip, malicious_path)

                    with zipfile.ZipFile(malicious_zip) as zf:
                        with self.assertRaises(ValueError) as cm:
                            hub._safe_extract_zip(zf, extract_dir)

                        # Verify the error message indicates absolute path
                        self.assertIn("absolute path", str(cm.exception))

                    os.remove(malicious_zip)

    def test_legacy_zip_load_zipslip_protection(self):
        # [FUNCTION DOC] Builds a single-entry zip with a traversal path and asserts that the
        # legacy zip loader raises ValueError mentioning directory traversal when pointed at it.
        """Test that _legacy_zip_load is protected against zipslip attacks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            malicious_zip = os.path.join(tmpdir, "malicious_legacy.zip")
            model_dir = os.path.join(tmpdir, "models")
            os.makedirs(model_dir)

            # Create a legacy-style malicious zip (single entry with traversal)
            malicious_path = "../../../etc/passwd"
            self._create_legacy_malicious_zip(malicious_zip, malicious_path)

            # Test that _legacy_zip_load rejects the malicious zip
            with self.assertRaises(ValueError) as cm:
                hub._legacy_zip_load(malicious_zip, model_dir, None, False)

            self.assertIn("directory traversal", str(cm.exception))

    def test_get_cache_or_reload_zipslip_protection(self):
        # [FUNCTION DOC] Builds a zip resembling a repo archive with one normal file and one
        # traversal entry, then asserts the safe extraction helper raises ValueError mentioning
        # directory traversal. Only exercises the helper directly, not the cache function itself.
        """Test that repository zip extraction is protected against zipslip attacks."""
        # This is more of an integration test that would require creating a malicious
        # GitHub repository archive, which is complex. Instead, we'll test the
        # _safe_extract_zip function directly in the context it would be used.

        with tempfile.TemporaryDirectory() as tmpdir:
            # Simulate what _get_cache_or_reload does
            cached_file = os.path.join(tmpdir, "test_repo.zip")
            hub_dir = os.path.join(tmpdir, "hub")
            os.makedirs(hub_dir)

            # Create a zip that looks like a GitHub repository but has malicious content
            malicious_entries = [
                "legitimate_repo_name-master/hubconf.py",  # Legitimate entry
                "../../../etc/passwd",  # Malicious entry
            ]

            with zipfile.ZipFile(cached_file, "w") as zf:
                zf.writestr(malicious_entries[0], "# hubconf content")
                zf.writestr(malicious_entries[1], "malicious content")

            # Test that safe extraction rejects this
            with zipfile.ZipFile(cached_file) as cached_zipfile:
                with self.assertRaises(ValueError) as cm:
                    hub._safe_extract_zip(cached_zipfile, hub_dir)

                self.assertIn("directory traversal", str(cm.exception))


if __name__ == "__main__":
    run_tests()
