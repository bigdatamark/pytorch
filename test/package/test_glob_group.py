# Owner(s): ["oncall: package/deploy"]
# [MODULE DOC] Tests for GlobGroup, the dotted-name (or custom separator) glob matcher used by
# the package exporter: single star, double star, partial components, custom separators, invalid
# patterns and exclusion lists. Falls back to a non-relative import of the shared base test
# case. Run as a script, it starts the test runner.

from collections.abc import Iterable

from torch.package import GlobGroup
from torch.testing._internal.common_utils import run_tests


try:
    from .common import PackageTestCase
except ImportError:
    # Support the case where we run this file directly.
    from common import PackageTestCase


class TestGlobGroup(PackageTestCase):
    # [CLASS DOC] Test case with two assertion helpers and one test per glob feature.
    def assertMatchesGlob(self, glob: GlobGroup, candidates: Iterable[str]):
        # [FUNCTION DOC] HELPER: Asserts the glob group matches every candidate string in
        # the iterable.
        for candidate in candidates:
            self.assertTrue(glob.matches(candidate))

    def assertNotMatchesGlob(self, glob: GlobGroup, candidates: Iterable[str]):
        # [FUNCTION DOC] HELPER: Asserts the glob group matches none of the candidate strings.
        for candidate in candidates:
            self.assertFalse(glob.matches(candidate))

    def test_one_star(self):
        # [FUNCTION DOC] A single star pattern after a package prefix matches exactly one
        # further component, but not the bare prefix, deeper names or a different prefix.
        glob_group = GlobGroup("torch.*")
        self.assertMatchesGlob(glob_group, ["torch.foo", "torch.bar"])
        self.assertNotMatchesGlob(glob_group, ["tor.foo", "torch.foo.bar", "torch"])

    def test_one_star_middle(self):
        # [FUNCTION DOC] A single star between two components matches exactly one component
        # there, and rejects missing, extra or misplaced components.
        glob_group = GlobGroup("foo.*.bar")
        self.assertMatchesGlob(glob_group, ["foo.q.bar", "foo.foo.bar"])
        self.assertNotMatchesGlob(
            glob_group,
            [
                "foo.bar",
                "foo.foo",
                "outer.foo.inner.bar",
                "foo.q.bar.more",
                "foo.one.two.bar",
            ],
        )

    def test_one_star_partial(self):
        # [FUNCTION DOC] A star inside a component matches any (possibly empty) run of
        # characters within that component, but the other literal parts must still match.
        glob_group = GlobGroup("fo*.bar")  # codespell:ignore
        self.assertMatchesGlob(
            glob_group,
            ["fo.bar", "foo.bar", "foobar.bar"],  # codespell:ignore
        )
        self.assertNotMatchesGlob(glob_group, ["oij.bar", "f.bar", "foo"])

    def test_one_star_multiple_in_component(self):
        # [FUNCTION DOC] With a slash separator, several stars within one component of a path
        # pattern each match arbitrary characters inside that component.
        glob_group = GlobGroup("foo/a*.htm*", separator="/")
        self.assertMatchesGlob(glob_group, ["foo/a.html", "foo/a.htm", "foo/abc.html"])

    def test_one_star_partial_extension(self):
        # [FUNCTION DOC] With a slash separator, a star before a file extension matches names
        # (including an empty one) in that directory only, not in nested or sibling directories.
        glob_group = GlobGroup("foo/*.txt", separator="/")
        self.assertMatchesGlob(
            glob_group, ["foo/hello.txt", "foo/goodbye.txt", "foo/.txt"]
        )
        self.assertNotMatchesGlob(
            glob_group, ["foo/bar/hello.txt", "bar/foo/hello.txt"]
        )

    def test_two_star(self):
        # [FUNCTION DOC] A trailing double star matches the prefix itself and any depth below
        # it, but not unrelated names that merely contain or extend the prefix text.
        glob_group = GlobGroup("torch.**")
        self.assertMatchesGlob(
            glob_group, ["torch.foo", "torch.bar", "torch.foo.bar", "torch"]
        )
        self.assertNotMatchesGlob(glob_group, ["what.torch", "torchvision"])

    def test_two_star_end(self):
        # [FUNCTION DOC] A leading double star matches the bare suffix component and any dotted
        # prefix of it, but not names where the suffix is only part of a component.
        glob_group = GlobGroup("**.torch")
        self.assertMatchesGlob(glob_group, ["torch", "bar.torch"])
        self.assertNotMatchesGlob(glob_group, ["visiontorch"])

    def test_two_star_middle(self):
        # [FUNCTION DOC] A double star between two components matches zero or more components
        # there, and rejects names without the separators or with trailing extra components.
        glob_group = GlobGroup("foo.**.baz")
        self.assertMatchesGlob(
            glob_group, ["foo.baz", "foo.bar.baz", "foo.bar1.bar2.baz"]
        )
        self.assertNotMatchesGlob(glob_group, ["foobaz", "foo.bar.baz.z"])

    def test_two_star_multiple(self):
        # [FUNCTION DOC] With a slash separator, a pattern with two double stars and a final
        # single star matches paths containing the middle component and a text file, and rejects
        # paths lacking it.
        glob_group = GlobGroup("**/bar/**/*.txt", separator="/")
        self.assertMatchesGlob(
            glob_group, ["bar/baz.txt", "a/bar/b.txt", "bar/foo/c.txt"]
        )
        self.assertNotMatchesGlob(glob_group, ["baz.txt", "a/b.txt"])

    def test_raw_two_star(self):
        # [FUNCTION DOC] A bare double star matches any non-empty name, including dotted ones,
        # but not the empty string.
        glob_group = GlobGroup("**")
        self.assertMatchesGlob(glob_group, ["bar", "foo.bar", "ab.c.d.e"])
        self.assertNotMatchesGlob(glob_group, [""])

    def test_invalid_raw(self):
        # [FUNCTION DOC] A double star attached to other characters within one component must
        # raise ValueError at construction.
        with self.assertRaises(ValueError):
            GlobGroup("a.**b")

    def test_exclude(self):
        # [FUNCTION DOC] An exclude pattern removes matching names from an include pattern;
        # names that only share text with the excluded name remain matched.
        glob_group = GlobGroup("torch.**", exclude=["torch.**.foo"])
        self.assertMatchesGlob(
            glob_group,
            ["torch", "torch.bar", "torch.barfoo"],
        )
        self.assertNotMatchesGlob(
            glob_group,
            ["torch.foo", "torch.some.foo"],
        )

    def test_exclude_from_all(self):
        # [FUNCTION DOC] Excluding two package prefixes from a match-everything pattern still
        # matches other names but rejects anything under the excluded prefixes.
        glob_group = GlobGroup("**", exclude=["foo.**", "bar.**"])
        self.assertMatchesGlob(glob_group, ["a", "hello", "anything.really"])
        self.assertNotMatchesGlob(glob_group, ["foo.bar", "foo.bar.baz"])

    def test_list_include_exclude(self):
        # [FUNCTION DOC] Constructs a group from a list of include patterns and a list of
        # excludes, and checks the exact-name excludes win while similar names stay matched.
        glob_group = GlobGroup(["foo", "bar.**"], exclude=["bar.baz", "bar.qux"])
        self.assertMatchesGlob(glob_group, ["foo", "bar.other", "bar.bazother"])
        self.assertNotMatchesGlob(glob_group, ["bar.baz", "bar.qux"])


if __name__ == "__main__":
    run_tests()
