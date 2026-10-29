# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.
# [MODULE DOC] Vendored PEP 440 version parsing and comparison. Provides parse, the Version class,
# the InvalidVersion error, the unanchored VERSION_PATTERN regular expression string, and private
# helpers that normalize pre/post/dev/local segments and compute the comparison key. The only
# import-time work is defining types and compiling a regular expression.
"""
.. testsetup::

    from packaging.version import parse, Version
"""

import itertools
import re
from typing import Any, Callable, NamedTuple, Optional, SupportsInt, Tuple, Union

from ._structures import Infinity, InfinityType, NegativeInfinity, NegativeInfinityType

__all__ = ["VERSION_PATTERN", "parse", "Version", "InvalidVersion"]

LocalType = Tuple[Union[int, str], ...]

CmpPrePostDevType = Union[InfinityType, NegativeInfinityType, Tuple[str, int]]
CmpLocalType = Union[
    NegativeInfinityType,
    Tuple[Union[Tuple[int, str], Tuple[NegativeInfinityType, Union[int, str]]], ...],
]
CmpKey = Tuple[
    int,
    Tuple[int, ...],
    CmpPrePostDevType,
    CmpPrePostDevType,
    CmpPrePostDevType,
    CmpLocalType,
]
VersionComparisonMethod = Callable[[CmpKey, CmpKey], bool]


class _Version(NamedTuple):
    epoch: int
    release: Tuple[int, ...]
    dev: Optional[Tuple[str, int]]
    pre: Optional[Tuple[str, int]]
    post: Optional[Tuple[str, int]]
    local: Optional[LocalType]


def parse(version: str) -> "Version":
    # [FUNCTION DOC] Parses a version string into a Version object; raises InvalidVersion if the string is not a valid PEP 440 version.
    """Parse the given version string.

    >>> parse('1.0.dev1')
    <Version('1.0.dev1')>

    :param version: The version string to parse.
    :raises InvalidVersion: When the version string is not a valid version.
    """
    return Version(version)


class InvalidVersion(ValueError):
    # [CLASS DOC] Error type raised when a version string cannot be parsed; carries only the message.
    """Raised when a version string is not a valid version.

    >>> Version("invalid")
    Traceback (most recent call last):
        ...
    packaging.version.InvalidVersion: Invalid version: 'invalid'
    """


class _BaseVersion:
    # [CLASS DOC] HELPER: Base class providing hashing and rich comparison between versions using the sort key attribute set by subclasses; comparisons with non-version objects return NotImplemented.
    _key: Tuple[Any, ...]

    def __hash__(self) -> int:
        # [FUNCTION DOC] HELPER: Hash of the comparison key.
        return hash(self._key)

    # Please keep the duplicated `isinstance` check
    # in the six comparisons hereunder
    # unless you find a way to avoid adding overhead function calls.
    def __lt__(self, other: "_BaseVersion") -> bool:
        # [FUNCTION DOC] HELPER: Less-than on comparison keys; NotImplemented for non-version operands.
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key < other._key

    def __le__(self, other: "_BaseVersion") -> bool:
        # [FUNCTION DOC] HELPER: Less-or-equal on comparison keys; NotImplemented for non-version operands.
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key <= other._key

    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] HELPER: Equality on comparison keys; NotImplemented for non-version operands.
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key == other._key

    def __ge__(self, other: "_BaseVersion") -> bool:
        # [FUNCTION DOC] HELPER: Greater-or-equal on comparison keys; NotImplemented for non-version operands.
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key >= other._key

    def __gt__(self, other: "_BaseVersion") -> bool:
        # [FUNCTION DOC] HELPER: Greater-than on comparison keys; NotImplemented for non-version operands.
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key > other._key

    def __ne__(self, other: object) -> bool:
        # [FUNCTION DOC] HELPER: Inequality on comparison keys; NotImplemented for non-version operands.
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key != other._key


# Deliberately not anchored to the start and end of the string, to make it
# easier for 3rd party code to reuse
_VERSION_PATTERN = r"""
    v?
    (?:
        (?:(?P<epoch>[0-9]+)!)?                           # epoch
        (?P<release>[0-9]+(?:\.[0-9]+)*)                  # release segment
        (?P<pre>                                          # pre-release
            [-_\.]?
            (?P<pre_l>alpha|a|beta|b|preview|pre|c|rc)
            [-_\.]?
            (?P<pre_n>[0-9]+)?
        )?
        (?P<post>                                         # post release
            (?:-(?P<post_n1>[0-9]+))
            |
            (?:
                [-_\.]?
                (?P<post_l>post|rev|r)
                [-_\.]?
                (?P<post_n2>[0-9]+)?
            )
        )?
        (?P<dev>                                          # dev release
            [-_\.]?
            (?P<dev_l>dev)
            [-_\.]?
            (?P<dev_n>[0-9]+)?
        )?
    )
    (?:\+(?P<local>[a-z0-9]+(?:[-_\.][a-z0-9]+)*))?       # local version
"""

VERSION_PATTERN = _VERSION_PATTERN
"""
A string containing the regular expression used to match a valid version.

The pattern is not anchored at either end, and is intended for embedding in larger
expressions (for example, matching a version number as part of a file name). The
regular expression should be compiled with the ``re.VERBOSE`` and ``re.IGNORECASE``
flags set.

:meta hide-value:
"""


class Version(_BaseVersion):
    # [CLASS DOC] PEP 440 version: parses a string into epoch, release tuple, pre, post, dev and local
    # parts (stored in an immutable named tuple) and a normalized sort key. Supports ordering, hashing,
    # string round-trip, and read-only property accessors. Invalid input raises InvalidVersion.
    """This class abstracts handling of a project's versions.

    A :class:`Version` instance is comparison aware and can be compared and
    sorted using the standard Python interfaces.

    >>> v1 = Version("1.0a5")
    >>> v2 = Version("1.0")
    >>> v1
    <Version('1.0a5')>
    >>> v2
    <Version('1.0')>
    >>> v1 < v2
    True
    >>> v1 == v2
    False
    >>> v1 > v2
    False
    >>> v1 >= v2
    False
    >>> v1 <= v2
    True
    """

    _regex = re.compile(r"^\s*" + VERSION_PATTERN + r"\s*$", re.VERBOSE | re.IGNORECASE)
    _key: CmpKey

    def __init__(self, version: str) -> None:
        # [FUNCTION DOC] Matches the string (surrounding whitespace allowed, case-insensitive) against the
        # version regex; raises InvalidVersion on mismatch. Otherwise stores the parsed parts and builds the comparison key.
        """Initialize a Version object.

        :param version:
            The string representation of a version which will be parsed and normalized
            before use.
        :raises InvalidVersion:
            If the ``version`` does not conform to PEP 440 in any way then this
            exception will be raised.
        """

        # Validate the version and parse it into pieces
        match = self._regex.search(version)
        if not match:
            raise InvalidVersion(f"Invalid version: '{version}'")

        # Store the parsed out pieces of the version
        self._version = _Version(
            epoch=int(match.group("epoch")) if match.group("epoch") else 0,
            release=tuple(int(i) for i in match.group("release").split(".")),
            pre=_parse_letter_version(match.group("pre_l"), match.group("pre_n")),
            post=_parse_letter_version(
                match.group("post_l"), match.group("post_n1") or match.group("post_n2")
            ),
            dev=_parse_letter_version(match.group("dev_l"), match.group("dev_n")),
            local=_parse_local_version(match.group("local")),
        )

        # Generate a key which will be used for sorting
        self._key = _cmpkey(
            self._version.epoch,
            self._version.release,
            self._version.pre,
            self._version.post,
            self._version.dev,
            self._version.local,
        )

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns the angle-bracket Version representation wrapping the normalized string.
        """A representation of the Version that shows all internal state.

        >>> Version('1.0.0')
        <Version('1.0.0')>
        """
        return f"<Version('{self}')>"

    def __str__(self) -> str:
        # [FUNCTION DOC] Builds the normalized version string: optional epoch with bang, dotted release, pre
        # letters and number, dot-post number, dot-dev number, and plus-local suffix.
        """A string representation of the version that can be rounded-tripped.

        >>> str(Version("1.0a5"))
        '1.0a5'
        """
        parts = []

        # Epoch
        if self.epoch != 0:
            parts.append(f"{self.epoch}!")

        # Release segment
        parts.append(".".join(str(x) for x in self.release))

        # Pre-release
        if self.pre is not None:
            parts.append("".join(str(x) for x in self.pre))

        # Post-release
        if self.post is not None:
            parts.append(f".post{self.post}")

        # Development release
        if self.dev is not None:
            parts.append(f".dev{self.dev}")

        # Local version segment
        if self.local is not None:
            parts.append(f"+{self.local}")

        return "".join(parts)

    @property
    def epoch(self) -> int:
        # [FUNCTION DOC] Property: epoch number (0 if absent).
        """The epoch of the version.

        >>> Version("2.0.0").epoch
        0
        >>> Version("1!2.0.0").epoch
        1
        """
        return self._version.epoch

    @property
    def release(self) -> Tuple[int, ...]:
        # [FUNCTION DOC] Property: tuple of integers of the release segment, trailing zeros kept.
        """The components of the "release" segment of the version.

        >>> Version("1.2.3").release
        (1, 2, 3)
        >>> Version("2.0.0").release
        (2, 0, 0)
        >>> Version("1!2.0.0.post0").release
        (2, 0, 0)

        Includes trailing zeroes but not the epoch or any pre-release / development /
        post-release suffixes.
        """
        return self._version.release

    @property
    def pre(self) -> Optional[Tuple[str, int]]:
        # [FUNCTION DOC] Property: pre-release as a (letter, number) pair, or None.
        """The pre-release segment of the version.

        >>> print(Version("1.2.3").pre)
        None
        >>> Version("1.2.3a1").pre
        ('a', 1)
        >>> Version("1.2.3b1").pre
        ('b', 1)
        >>> Version("1.2.3rc1").pre
        ('rc', 1)
        """
        return self._version.pre

    @property
    def post(self) -> Optional[int]:
        # [FUNCTION DOC] Property: post-release number, or None if absent.
        """The post-release number of the version.

        >>> print(Version("1.2.3").post)
        None
        >>> Version("1.2.3.post1").post
        1
        """
        return self._version.post[1] if self._version.post else None

    @property
    def dev(self) -> Optional[int]:
        # [FUNCTION DOC] Property: development release number, or None if absent.
        """The development number of the version.

        >>> print(Version("1.2.3").dev)
        None
        >>> Version("1.2.3.dev1").dev
        1
        """
        return self._version.dev[1] if self._version.dev else None

    @property
    def local(self) -> Optional[str]:
        # [FUNCTION DOC] Property: local version segments joined by dots, or None if absent.
        """The local version segment of the version.

        >>> print(Version("1.2.3").local)
        None
        >>> Version("1.2.3+abc").local
        'abc'
        """
        if self._version.local:
            return ".".join(str(x) for x in self._version.local)
        else:
            return None

    @property
    def public(self) -> str:
        # [FUNCTION DOC] Property: normalized version string with any local segment removed.
        """The public portion of the version.

        >>> Version("1.2.3").public
        '1.2.3'
        >>> Version("1.2.3+abc").public
        '1.2.3'
        >>> Version("1.2.3+abc.dev1").public
        '1.2.3'
        """
        return str(self).split("+", 1)[0]

    @property
    def base_version(self) -> str:
        # [FUNCTION DOC] Property: epoch (if nonzero) plus release segment only, no pre/post/dev/local.
        """The "base version" of the version.

        >>> Version("1.2.3").base_version
        '1.2.3'
        >>> Version("1.2.3+abc").base_version
        '1.2.3'
        >>> Version("1!1.2.3+abc.dev1").base_version
        '1!1.2.3'

        The "base version" is the public version of the project without any pre or post
        release markers.
        """
        parts = []

        # Epoch
        if self.epoch != 0:
            parts.append(f"{self.epoch}!")

        # Release segment
        parts.append(".".join(str(x) for x in self.release))

        return "".join(parts)

    @property
    def is_prerelease(self) -> bool:
        # [FUNCTION DOC] Property: true if the version has a dev or pre-release part.
        """Whether this version is a pre-release.

        >>> Version("1.2.3").is_prerelease
        False
        >>> Version("1.2.3a1").is_prerelease
        True
        >>> Version("1.2.3b1").is_prerelease
        True
        >>> Version("1.2.3rc1").is_prerelease
        True
        >>> Version("1.2.3dev1").is_prerelease
        True
        """
        return self.dev is not None or self.pre is not None

    @property
    def is_postrelease(self) -> bool:
        # [FUNCTION DOC] Property: true if the version has a post-release part.
        """Whether this version is a post-release.

        >>> Version("1.2.3").is_postrelease
        False
        >>> Version("1.2.3.post1").is_postrelease
        True
        """
        return self.post is not None

    @property
    def is_devrelease(self) -> bool:
        # [FUNCTION DOC] Property: true if the version has a dev part.
        """Whether this version is a development release.

        >>> Version("1.2.3").is_devrelease
        False
        >>> Version("1.2.3.dev1").is_devrelease
        True
        """
        return self.dev is not None

    @property
    def major(self) -> int:
        # [FUNCTION DOC] Property: first release number, or 0 if missing.
        """The first item of :attr:`release` or ``0`` if unavailable.

        >>> Version("1.2.3").major
        1
        """
        return self.release[0] if len(self.release) >= 1 else 0

    @property
    def minor(self) -> int:
        # [FUNCTION DOC] Property: second release number, or 0 if missing.
        """The second item of :attr:`release` or ``0`` if unavailable.

        >>> Version("1.2.3").minor
        2
        >>> Version("1").minor
        0
        """
        return self.release[1] if len(self.release) >= 2 else 0

    @property
    def micro(self) -> int:
        # [FUNCTION DOC] Property: third release number, or 0 if missing.
        """The third item of :attr:`release` or ``0`` if unavailable.

        >>> Version("1.2.3").micro
        3
        >>> Version("1").micro
        0
        """
        return self.release[2] if len(self.release) >= 3 else 0


def _parse_letter_version(
    letter: Optional[str], number: Union[str, bytes, SupportsInt, None]
) -> Optional[Tuple[str, int]]:
    # [FUNCTION DOC] HELPER: Normalizes a pre/post/dev (letter, number) pair. With a letter: missing number
    # becomes 0, letter is lowercased and alternate spellings are canonicalized (alpha to a, beta to b,
    # c/pre/preview to rc, rev/r to post). With only a number: treated as an implicit post release.
    # Returns (letter, int) or None when neither is given. Non-numeric number raises ValueError.

    if letter:
        # We consider there to be an implicit 0 in a pre-release if there is
        # not a numeral associated with it.
        if number is None:
            number = 0

        # We normalize any letters to their lower case form
        letter = letter.lower()

        # We consider some words to be alternate spellings of other words and
        # in those cases we want to normalize the spellings to our preferred
        # spelling.
        if letter == "alpha":
            letter = "a"
        elif letter == "beta":
            letter = "b"
        elif letter in ["c", "pre", "preview"]:
            letter = "rc"
        elif letter in ["rev", "r"]:
            letter = "post"

        return letter, int(number)
    if not letter and number:
        # We assume if we are given a number, but we are not given a letter
        # then this is using the implicit post release syntax (e.g. 1.0-1)
        letter = "post"

        return letter, int(number)

    return None


_local_version_separators = re.compile(r"[\._-]")


def _parse_local_version(local: Optional[str]) -> Optional[LocalType]:
    # [FUNCTION DOC] HELPER: Splits a local version string on dot, underscore or dash; numeric parts become
    # integers, others are lowercased. Returns a tuple, or None if input is None.
    """
    Takes a string like abc.1.twelve and turns it into ("abc", 1, "twelve").
    """
    if local is not None:
        return tuple(
            part.lower() if not part.isdigit() else int(part)
            for part in _local_version_separators.split(local)
        )
    return None


def _cmpkey(
    epoch: int,
    release: Tuple[int, ...],
    pre: Optional[Tuple[str, int]],
    post: Optional[Tuple[str, int]],
    dev: Optional[Tuple[str, int]],
    local: Optional[LocalType],
) -> CmpKey:
    # [FUNCTION DOC] HELPER: Builds the sortable key (epoch, release with trailing zeros stripped, pre, post,
    # dev, local) using infinity sentinels so that dev-only sorts before pre-releases, final releases
    # sort after pre-releases, absent post sorts before present, absent dev sorts after present, and
    # local segments follow PEP 440 ordering (absent local first, numeric above alphanumeric).

    # When we compare a release version, we want to compare it with all of the
    # trailing zeros removed. So we'll use a reverse the list, drop all the now
    # leading zeros until we come to something non zero, then take the rest
    # re-reverse it back into the correct order and make it a tuple and use
    # that for our sorting key.
    _release = tuple(
        reversed(list(itertools.dropwhile(lambda x: x == 0, reversed(release))))
    )

    # We need to "trick" the sorting algorithm to put 1.0.dev0 before 1.0a0.
    # We'll do this by abusing the pre segment, but we _only_ want to do this
    # if there is not a pre or a post segment. If we have one of those then
    # the normal sorting rules will handle this case correctly.
    if pre is None and post is None and dev is not None:
        _pre: CmpPrePostDevType = NegativeInfinity
    # Versions without a pre-release (except as noted above) should sort after
    # those with one.
    elif pre is None:
        _pre = Infinity
    else:
        _pre = pre

    # Versions without a post segment should sort before those with one.
    if post is None:
        _post: CmpPrePostDevType = NegativeInfinity

    else:
        _post = post

    # Versions without a development segment should sort after those with one.
    if dev is None:
        _dev: CmpPrePostDevType = Infinity

    else:
        _dev = dev

    if local is None:
        # Versions without a local segment should sort before those with one.
        _local: CmpLocalType = NegativeInfinity
    else:
        # Versions with a local segment need that segment parsed to implement
        # the sorting rules in PEP440.
        # - Alpha numeric segments sort before numeric segments
        # - Alpha numeric segments sort lexicographically
        # - Numeric segments sort numerically
        # - Shorter versions sort before longer versions when the prefixes
        #   match exactly
        _local = tuple(
            (i, "") if isinstance(i, int) else (NegativeInfinity, i) for i in local
        )

    return epoch, _release, _pre, _post, _dev, _local
