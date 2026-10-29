#!/usr/bin/env python3

# [MODULE DOC]
# Git helper utilities for the GitHub CI scripts: a thin GitRepo wrapper around the git command
# line, a GitCommit record, parsing of "git show --format=fuller" output, glob pattern to regex
# conversion, ghstack branch sync checks and a retry decorator. Import time side effects: none
# beyond defining a type variable and a compiled regex.

from __future__ import annotations

import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterator
from datetime import datetime
from functools import wraps
from typing import Any, cast, TypeVar


T = TypeVar("T")

RE_GITHUB_URL_MATCH = re.compile("^https://.*@?github.com/(.+)/(.+)$")


def get_git_remote_name() -> str:
    # [FUNCTION DOC] HELPER: Returns the git remote name from env var GIT_REMOTE_NAME,
    # default origin.
    return os.getenv("GIT_REMOTE_NAME", "origin")


def get_git_repo_dir() -> str:
    # [FUNCTION DOC] HELPER: Returns env var GIT_REPO_DIR, default the repo root (two levels
    # above this file).
    from pathlib import Path

    return os.getenv("GIT_REPO_DIR", str(Path(__file__).resolve().parents[2]))


def fuzzy_list_to_dict(items: list[tuple[str, str]]) -> dict[str, list[str]]:
    # [FUNCTION DOC] Converts a list of key/value pairs into a mapping from key to the list of
    # all its values, keeping duplicate keys (values stay in input order). Pure function.
    """
    Converts list to dict preserving elements with duplicate keys
    """
    rc: dict[str, list[str]] = defaultdict(list)
    for key, val in items:
        rc[key].append(val)
    return dict(rc)


def _check_output(items: list[str], encoding: str = "utf-8") -> str:
    # [FUNCTION DOC] HELPER: Runs a command (list of args) with stdout and stderr merged and
    # returns the decoded text. On non-zero exit prints stdout/stderr and raises RuntimeError
    # whose message embeds the command, exit code and captured output.
    from subprocess import CalledProcessError, check_output, STDOUT

    try:
        return check_output(items, stderr=STDOUT).decode(encoding)
    except CalledProcessError as e:
        msg = f"Command `{' '.join(e.cmd)}` returned non-zero exit code {e.returncode}"
        stdout = e.stdout.decode(encoding) if e.stdout is not None else ""
        stderr = e.stderr.decode(encoding) if e.stderr is not None else ""
        # These get swallowed up, so print them here for debugging
        print(f"stdout: \n{stdout}")
        print(f"stderr: \n{stderr}")
        if len(stderr) == 0:
            msg += f"\n```\n{stdout}```"
        else:
            msg += f"\nstdout:\n```\n{stdout}```\nstderr:\n```\n{stderr}```"
        raise RuntimeError(msg) from e


class GitCommit:
    # [CLASS DOC]
    # Plain record of one git commit: commit_hash, author, author_date, title, body and an
    # optional commit_date. The membership test checks whether text occurs in the title or body.
    commit_hash: str
    title: str
    body: str
    author: str
    author_date: datetime
    commit_date: datetime | None

    def __init__(
        self,
        commit_hash: str,
        author: str,
        author_date: datetime,
        title: str,
        body: str,
        commit_date: datetime | None = None,
    ) -> None:
        # [FUNCTION DOC] Stores the given commit fields as attributes; no validation.
        self.commit_hash = commit_hash
        self.author = author
        self.author_date = author_date
        self.commit_date = commit_date
        self.title = title
        self.body = body

    def __repr__(self) -> str:
        # [FUNCTION DOC] Returns the title followed by the commit hash in parentheses.
        return f"{self.title} ({self.commit_hash})"

    def __contains__(self, item: Any) -> bool:
        # [FUNCTION DOC] Returns True if the given text occurs in the commit body or title.
        return item in self.body or item in self.title


def parse_fuller_format(lines: str | list[str]) -> GitCommit:
    # [FUNCTION DOC] Parses git show output in fuller format with unix dates into a GitCommit.
    # Accepts a string (split on newlines) or a list of lines. Drops a Merge line at position 2
    # if present (this mutates a caller supplied list). Raises AssertionError if there are fewer
    # than 8 lines or the header lines (commit, Author, AuthorDate, Commit, CommitDate, blank)
    # are misplaced. Title is line 7; body is the remaining lines joined by newlines.
    """
    Expect commit message generated using `--format=fuller --date=unix` format, i.e.:
        commit <sha1>
        Author:     <author>
        AuthorDate: <author date>
        Commit:     <committer>
        CommitDate: <committer date>

        <title line>

        <full commit message>

    """
    if isinstance(lines, str):
        lines = lines.split("\n")
    # TODO: Handle merge commits correctly
    if len(lines) > 1 and lines[1].startswith("Merge:"):
        del lines[1]
    if len(lines) <= 7:
        raise AssertionError(
            f"Expected at least 8 lines in git log fuller format, got {len(lines)}"
        )
    if not lines[0].startswith("commit"):
        raise AssertionError(f"Expected line 0 to start with 'commit', got: {lines[0]}")
    if not lines[1].startswith("Author: "):
        raise AssertionError(
            f"Expected line 1 to start with 'Author: ', got: {lines[1]}"
        )
    if not lines[2].startswith("AuthorDate: "):
        raise AssertionError(
            f"Expected line 2 to start with 'AuthorDate: ', got: {lines[2]}"
        )
    if not lines[3].startswith("Commit: "):
        raise AssertionError(
            f"Expected line 3 to start with 'Commit: ', got: {lines[3]}"
        )
    if not lines[4].startswith("CommitDate: "):
        raise AssertionError(
            f"Expected line 4 to start with 'CommitDate: ', got: {lines[4]}"
        )
    if len(lines[5]) != 0:
        raise AssertionError(f"Expected line 5 to be empty, got: {lines[5]}")
    return GitCommit(
        commit_hash=lines[0].split()[1].strip(),
        author=lines[1].split(":", 1)[1].strip(),
        author_date=datetime.fromtimestamp(int(lines[2].split(":", 1)[1].strip())),
        commit_date=datetime.fromtimestamp(int(lines[4].split(":", 1)[1].strip())),
        title=lines[6].strip(),
        body="\n".join(lines[7:]),
    )


class GitRepo:
    # [CLASS DOC]
    # Wrapper that runs git commands via the git command line in a given repo directory. State:
    # repo_dir, remote name and a debug flag that echoes each command. Methods cover rev-list,
    # branch queries, checkout, fetch, ref lookup, patch ids, cherry-pick, revert, branch
    # diffs, push with retry, remote url parsing and commit message editing. Failures of git
    # surface as RuntimeError.
    def __init__(self, path: str, remote: str = "origin", debug: bool = False) -> None:
        # [FUNCTION DOC] Stores repo_dir, remote and debug; no git command is run.
        self.repo_dir = path
        self.remote = remote
        self.debug = debug

    def _run_git(self, *args: Any) -> str:
        # [FUNCTION DOC] HELPER: Runs git against the repo dir, echoing the command when debug
        # is set, and returns the output text. Raises RuntimeError on failure.
        if self.debug:
            print(f"+ git -C {self.repo_dir} {' '.join(args)}")
        return _check_output(["git", "-C", self.repo_dir] + list(args))

    def revlist(self, revision_range: str) -> list[str]:
        # [FUNCTION DOC] Returns the list of commit hashes in the revision range (limited to the
        # current directory path), or an empty list.
        rc = self._run_git("rev-list", revision_range, "--", ".").strip()
        return rc.split("\n") if len(rc) > 0 else []

    def branches_containing_ref(
        self, ref: str, *, include_remote: bool = True
    ) -> list[str]:
        # [FUNCTION DOC] Returns names of branches containing the ref (remote branches
        # by default, local ones if include_remote is False), stripped of whitespace, or
        # an empty list.
        rc = (
            self._run_git("branch", "--remote", "--contains", ref)
            if include_remote
            else self._run_git("branch", "--contains", ref)
        )
        return [x.strip() for x in rc.split("\n") if x.strip()] if len(rc) > 0 else []

    def current_branch(self) -> str | None:
        # [FUNCTION DOC] Returns the short name of the current branch, or None in
        # detached HEAD state.
        try:
            return self._run_git("symbolic-ref", "--short", "HEAD").strip()
        except RuntimeError:
            # we are in detached HEAD state
            return None

    def checkout(self, branch: str) -> None:
        # [FUNCTION DOC] Checks out the given branch.
        self._run_git("checkout", branch)

    def create_branch_and_checkout(self, branch: str) -> None:
        # [FUNCTION DOC] Creates the given branch from HEAD and checks it out.
        self._run_git("checkout", "-b", branch)

    def fetch(self, ref: str | None = None, branch: str | None = None) -> None:
        # [FUNCTION DOC] Fetches from the remote. With no ref and no branch fetches
        # everything; with only a ref fetches that ref; with both fetches the ref into the
        # local branch name.
        # KNOWN ISSUE: with only a branch given the ref is None and a literal None
        #     ref is passed on.
        if branch is None and ref is None:
            self._run_git("fetch", self.remote)
        elif branch is None:
            self._run_git("fetch", self.remote, ref)
        else:
            self._run_git("fetch", self.remote, f"{ref}:{branch}")

    def show_ref(self, name: str) -> str:
        # [FUNCTION DOC] Returns the hash a ref name resolves to. Raises RuntimeError if the
        # name resolves to several different hashes (ambiguous).
        refs = self._run_git("show-ref", "-s", name).strip().split("\n")
        if not all(refs[i] == refs[0] for i in range(1, len(refs))):
            raise RuntimeError(f"reference {name} is ambiguous")
        return refs[0]

    def rev_parse(self, name: str) -> str:
        # [FUNCTION DOC] Returns the verified full hash of a revision name; raises
        # RuntimeError if invalid.
        return self._run_git("rev-parse", "--verify", name).strip()

    def get_merge_base(self, from_ref: str, to_ref: str) -> str:
        # [FUNCTION DOC] Returns the merge-base hash of two refs.
        return self._run_git("merge-base", from_ref, to_ref).strip()

    def patch_id(self, ref: str | list[str]) -> list[tuple[str, str]]:
        # [FUNCTION DOC] Returns stable patch ids as a list of (patch id, commit hash) pairs for
        # one ref or a list of refs (joined with spaces and run through a shell). An empty list
        # gives an empty result.
        # KNOWN ISSUE: refs are interpolated into a shell command line without quoting.
        is_list = isinstance(ref, list)
        if is_list:
            if len(ref) == 0:
                return []
            ref = " ".join(ref)
        rc = _check_output(
            ["sh", "-c", f"git -C {self.repo_dir} show {ref}|git patch-id --stable"]
        ).strip()
        return [cast(tuple[str, str], x.split(" ", 1)) for x in rc.split("\n")]

    def commits_resolving_gh_pr(self, pr_num: int) -> list[str]:
        # [FUNCTION DOC] Returns hashes of commits whose message contains the "Pull Request
        # resolved" line for the given PR number of this repo (owner and name from
        # gh_owner_and_name), or an empty list.
        owner, name = self.gh_owner_and_name()
        msg = f"Pull Request resolved: https://github.com/{owner}/{name}/pull/{pr_num}"
        rc = self._run_git("log", "--format=%H", "--grep", msg).strip()
        return rc.split("\n") if len(rc) > 0 else []

    def get_commit(self, ref: str) -> GitCommit:
        # [FUNCTION DOC] Returns the GitCommit for a ref by parsing git show output in
        # fuller format.
        return parse_fuller_format(
            self._run_git("show", "--format=fuller", "--date=unix", "--shortstat", ref)
        )

    def cherry_pick(self, ref: str) -> None:
        # [FUNCTION DOC] Cherry-picks the ref onto the current branch, recording the
        # source commit.
        self._run_git("cherry-pick", "-x", ref)

    def revert(self, ref: str) -> None:
        # [FUNCTION DOC] Reverts the ref without opening an editor.
        self._run_git("revert", "--no-edit", ref)

    def compute_branch_diffs(
        self, from_branch: str, to_branch: str
    ) -> tuple[list[str], list[str]]:
        # [FUNCTION DOC] Returns two lists of commit hashes: commits only in from_branch and
        # commits only in to_branch since their merge base. Commits with equal patch ids on both
        # sides are removed; when the counts differ, pairs of same patch id commits are popped
        # and removed, raising RuntimeError if title or author date differ (except for a
        # hardcoded list of known commits on pytorch/pytorch). Also drops a hardcoded set of
        # excluded commits for pytorch/pytorch. Can be slow when the merge base is far away.
        """
        Returns list of commits that are missing in each other branch since their merge base
        Might be slow if merge base is between two branches is pretty far off
        """
        from_ref = self.rev_parse(from_branch)
        to_ref = self.rev_parse(to_branch)
        merge_base = self.get_merge_base(from_ref, to_ref)
        from_commits = self.revlist(f"{merge_base}..{from_ref}")
        to_commits = self.revlist(f"{merge_base}..{to_ref}")
        from_ids = fuzzy_list_to_dict(self.patch_id(from_commits))
        to_ids = fuzzy_list_to_dict(self.patch_id(to_commits))
        for patch_id in set(from_ids).intersection(set(to_ids)):
            from_values = from_ids[patch_id]
            to_values = to_ids[patch_id]
            if len(from_values) != len(to_values):
                # Eliminate duplicate commits+reverts from the list
                while len(from_values) > 0 and len(to_values) > 0:
                    frc = self.get_commit(from_values.pop())
                    toc = self.get_commit(to_values.pop())
                    # FRC branch might have PR number added to the title
                    if frc.title != toc.title or frc.author_date != toc.author_date:
                        # HACK: Same commit were merged, reverted and landed again
                        # which creates a tracking problem
                        if (
                            "pytorch/pytorch" not in self.remote_url()
                            or frc.commit_hash
                            not in {
                                "0a6a1b27a464ba5be5f587cce2ee12ab8c504dbf",
                                "6d0f4a1d545a8f161df459e8d4ccafd4b9017dbe",
                                "edf909e58f06150f7be41da2f98a3b9de3167bca",
                                "a58c6aea5a0c9f8759a4154e46f544c8b03b8db1",
                                "7106d216c29ca16a3504aa2bedad948ebcf4abc2",
                            }
                        ):
                            raise RuntimeError(
                                f"Unexpected differences between {frc} and {toc}"
                            )
                    from_commits.remove(frc.commit_hash)
                    to_commits.remove(toc.commit_hash)
                continue
            for commit in from_values:
                from_commits.remove(commit)
            for commit in to_values:
                to_commits.remove(commit)
        # Another HACK: Patch-id is not stable for commits with binary files or for big changes across commits
        # I.e. cherry-picking those from one branch into another will change patchid
        if "pytorch/pytorch" in self.remote_url():
            for excluded_commit in {
                "8e09e20c1dafcdbdb45c2d1574da68a32e54a3a5",
                "5f37e5c2a39c3acb776756a17730b865f0953432",
                "b5222584e6d6990c6585981a936defd1af14c0ba",
                "84d9a2e42d5ed30ec3b8b4140c38dd83abbce88d",
                "f211ec90a6cdc8a2a5795478b5b5c8d7d7896f7e",
            }:
                if excluded_commit in from_commits:
                    from_commits.remove(excluded_commit)

        return (from_commits, to_commits)

    def cherry_pick_commits(self, from_branch: str, to_branch: str) -> None:
        # [FUNCTION DOC] Cherry-picks, oldest first, every commit missing from to_branch onto
        # it, then returns to the original branch. Raises AssertionError when HEAD is detached.
        # Prints progress. Does not push.
        orig_branch = self.current_branch()
        if orig_branch is None:
            raise AssertionError("Must be on a branch to cherry pick commits")
        self.checkout(to_branch)
        from_commits, to_commits = self.compute_branch_diffs(from_branch, to_branch)
        if len(from_commits) == 0:
            print("Nothing to do")
            self.checkout(orig_branch)
            return
        for commit in reversed(from_commits):
            print(f"Cherry picking commit {commit}")
            self.cherry_pick(commit)
        self.checkout(orig_branch)

    def push(self, branch: str, dry_run: bool, retry: int = 3) -> None:
        # [FUNCTION DOC] Pushes the branch to the remote (a dry run if dry_run). On RuntimeError
        # it prints, fetches and rebases onto the remote branch, then retries.
        # KNOWN ISSUE: the loop has no break, so a successful push is repeated for every retry.
        # KNOWN ISSUE: after the final failed attempt no error is raised to the caller.
        for cnt in range(retry):
            try:
                if dry_run:
                    self._run_git("push", "--dry-run", self.remote, branch)
                else:
                    self._run_git("push", self.remote, branch)
            except RuntimeError as e:
                print(f"{cnt} push attempt failed with {e}")
                self.fetch()
                self._run_git("rebase", f"{self.remote}/{branch}")

    def head_hash(self) -> str:
        # [FUNCTION DOC] Returns the hash of HEAD.
        return self._run_git("show-ref", "--hash", "HEAD").strip()

    def remote_url(self) -> str:
        # [FUNCTION DOC] Returns the url of the remote as printed by git, including the trailing
        # newline (not stripped).
        return self._run_git("remote", "get-url", self.remote)

    def gh_owner_and_name(self) -> tuple[str, str]:
        # [FUNCTION DOC] Returns (owner, name) parsed from the url in env var GIT_REMOTE_URL,
        # or the remote url if unset. Raises RuntimeError if the url does not match the
        # github https form.
        url = os.getenv("GIT_REMOTE_URL", None)
        if url is None:
            url = self.remote_url()
        rc = RE_GITHUB_URL_MATCH.match(url)
        if rc is None:
            raise RuntimeError(f"Unexpected url format {url}")
        return cast(tuple[str, str], rc.groups())

    def commit_message(self, ref: str) -> str:
        # [FUNCTION DOC] Returns the full message of the commit at ref.
        return self._run_git("log", "-1", "--format=%B", ref)

    def amend_commit_message(self, msg: str) -> None:
        # [FUNCTION DOC] Amends HEAD with the given message (rewrites history; no push).
        self._run_git("commit", "--amend", "-m", msg)

    def diff(self, from_ref: str, to_ref: str | None = None) -> str:
        # [FUNCTION DOC] Returns the diff text of a single commit (ref against its parent) when
        # to_ref is None, otherwise the diff between from_ref and to_ref.
        if to_ref is None:
            return self._run_git("diff", f"{from_ref}^!")
        return self._run_git("diff", f"{from_ref}..{to_ref}")


def clone_repo(username: str, password: str, org: str, project: str) -> GitRepo:
    # [FUNCTION DOC] Clones the github project org/project with the given credentials embedded
    # in the url into a new temporary directory and returns a GitRepo for it. Raises
    # RuntimeError if the clone fails. Side effects: creates the temp directory (never removed)
    # and network access.
    path = tempfile.mkdtemp()
    _check_output(
        [
            "git",
            "clone",
            f"https://{username}:{password}@github.com/{org}/{project}",
            path,
        ]
    ).strip()
    return GitRepo(path=path)


class PeekableIterator(Iterator[str]):
    # [CLASS DOC]
    # Character iterator over a string that supports looking at the next character without
    # consuming it. State: the string and the index of the last consumed character.
    def __init__(self, val: str) -> None:
        # [FUNCTION DOC] Stores the string and sets the position before the first character.
        self._val = val
        self._idx = -1

    def peek(self) -> str | None:
        # [FUNCTION DOC] Returns the next character without consuming it, or None at the end.
        if self._idx + 1 >= len(self._val):
            return None
        return self._val[self._idx + 1]

    def __iter__(self) -> PeekableIterator:
        # [FUNCTION DOC] Returns itself.
        return self

    def __next__(self) -> str:
        # [FUNCTION DOC] Consumes and returns the next character; raises
        # StopIteration at the end.
        rc = self.peek()
        if rc is None:
            raise StopIteration
        self._idx += 1
        return rc


def patterns_to_regex(allowed_patterns: list[str]) -> Any:
    # [FUNCTION DOC] Converts a list of glob-like patterns into one compiled regular expression
    # that matches any of them. A single star matches any characters except slash, a double star
    # matches any characters; dot and plus are escaped; the question mark is not translated.
    # Raises AssertionError if a pattern contains braces, parens, brackets or backslash.
    # KNOWN ISSUE: the docstring says a question mark matches one character, but it is copied
    #     into the regex unchanged, where it makes the previous character optional.
    """
    pattern is glob-like, i.e. the only special sequences it has are:
      - ? - matches single character
      - * - matches any non-folder separator characters or no character
      - ** - matches any characters or no character
      Assuming that patterns are free of braces and backslashes
      the only character that needs to be escaped are dot and plus
    """
    rc = "("
    for idx, pattern in enumerate(allowed_patterns):
        if idx > 0:
            rc += "|"
        pattern_ = PeekableIterator(pattern)
        if any(c in pattern for c in "{}()[]\\"):
            raise AssertionError(
                f"Pattern contains invalid characters (braces/parens/brackets/backslash): {pattern}"
            )
        for c in pattern_:
            if c == ".":
                rc += "\\."
            elif c == "+":
                rc += "\\+"
            elif c == "*":
                if pattern_.peek() == "*":
                    next(pattern_)
                    rc += ".*"
                else:
                    rc += "[^/]*"
            else:
                rc += c
    rc += ")"
    return re.compile(rc)


def _shasum(value: str) -> str:
    # [FUNCTION DOC] HELPER: Returns the hex sha256 digest of the utf-8 encoded string.
    import hashlib

    m = hashlib.sha256()
    m.update(value.encode("utf-8"))
    return m.hexdigest()


def is_commit_hash(ref: str) -> bool:
    # [FUNCTION DOC] Returns True if the ref parses as a hexadecimal number, else False.
    # KNOWN ISSUE: any hex-looking name (for example a branch called dead) counts as a hash.
    "True if ref is hexadecimal number, else false"
    try:
        int(ref, 16)
    except ValueError:
        return False
    return True


def are_ghstack_branches_in_sync(
    repo: GitRepo, head_ref: str, base_ref: str | None = None
) -> bool:
    # [FUNCTION DOC] Returns True if the diff of the ghstack orig branch against its parent
    # equals the diff between base and head. head_ref ends with /head; orig and base names are
    # derived from it (base_ref may be passed, and is used as is if it looks like a commit
    # hash). Runs git diffs.
    """Checks that diff between base and head is the same as diff between orig and its parent"""
    orig_ref = re.sub(r"/head$", "/orig", head_ref)
    if base_ref is None:
        base_ref = re.sub(r"/head$", "/base", head_ref)
    orig_diff_sha = _shasum(repo.diff(f"{repo.remote}/{orig_ref}"))
    head_diff_sha = _shasum(
        repo.diff(
            base_ref if is_commit_hash(base_ref) else f"{repo.remote}/{base_ref}",
            f"{repo.remote}/{head_ref}",
        )
    )
    return orig_diff_sha == head_diff_sha


def retries_decorator(
    rc: Any = None, num_retries: int = 3
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    # [FUNCTION DOC] Returns a decorator that calls the wrapped function up to num_retries
    # times, printing a message on each exception and swallowing it. If all attempts fail it
    # returns the value rc.
    def decorator(f: Callable[..., T]) -> Callable[..., T]:
        # [FUNCTION DOC] HELPER: Wraps f with the retry loop; preserves f's metadata.
        @wraps(f)
        def wrapper(*args: list[Any], **kwargs: dict[str, Any]) -> T:
            # [FUNCTION DOC] HELPER: Calls f; if every attempt fails returns rc.
            for idx in range(num_retries):
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    print(
                        f'Attempt {idx} of {num_retries} to call {f.__name__} failed with "{e}"'
                    )
            return cast(T, rc)

        return wrapper

    return decorator
