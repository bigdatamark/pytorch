#!/usr/bin/env python3
# [MODULE DOC] Command line tool that creates or force-removes git worktrees of the repository,
# cloning every submodule (recursively) from the local checkout's object store instead of the
# network. Main contents: git query helpers, recursive submodule cloning, the create and remove
# subcommands and main. Running it as a script invokes main; no import-time side effects.
"""Create or remove git worktrees with submodules cloned locally.

This avoids fetching submodules from remote, which is slow for large repos
like PyTorch. Instead, each submodule is cloned directly from the local
checkout's git object store, so no network access is needed.

Usage:
    python tools/create_worktree.py                  # pytorch-worktree-1
    python tools/create_worktree.py my-worktree      # custom name
    python tools/create_worktree.py remove my-worktree  # force-remove
"""

import argparse
import configparser
import subprocess
import sys
from pathlib import Path


def get_repo_root() -> Path:
    # [FUNCTION DOC] Returns the top-level directory of the current git repository by running
    # git. Raises CalledProcessError if not inside a repository.
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def get_existing_worktrees(repo_root: Path) -> list[str]:
    # [FUNCTION DOC] Returns the directory names (last path component only) of all worktrees
    # that git lists for the repository, including the main one. Raises CalledProcessError if
    # git fails.
    result = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
        cwd=repo_root,
    )
    names = []
    for line in result.stdout.splitlines():
        if line.startswith("worktree "):
            names.append(Path(line.split(" ", 1)[1]).name)
    return names


def next_worktree_name(repo_root: Path) -> str:
    # [FUNCTION DOC] Returns the first unused name of the form pytorch-worktree-N, counting
    # from 1, that does not match an existing worktree directory name.
    existing = get_existing_worktrees(repo_root)
    n = 1
    while True:
        name = f"pytorch-worktree-{n}"
        if name not in existing:
            return name
        n += 1


def parse_gitmodules(root: Path) -> list[dict[str, str]]:
    # [FUNCTION DOC] Reads the gitmodules file under root and returns a list of mappings with
    # the path and url of each declared submodule; empty list if the file does not exist.
    gitmodules = root / ".gitmodules"
    if not gitmodules.exists():
        return []
    config = configparser.ConfigParser()
    config.read(gitmodules)
    modules = []
    for section in config.sections():
        if section.startswith('submodule "'):
            path = config.get(section, "path")
            url = config.get(section, "url")
            modules.append({"path": path, "url": url})
    return modules


def get_submodule_commit(parent_repo: Path, submodule_path: str) -> str | None:
    # [FUNCTION DOC] Returns the commit hash that the parent repository records at HEAD for the
    # submodule path (third field of the tree listing), or None if git fails or no entry exists.
    """Get the commit hash a parent repo expects for a submodule."""
    result = subprocess.run(
        ["git", "ls-tree", "HEAD", submodule_path],
        capture_output=True,
        text=True,
        cwd=parent_repo,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    # format: <mode> <type> <hash>\t<path>
    parts = result.stdout.strip().split()
    return parts[2] if len(parts) >= 3 else None


def resolve_git_dir(submodule_worktree: Path) -> Path | None:
    # [FUNCTION DOC] Returns the real git directory of a submodule checkout: the git entry
    # itself if it is a directory, or the resolved target of a gitdir pointer file. Returns None
    # if there is no git entry or the pointer target is missing or malformed.
    """Resolve the actual git object directory for a submodule.

    Submodules can have .git as either a directory (standalone clone) or a
    file containing 'gitdir: <path>' (gitlink). This resolves to the actual
    git directory in both cases.
    """
    git_path = submodule_worktree / ".git"
    if not git_path.exists():
        return None
    if git_path.is_dir():
        return git_path
    # gitlink: read the target
    content = git_path.read_text().strip()
    if content.startswith("gitdir: "):
        target = content[len("gitdir: ") :]
        resolved = (submodule_worktree / target).resolve()
        if resolved.exists():
            return resolved
    return None


def clone_submodule_recursive(
    worktree_root: Path,
    source_root: Path,
    submodule_path: str,
    url: str,
    depth: int = 0,
) -> None:
    # [FUNCTION DOC] Clones one submodule from the source checkout's local git directory into
    # the worktree, checks out the commit recorded by the source parent repo, resets origin to
    # the real url, runs submodule init, then recurses into nested submodules found in the new
    # clone. Skips (with a message) submodules not initialized in the source. depth only controls
    # print indentation. Failures of nested clones are printed as warnings to stderr; a failure
    # of the top-level git commands raises CalledProcessError. Side effects: removes an empty
    # placeholder directory, runs git, prints progress.
    prefix = "  " * depth
    source_sub = source_root / submodule_path
    worktree_sub = worktree_root / submodule_path

    git_dir = resolve_git_dir(source_sub)
    if git_dir is None:
        print(f"{prefix}  skipping {submodule_path} (not initialized in source)")
        return

    commit = get_submodule_commit(source_root, submodule_path)

    print(f"{prefix}  cloning {submodule_path}...")

    # The worktree checkout creates empty dirs for submodule paths; remove them
    # so git clone can create the directory itself.
    if worktree_sub.exists():
        try:
            worktree_sub.rmdir()
        except OSError:
            pass

    # Clone from the resolved git directory — works for both standalone .git
    # dirs and gitlinks pointing into .git/modules/. No network needed.
    subprocess.run(
        ["git", "clone", str(git_dir), str(worktree_sub)],
        check=True,
        capture_output=True,
        text=True,
    )

    # Checkout the exact commit the parent repo expects.
    if commit:
        subprocess.run(
            ["git", "checkout", "--quiet", commit],
            cwd=worktree_sub,
            check=True,
            capture_output=True,
            text=True,
        )

    # Point the remote back to the real URL so future fetches work.
    subprocess.run(
        ["git", "remote", "set-url", "origin", url],
        cwd=worktree_sub,
        check=True,
        capture_output=True,
        text=True,
    )

    # Register nested submodules in this repo's config so that
    # `git submodule status --recursive` recognizes them.
    subprocess.run(
        ["git", "submodule", "init"],
        cwd=worktree_sub,
        capture_output=True,
        text=True,
    )

    # Recurse into nested submodules
    nested = parse_gitmodules(worktree_sub)
    for mod in nested:
        try:
            clone_submodule_recursive(
                worktree_sub, source_sub, mod["path"], mod["url"], depth + 1
            )
        except subprocess.CalledProcessError as e:
            print(
                f"{prefix}    WARNING: nested {mod['path']}: {e}",
                file=sys.stderr,
            )


def cmd_create(args: argparse.Namespace) -> None:
    # [FUNCTION DOC] Implements the create subcommand: adds a git worktree for args.commit under
    # args.parent_dir (default: parent of the repo root) named args.name (default: next free
    # pytorch-worktree-N), then clones all submodules from the local checkout. A failure to
    # clone a submodule prints a warning to stderr and continues; failure to add the worktree
    # raises CalledProcessError.
    repo_root = get_repo_root()
    parent_dir = Path(args.parent_dir) if args.parent_dir else repo_root.parent
    name = args.name or next_worktree_name(repo_root)
    worktree_path = parent_dir / name

    print(f"Creating worktree at {worktree_path}")

    cmd = ["git", "worktree", "add", str(worktree_path), args.commit]
    subprocess.run(cmd, check=True, cwd=repo_root)

    modules = parse_gitmodules(repo_root)
    if modules:
        print(f"Cloning {len(modules)} submodules from local checkout...")
        for mod in modules:
            try:
                clone_submodule_recursive(
                    worktree_path, repo_root, mod["path"], mod["url"]
                )
            except subprocess.CalledProcessError as e:
                print(
                    f"  WARNING: failed to clone {mod['path']}: {e}",
                    file=sys.stderr,
                )
                if e.stderr:
                    print(f"    {e.stderr.strip()}", file=sys.stderr)

    print(f"\nWorktree ready at {worktree_path}")


def cmd_remove(args: argparse.Namespace) -> None:
    # [FUNCTION DOC] Implements the remove subcommand: force-removes the worktree args.name
    # located under args.parent_dir (default: parent of the repo root). Raises
    # CalledProcessError if git fails.
    repo_root = get_repo_root()
    parent_dir = Path(args.parent_dir) if args.parent_dir else repo_root.parent
    worktree_path = parent_dir / args.name

    print(f"Removing worktree at {worktree_path}")
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(worktree_path)],
        check=True,
        cwd=repo_root,
    )
    print("Done.")


def main() -> None:
    # [FUNCTION DOC] Command line entry point: parses the create and remove subcommands, and
    # defaults to create with default options when no subcommand is given, then dispatches.
    # KNOWN ISSUE: a bare worktree name as the first argument (as in the usage text) is not a
    # valid subcommand and makes argparse exit with an error.
    parser = argparse.ArgumentParser(
        description="Create or remove git worktrees with locally-cloned submodules."
    )
    subparsers = parser.add_subparsers(dest="command")

    # "create" subcommand (also the default when no subcommand given)
    create_parser = subparsers.add_parser(
        "create", help="Create a new worktree (default)"
    )
    create_parser.add_argument(
        "name",
        nargs="?",
        default=None,
        help="Worktree directory name (default: pytorch-worktree-$n)",
    )
    create_parser.add_argument(
        "--commit",
        default="HEAD",
        help="Commit/branch to check out in the worktree (default: HEAD)",
    )
    create_parser.add_argument(
        "--parent-dir",
        default=None,
        help="Parent directory for the worktree (default: parent of repo root)",
    )

    # "remove" subcommand
    remove_parser = subparsers.add_parser("remove", help="Force-remove a worktree")
    remove_parser.add_argument(
        "name",
        help="Worktree directory name to remove",
    )
    remove_parser.add_argument(
        "--parent-dir",
        default=None,
        help="Parent directory of the worktree (default: parent of repo root)",
    )

    args = parser.parse_args()

    # Default to "create" when no subcommand is given
    if args.command is None:
        args.command = "create"
        args.name = None
        args.commit = "HEAD"
        args.parent_dir = None

    if args.command == "create":
        cmd_create(args)
    elif args.command == "remove":
        cmd_remove(args)


if __name__ == "__main__":
    main()
