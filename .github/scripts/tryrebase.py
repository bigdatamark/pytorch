#!/usr/bin/env python3
# [MODULE DOC]
# Merge-bot rebase command: rebases a PR (plain or ghstack) onto a target branch, force
# pushes the result to the PR head, and posts status comments on the PR. Entry point is
# main (CLI: pr_num, --branch, --dry-run); exit status 0 on a push, 1 when nothing changed.
# Depends on trymerge.GitHubPR and the gitutils/github_utils helpers. No import-time effects.

import contextlib
import os
import re
import subprocess
import sys
from collections.abc import Generator
from typing import Any

from github_utils import gh_post_pr_comment as gh_post_comment
from gitutils import get_git_remote_name, get_git_repo_dir, GitRepo
from trymerge import GitHubPR


SAME_SHA_ERROR = (
    "\n```\nAborting rebase because rebasing the branch resulted in the same sha as the target branch.\n"
    + "This usually happens because the PR has already been merged.  Please rebase locally and push.\n```"
)


def parse_args() -> Any:
    # [FUNCTION DOC] HELPER: Parses command line: positional pr_num (int), optional
    # --branch (target branch name) and --dry-run flag. Returns the argparse namespace.
    from argparse import ArgumentParser

    parser = ArgumentParser("Rebase PR into branch")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--branch", type=str)
    parser.add_argument("pr_num", type=int)
    return parser.parse_args()


def post_already_uptodate(
    pr: GitHubPR, repo: GitRepo, onto_branch: str, dry_run: bool
) -> None:
    # [FUNCTION DOC]
    # Posts a comment that the PR was already up to date. If the chosen target is not the
    # default branch and differs from it, the comment also suggests rebasing against the
    # default branch. Side effect: posts a PR comment (skipped when dry_run).
    msg = f"Tried to rebase and push PR #{pr.pr_num}, but it was already up to date."
    def_branch = pr.default_branch()
    def_branch_fcn = f"refs/remotes/{repo.remote}/{def_branch}"
    if onto_branch != def_branch_fcn and repo.rev_parse(
        def_branch_fcn
    ) != repo.rev_parse(onto_branch):
        def_branch_url = f"https://github.com/{pr.org}/{pr.project}/tree/{def_branch}"
        msg += f" Try rebasing against [{def_branch}]({def_branch_url}) by issuing:"
        msg += f"\n`@pytorchbot rebase -b {def_branch}`"

    gh_post_comment(
        pr.org,
        pr.project,
        pr.pr_num,
        msg,
        dry_run=dry_run,
    )


def rebase_onto(
    pr: GitHubPR, repo: GitRepo, onto_branch: str, dry_run: bool = False
) -> bool:
    # [FUNCTION DOC]
    # Rebases a non-ghstack PR: fetches the PR head, finds its fork point from the base
    # branch, replays only the PR's commits onto onto_branch, and force pushes to the head
    # repo (push is a dry run if dry_run). Raises an error if the result equals the target
    # sha. Posts a comment: already-up-to-date (returns False) or success (returns True).
    branch = f"pull/{pr.pr_num}/head"
    remote_url = f"https://github.com/{pr.info['headRepository']['nameWithOwner']}.git"
    refspec = f"{branch}:{pr.head_ref()}"

    repo.fetch(branch, branch)
    # Rebase only the PR's own commits. The 2-arg `git rebase <onto_branch>
    # <branch>` form replays all of onto_branch..branch, which grafts in trunk
    # commits when the PR has merged its base in and onto_branch (e.g.
    # viable/strict) lags that base. Anchoring on the fork point from the PR's
    # base branch excludes anything already on the base. See #187374.
    base_branch = f"refs/remotes/{repo.remote}/{pr.base_ref()}"
    fork_point = repo.get_merge_base(base_branch, branch)
    repo._run_git("rebase", "--onto", onto_branch, fork_point, branch)

    if repo.rev_parse(branch) == repo.rev_parse(onto_branch):
        raise Exception(SAME_SHA_ERROR)  # noqa: TRY002

    if dry_run:
        push_result = repo._run_git("push", "--dry-run", "-f", remote_url, refspec)
    else:
        push_result = repo._run_git("push", "-f", remote_url, refspec)
    if "Everything up-to-date" in push_result:
        post_already_uptodate(pr, repo, onto_branch, dry_run)
        return False
    else:
        gh_post_comment(
            pr.org,
            pr.project,
            pr.pr_num,
            f"Successfully rebased `{pr.head_ref()}` onto `{onto_branch}`, please pull locally "
            + f"before adding more changes (for example, via `git checkout {pr.head_ref()} && "
            + "git pull --rebase`)",
            dry_run=dry_run,
        )
        return True


def rebase_ghstack_onto(
    pr: GitHubPR, repo: GitRepo, onto_branch: str, dry_run: bool = False
) -> bool:
    # [FUNCTION DOC]
    # Rebases a ghstack PR: installs ghstack via pip if missing, rebases the orig branch
    # onto onto_branch, sets the global git identity to the orig commit's author, writes
    # .ghstackrc and copies GITHUB_TOKEN into OAUTH_TOKEN, then runs ghstack to push.
    # Posts a comment on each updated PR parsed from the output. Returns False for dry run
    # (unsupported) or when this PR was skipped as up to date, else True. Raises on same sha
    # or ghstack failure. Side effects: global git config, env, file write, pip install.
    if (
        subprocess.run(
            [sys.executable, "-m", "ghstack", "--help"],
            capture_output=True,
            check=False,
        ).returncode
        != 0
    ):
        subprocess.run([sys.executable, "-m", "pip", "install", "ghstack"], check=True)
    orig_ref = f"{re.sub(r'/head$', '/orig', pr.head_ref())}"

    repo.fetch(orig_ref, orig_ref)
    repo._run_git("rebase", onto_branch, orig_ref)

    if repo.rev_parse(orig_ref) == repo.rev_parse(onto_branch):
        raise Exception(SAME_SHA_ERROR)  # noqa: TRY002

    # steal the identity of the committer of the commit on the orig branch
    email = repo._run_git("log", orig_ref, "--pretty=format:%ae", "-1")
    name = repo._run_git("log", orig_ref, "--pretty=format:%an", "-1")
    repo._run_git("config", "--global", "user.email", email)
    repo._run_git("config", "--global", "user.name", name)

    os.environ["OAUTH_TOKEN"] = os.environ["GITHUB_TOKEN"]
    with open(".ghstackrc", "w+") as f:
        f.write(
            "[ghstack]\n"
            + "github_url=github.com\n"
            + "github_username=pytorchmergebot\n"
            + "remote_name=origin"
        )

    if dry_run:
        print("Don't know how to dry-run ghstack")
        return False
    else:
        ghstack_result = subprocess.run(["ghstack"], capture_output=True, check=True)
        push_result = ghstack_result.stdout.decode("utf-8")
        print(push_result)
        if ghstack_result.returncode != 0:
            print(ghstack_result.stderr.decode("utf-8"))
            raise Exception(f"\n```{push_result}```")  # noqa: TRY002
        # The contents of a successful push result should look like:
        # Summary of changes (ghstack 0.6.0)

        #  - Updated https://github.com/clee2000/random-testing-public/pull/2
        #  - Updated https://github.com/clee2000/random-testing-public/pull/1

        # Facebook employees can import your changes by running
        # (on a Facebook machine):

        #     ghimport -s https://github.com/clee2000/random-testing-public/pull/2

        # If you want to work on this diff stack on another machine:

        #     ghstack checkout https://github.com/clee2000/random-testing-public/pull/2
        org, project = repo.gh_owner_and_name()
        for line in push_result.splitlines():
            if "Updated" in line:
                pr_num = int(line.split("/")[-1])
                if pr_num != pr.pr_num:
                    gh_post_comment(
                        pr.org,
                        pr.project,
                        pr_num,
                        f"Rebased `{orig_ref}` onto `{onto_branch}` because #{pr.pr_num} was rebased, "
                        "please pull locally before adding more changes (for example, via `ghstack "
                        + f"checkout https://github.com/{org}/{project}/pull/{pr_num}`)",
                        dry_run=dry_run,
                    )
                else:
                    gh_post_comment(
                        pr.org,
                        pr.project,
                        pr_num,
                        f"Successfully rebased `{orig_ref}` onto `{onto_branch}`, please pull locally "
                        + "before adding more changes (for example, via `ghstack "
                        + f"checkout https://github.com/{org}/{project}/pull/{pr.pr_num}`)",
                        dry_run=dry_run,
                    )

        if (
            f"Skipped https://github.com/{org}/{project}/pull/{pr.pr_num}"
            in push_result
        ):
            post_already_uptodate(pr, repo, onto_branch, dry_run)
            return False
        return True


def additional_rebase_failure_info(e: Exception) -> str:
    # [FUNCTION DOC]
    # Returns an extra hint paragraph when the error text shows a git permission-denied
    # push failure (likely maintainer edits disallowed), otherwise an empty string.
    if re.search(
        r"remote: Permission to .* denied to .*\.\nfatal: unable to access", str(e)
    ):
        return (
            "\nThis is likely because the author did not allow edits from maintainers on the PR or because the "
            "repo has additional permissions settings that mergebot does not qualify."
        )
    return ""


@contextlib.contextmanager
def git_config_guard(repo: GitRepo) -> Generator[None, None, None]:
    # [FUNCTION DOC] HELPER: Context manager; records user.email and user.name, yields,
    # and on exit restores any recorded non-empty values to global git config.
    """Restores user.name and user.email global properties after context is finished"""
    user_email = repo._run_git("config", "user.email")
    user_name = repo._run_git("config", "user.name")
    try:
        yield
    finally:
        if user_email:
            repo._run_git("config", "--global", "user.email", user_email)
        if user_name:
            repo._run_git("config", "--global", "user.name", user_name)


def main() -> None:
    # [FUNCTION DOC]
    # Entry point: resolves the target branch (given or the default), posts a "rebase
    # started" comment, stops if the PR is closed, then rebases (ghstack path guarded by
    # git_config_guard) and exits with status 0 if pushed, 1 otherwise. Any exception is
    # reported as a PR comment with hints and the run URL.
    # KNOWN ISSUE: after reporting a failure it returns normally, so the exit status is 0.
    args = parse_args()
    repo = GitRepo(get_git_repo_dir(), get_git_remote_name(), debug=True)
    org, project = repo.gh_owner_and_name()

    pr = GitHubPR(org, project, args.pr_num)
    onto_branch = args.branch if args.branch else pr.default_branch()
    onto_branch = f"refs/remotes/{repo.remote}/{onto_branch}"
    onto_branch_url = (
        f"https://github.com/{org}/{project}/commit/{repo.rev_parse(onto_branch)}"
    )

    msg = f"@pytorchbot started a rebase job onto [{onto_branch}]({onto_branch_url})."
    msg += f" Check the current status [here]({os.getenv('GH_RUN_URL')})"
    gh_post_comment(org, project, args.pr_num, msg, dry_run=args.dry_run)

    if pr.is_closed():
        gh_post_comment(
            org,
            project,
            args.pr_num,
            f"PR #{args.pr_num} is closed, won't rebase",
            dry_run=args.dry_run,
        )
        return

    try:
        if pr.is_ghstack_pr():
            with git_config_guard(repo):
                rc = rebase_ghstack_onto(pr, repo, onto_branch, dry_run=args.dry_run)
        else:
            rc = rebase_onto(pr, repo, onto_branch, dry_run=args.dry_run)
        sys.exit(0 if rc else 1)

    except Exception as e:
        msg = f"Rebase failed due to {e}"
        msg += additional_rebase_failure_info(e)
        run_url = os.getenv("GH_RUN_URL")
        if run_url is not None:
            msg += f"\nRaised by {run_url}"
        gh_post_comment(org, project, args.pr_num, msg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
