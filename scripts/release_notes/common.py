# [MODULE DOC] Shared helpers for the release notes scripts. Holds the category and topic
# vocabularies (category groups for quantization, distributed, optimizer and AOT dispatcher,
# frontend categories, the full categories list and topics list), the Features record for commit
# metadata, git helpers run through the shell, GitHub GraphQL access for pull request labels,
# author and approvers, and a JSON backed cache of per-commit features. Import-time side effect:
# a GitHub token is obtained immediately, from the token environment variable or else the
# ghstack configuration file in the home directory, so importing fails if neither is available.
# Also imports requests.
import json
import locale
import os
import re
import subprocess
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path

import requests


@dataclass
class CategoryGroup:
    # [CLASS DOC] A named group of release note category labels that are treated as one: the
    # group name is the category used in the lists, and the member category labels map to it.
    name: str
    categories: list


frontend_categories = [
    "meta",
    "nn",
    "linalg",
    "cpp",
    "python",
    "complex",
    "vmap",
    "autograd",
    "build",
    "memory_format",
    "foreach",
    "dataloader",
    "sparse",
    "nested tensor",
]

pytorch_2_categories = [
    "dynamo",
    "inductor",
]

# These will all get mapped to quantization
quantization = CategoryGroup(
    name="quantization",
    categories=[
        "quantization",
        "AO frontend",
        "AO Pruning",
    ],
)

# Distributed has a number of release note sub-area labels. Each sub-area
# label is kept as its own category so it gets its own worksheet file.
distributed = CategoryGroup(
    name="distributed",
    categories=[
        "distributed",
        "distributed (c10d)",
        "distributed (composable)",
        "distributed (ddp)",
        "distributed (fsdp)",
        "distributed (fsdp2)",
        "distributed (dtensor)",
        "distributed (checkpoint)",
        "distributed (rpc)",
        "distributed (sharded)",
        "distributed (pipeline)",
        "distributed (torchelastic)",
        "distributed (symm_mem)",
    ],
)

# "optim" and "optimizer" are duplicate labels for the same area; map to one
optimizer = CategoryGroup(
    name="optim",
    categories=[
        "optim",
        "optimizer",
    ],
)

# "aot autograd" and "aotdispatcher" are the same subsystem; map to one
aotdispatcher = CategoryGroup(
    name="aotdispatcher",
    categories=[
        "aot autograd",
        "aotdispatcher",
    ],
)

categories = (
    [
        "Uncategorized",
        "lazy",
        "hub",
        "mobile",
        "jit",
        "visualization",
        "onnx",
        "caffe2",
        "amd",
        "rocm",
        "cuda",
        "cpu",
        "cudnn",
        "xla",
        "benchmark",
        "profiler",
        "performance_as_product",
        "package",
        "dispatcher",
        "releng",
        "fx",
        "code_coverage",
        "vulkan",
        "skip",
        "composability",
        # 2.0 release
        "mps",
        "intel",
        "functorch",
        "gnn",
        "distributions",
        "serialization",
    ]
    + [f"{category}_frontend" for category in frontend_categories]
    + pytorch_2_categories
    + [quantization.name]
    + distributed.categories
    + [optimizer.name]
    + [aotdispatcher.name]
)


topics = [
    "bc breaking",
    "deprecation",
    "new features",
    "improvements",
    "bug fixes",
    "performance",
    "docs",
    "devs",
    "Untopiced",
    "not user facing",
    "security",
]


Features = namedtuple(
    "Features",
    ["title", "body", "pr_number", "files_changed", "labels", "author", "accepters"],
)


def dict_to_features(dct):
    # [FUNCTION DOC] Builds a Features record from a plain mapping with the same keys; the
    # accepters list is converted to a tuple. Raises KeyError if a key is missing.
    return Features(
        title=dct["title"],
        body=dct["body"],
        pr_number=dct["pr_number"],
        files_changed=dct["files_changed"],
        labels=dct["labels"],
        author=dct["author"],
        accepters=tuple(dct["accepters"]),
    )


def features_to_dict(features):
    # [FUNCTION DOC] Converts a Features record to a plain mapping of field name to value.
    return dict(features._asdict())


def run(command):
    # [FUNCTION DOC] Runs a shell command and returns a triple of exit code, standard output and
    # standard error, both decoded with the preferred locale encoding and stripped.
    """Returns (return-code, stdout, stderr)"""
    p = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    output, err = p.communicate()
    rc = p.returncode
    enc = locale.getpreferredencoding()
    output = output.decode(enc)
    err = err.decode(enc)
    return rc, output.strip(), err.strip()


def commit_body(commit_hash):
    # [FUNCTION DOC] Returns the body text of a commit message from git, or None if git fails.
    cmd = f"git log -n 1 --pretty=format:%b {commit_hash}"
    ret, out, err = run(cmd)
    return out if ret == 0 else None


def commit_title(commit_hash):
    # [FUNCTION DOC] Returns the subject line of a commit from git, or None if git fails.
    cmd = f"git log -n 1 --pretty=format:%s {commit_hash}"
    ret, out, err = run(cmd)
    return out if ret == 0 else None


def commit_files_changed(commit_hash):
    # [FUNCTION DOC] Returns the list of file paths changed in a commit (split by lines), or
    # None if git fails. An empty change set gives a list with one empty string.
    cmd = f"git diff-tree --no-commit-id --name-only -r {commit_hash}"
    ret, out, err = run(cmd)
    return out.split("\n") if ret == 0 else None


def parse_pr_number(body, commit_hash, title):
    # [FUNCTION DOC] Extracts the pull request number from the resolved-pull-request line in a
    # commit body. Returns the first number as text, or None when absent. Prints a warning when
    # none is found (unless the title looks like a revert or submodule update) or when several
    # are found. Raises TypeError if the body is None, and AttributeError if the title is None
    # when no match exists.
    regex = r"Pull Request resolved: https://github.com/pytorch/pytorch/pull/([0-9]+)"
    matches = re.findall(regex, body)
    if len(matches) == 0:
        if "revert" not in title.lower() and "updating submodules" not in title.lower():
            print(f"[{commit_hash}: {title}] Could not parse PR number, ignoring PR")
        return None
    if len(matches) > 1:
        print(f"[{commit_hash}: {title}] Got two PR numbers, using the first one")
        return matches[0]
    return matches[0]


def get_ghstack_token():
    # [FUNCTION DOC] Reads the GitHub oauth token from the ghstack configuration file in the
    # home directory. Raises RuntimeError if no token line exists and OSError if the file is
    # missing.
    pattern = "github_oauth = (.*)"
    with open(Path("~/.ghstackrc").expanduser(), "r+") as f:
        config = f.read()
    matches = re.findall(pattern, config)
    if len(matches) == 0:
        raise RuntimeError("Can't find a github oauth token")
    return matches[0]


def get_token():
    # [FUNCTION DOC] Returns the token from the environment variable if set (printing a note),
    # otherwise the ghstack token.
    env_token = os.environ.get("GITHUB_TOKEN")
    if env_token is not None:
        print("using GITHUB_TOKEN from environment variable")
        return env_token
    else:
        return get_ghstack_token()


token = get_token()

headers = {"Authorization": f"token {token}"}


def run_query(query):
    # [FUNCTION DOC] Posts a GraphQL query to the GitHub API with the module's authorization
    # header and returns the decoded JSON on status 200. Raises a generic Exception with the
    # status and body otherwise.
    request = requests.post(
        "https://api.github.com/graphql",  # @lint-ignore
        json={"query": query},
        headers=headers,
    )
    if request.status_code == 200:
        return request.json()
    else:
        raise Exception(  # noqa: TRY002
            f"Query failed to run by returning code of {request.status_code}. {request.json()}"
        )


_ERRORS = []
_MAX_ERROR_LEN = 20


def github_data(pr_number):
    # [FUNCTION DOC] Queries GitHub for a pull request's labels, author login and up to the last
    # five approving reviewers. Returns a triple: label names, author login, and a sorted de-
    # duplicated tuple of approver logins. If the response has errors, records them in a module
    # list and returns empty labels, the text None as author and no approvers, until twenty
    # errors accumulate, after which it raises an Exception.
    query = (
        """
    {
      repository(owner: "pytorch", name: "pytorch") {
        pullRequest(number: %s ) {
          author {
            login
          }
          reviews(last: 5, states: APPROVED) {
            nodes {
              author {
                login
              }
            }
          }
          labels(first: 10) {
            edges {
              node {
                name
              }
            }
          }
        }
      }
    }
    """  # noqa: UP031
        % pr_number
    )
    query = run_query(query)
    if query.get("errors"):
        global _ERRORS
        _ERRORS.append(query.get("errors"))
        if len(_ERRORS) < _MAX_ERROR_LEN:
            return [], "None", ()
        else:
            raise Exception(  # noqa: TRY002
                f"Got {_MAX_ERROR_LEN} errors: {_ERRORS}, please check if"
                " there is something wrong"
            )
    edges = query["data"]["repository"]["pullRequest"]["labels"]["edges"]
    labels = [edge["node"]["name"] for edge in edges]
    author = query["data"]["repository"]["pullRequest"]["author"]["login"]
    nodes = query["data"]["repository"]["pullRequest"]["reviews"]["nodes"]

    # using set to dedup multiple accepts from same accepter
    accepters = {node["author"]["login"] for node in nodes}
    accepters = tuple(sorted(accepters))

    return labels, author, accepters


def get_features(commit_hash):
    # [FUNCTION DOC] Collects a Features record for a commit: title, body and changed files from
    # git, the pull request number parsed from the body, and, if a number was found, labels,
    # author and approvers from GitHub. Otherwise labels are empty, author is empty and
    # approvers are empty.
    title, body, files_changed = (
        commit_title(commit_hash),
        commit_body(commit_hash),
        commit_files_changed(commit_hash),
    )
    pr_number = parse_pr_number(body, commit_hash, title)
    labels = []
    author = ""
    accepters = ()
    if pr_number is not None:
        labels, author, accepters = github_data(pr_number)
    result = Features(title, body, pr_number, files_changed, labels, author, accepters)
    return result


_commit_data_cache = None


def get_commit_data_cache(path="results/data.json"):
    # [FUNCTION DOC] Returns the process-wide commit data cache, creating it on first use with
    # the given file path. Later calls ignore the path argument.
    global _commit_data_cache
    if _commit_data_cache is None:
        _commit_data_cache = _CommitDataCache(path)
    return _commit_data_cache


class _CommitDataCache:
    # [CLASS DOC] Persistent mapping from commit hash to Features, stored as a JSON file.
    # Missing entries are fetched with get_features on demand and the whole file is rewritten
    # after each addition.
    def __init__(self, path):
        # [FUNCTION DOC] Loads the cache file if it exists; otherwise creates its parent
        # directory (the file itself is written later).
        self.path = path
        self.data = {}
        if os.path.exists(path):
            self.data = self.read_from_disk()
        else:
            os.makedirs(Path(path).parent, exist_ok=True)

    def get(self, commit):
        # [FUNCTION DOC] Returns the Features for a commit, fetching from git and GitHub and
        # saving the cache file if not yet cached.
        if commit not in self.data.keys():
            # Fetch and cache the data
            self.data[commit] = get_features(commit)
            self.write_to_disk()
        return self.data[commit]

    def read_from_disk(self):
        # [FUNCTION DOC] Reads the JSON file and returns a mapping from commit hash to Features.
        with open(self.path) as f:
            data = json.load(f)
            data = {commit: dict_to_features(dct) for commit, dct in data.items()}
        return data

    def write_to_disk(self):
        # [FUNCTION DOC] Writes the entire cache to the JSON file, overwriting it.
        data = {commit: features._asdict() for commit, features in self.data.items()}
        with open(self.path, "w") as f:
            json.dump(data, f)
