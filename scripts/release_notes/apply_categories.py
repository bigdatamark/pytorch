# [MODULE DOC] Script run at import time (no functions) that carries category assignments from a
# saved categories file into the commit list. Both files are fixed relative paths under a
# results directory in the current directory. Builds a mapping from commit hash to category,
# then for each commit list row with a mapped category other than Uncategorized, overwrites its
# category, and rewrites the commit list file with a header.
# KNOWN ISSUE: both files are read with explicit field names, so their existing header line is
#   treated as a data row; the rewritten commit list then holds a duplicated header line as a
#   bogus data row each time the script runs.
# Quick script to apply categorized items to the
# base commitlist . Useful if you are refactoring any code
# but want to keep the previous data on categories

import csv

import commitlist


category_csv = "results/category_data.csv"
commitlist_csv = "results/commitlist.csv"

with open(category_csv) as category_data:
    reader = csv.DictReader(category_data, commitlist.commit_fields)
    rows = list(reader)
    category_map = {row["commit_hash"]: row["category"] for row in rows}

with open(commitlist_csv) as commitlist_data:
    reader = csv.DictReader(commitlist_data, commitlist.commit_fields)
    commitlist_rows = list(reader)

for row in commitlist_rows:
    hash = row["commit_hash"]
    if hash in category_map and category_map[hash] != "Uncategorized":
        row["category"] = category_map[hash]

with open(commitlist_csv, "w") as commitlist_write:
    writer = csv.DictWriter(commitlist_write, commitlist.commit_fields)
    writer.writeheader()
    writer.writerows(commitlist_rows)
