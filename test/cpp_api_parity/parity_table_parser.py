# [MODULE DOC] Parser for the C++/Python API parity tracker Markdown file.
# Defines the ParityStatus named tuple (has_impl_parity, has_doc_parity
# booleans) and one function that turns the tracker tables into a nested
# mapping. The bare string after the tuple is only a format description, not a
# real docstring. No import-time side effects.
from collections import namedtuple


ParityStatus = namedtuple("ParityStatus", ["has_impl_parity", "has_doc_parity"])

"""
This function expects the parity tracker Markdown file to have the following format:

```
## package1_name

API | Implementation Parity | Doc Parity
------------- | ------------- | -------------
API_Name|No|No
...

## package2_name

API | Implementation Parity | Doc Parity
------------- | ------------- | -------------
API_Name|No|No
...
```

The returned dict has the following format:

```
Dict[package_name]
    -> Dict[api_name]
        -> ParityStatus
```
"""


def parse_parity_tracker_table(file_path):
    # [FUNCTION DOC] Reads the tracker file at file_path and returns a mapping
    # from package name to a mapping from API name to ParityStatus. Text is split
    # on the two-hash marker; content before the first marker is ignored. For each
    # package the first non-blank line is the name, the next two lines (table
    # header and separator) are skipped, and every remaining non-blank line is a
    # three-column row separated by vertical bars. Raises RuntimeError for a
    # duplicated package name or a Yes/No value that is neither; a row with a
    # wrong column count raises ValueError. KNOWN ISSUE: splitting on the two-hash
    # marker also breaks on deeper headings or any text containing it, and a
    # package with fewer than three lines yields no rows silently.
    def parse_parity_choice(str):
        # [FUNCTION DOC] HELPER: Converts the text Yes to true and No to false. Raises
        # RuntimeError for any other value. The parameter name shadows the builtin
        # str.
        if str in ["Yes", "No"]:
            return str == "Yes"
        else:
            raise RuntimeError(
                f'{str} is not a supported parity choice. The valid choices are "Yes" and "No".'
            )

    parity_tracker_dict = {}

    with open(file_path) as f:
        all_text = f.read()
        packages = all_text.split("##")
        for package in packages[1:]:
            lines = [line.strip() for line in package.split("\n") if line.strip() != ""]
            package_name = lines[0]
            if package_name in parity_tracker_dict:
                raise RuntimeError(
                    f"Duplicated package name `{package_name}` found in {file_path}"
                )
            else:
                parity_tracker_dict[package_name] = {}
            for api_status in lines[3:]:
                api_name, has_impl_parity_str, has_doc_parity_str = (
                    x.strip() for x in api_status.split("|")
                )
                parity_tracker_dict[package_name][api_name] = ParityStatus(
                    has_impl_parity=parse_parity_choice(has_impl_parity_str),
                    has_doc_parity=parse_parity_choice(has_doc_parity_str),
                )

    return parity_tracker_dict
