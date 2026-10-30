# [MODULE DOC] Script that generates the ExportDB rst documentation pages from the
# registered export example cases: one page section per example (source code plus the
# exported graph or the error text), one page per tag, and an index page, written
# under the generated/exportdb directory of the source tree. Defines path constants at
# import time; generation only runs when executed as main.
import inspect
import os
import re
from pathlib import Path

import torch
import torch._dynamo as torchdynamo
from torch._export.db.case import ExportCase
from torch._export.db.examples import all_examples
from torch.export import export


PWD = Path(__file__).absolute().parent
ROOT = Path(__file__).absolute().parents[3]
SOURCE = ROOT / "source"
EXPORTDB_SOURCE = SOURCE / "generated" / "exportdb"


def generate_example_rst(example_case: ExportCase):
    # [FUNCTION DOC] Builds and returns the rst text for one export example case: title,
    # tag and support level note, the original source file text (split at the rewrite
    # decorator marker), the export call, and the result of strict export (graph text with
    # file comment lines removed) or a one line message for unsupported, assertion or
    # runtime errors. If a rewrite decorator is present, appends the rewritten source.
    # Reads the source file of the model. Raises AssertionError if the source splits into
    # more than two parts.
    """
    Generates the .rst files for all the examples in db/examples/
    """

    model = example_case.model

    tags = ", ".join(f":doc:`{tag} <{tag}>`" for tag in example_case.tags)

    source_file = (
        inspect.getfile(model.__class__)
        if isinstance(model, torch.nn.Module)
        else inspect.getfile(model)
    )
    with open(source_file) as file:
        source_code = file.read()
    source_code = source_code.replace("\n", "\n    ")
    splitted_source_code = re.split(r"@export_rewrite_case.*\n", source_code)

    if len(splitted_source_code) not in {1, 2}:
        raise AssertionError(
            f"more than one @export_rewrite_case decorator in {source_code}"
        )

    more_arguments = ""
    if example_case.example_kwargs:
        more_arguments += ", example_kwargs"
    if example_case.dynamic_shapes:
        more_arguments += ", dynamic_shapes=dynamic_shapes"

    # Generate contents of the .rst file
    title = f"{example_case.name}"
    doc_contents = f"""{title}
{"^" * (len(title))}

.. note::

    Tags: {tags}

    Support Level: {example_case.support_level.name}

Original source code:

.. code-block:: python

    {splitted_source_code[0]}

    torch.export.export(model, example_args{more_arguments})

Result:

.. code-block::

"""

    # Get resulting graph from dynamo trace
    try:
        exported_program = export(
            model,
            example_case.example_args,
            example_case.example_kwargs,
            dynamic_shapes=example_case.dynamic_shapes,
            strict=True,
        )
        graph_output = str(exported_program)
        graph_output = re.sub(r"        # File(.|\n)*?\n", "", graph_output)
        graph_output = graph_output.replace("\n", "\n    ")
        output = f"    {graph_output}"
    except torchdynamo.exc.Unsupported as e:
        output = "    Unsupported: " + str(e).split("\n")[0]
    except AssertionError as e:
        output = "    AssertionError: " + str(e).split("\n")[0]
    except RuntimeError as e:
        output = "    RuntimeError: " + str(e).split("\n")[0]

    doc_contents += output + "\n"

    if len(splitted_source_code) == 2:
        doc_contents += f"""\n
You can rewrite the example above to something like the following:

.. code-block:: python

{splitted_source_code[1]}

"""

    return doc_contents


def generate_index_rst(example_cases, tag_to_modules, support_level_to_modules):
    # [FUNCTION DOC] Writes index.rst into the exportdb output directory. Builds a section
    # per support level (title-cased name) containing the example texts, a toctree listing
    # the tags, and reads the intro blurb from blurb.txt next to this script. The
    # example_cases argument is unused. Returns nothing; overwrites the file.
    """
    Generates the index.rst file
    """

    support_contents = ""
    for k, v in support_level_to_modules.items():
        support_level = k.name.lower().replace("_", " ").title()
        module_contents = "\n\n".join(v)
        support_contents += f"""
{support_level}
{"-" * (len(support_level))}

{module_contents}
"""

    tag_names = "\n    ".join(t for t in tag_to_modules)

    with open(os.path.join(PWD, "blurb.txt")) as file:
        blurb = file.read()

    # Generate contents of the .rst file
    doc_contents = f""".. _torch.export_db:

ExportDB
========

{blurb}

.. toctree::
    :maxdepth: 1
    :caption: Tags

    {tag_names}

{support_contents}
"""

    with open(os.path.join(EXPORTDB_SOURCE, "index.rst"), "w") as f:
        f.write(doc_contents)


def generate_tag_rst(tag_to_modules):
    # [FUNCTION DOC] For each tag in the mapping from tag to list of example rst texts,
    # writes a file named after the tag in the exportdb output directory, with a title and
    # the joined example texts in which heading underlines of three or more equals signs
    # are turned into dashes (to demote them one level). Overwrites existing files.
    # Returns nothing.
    """
    For each tag that shows up in each ExportCase.tag, generate an .rst file
    containing all the examples that have that tag.
    """

    for tag, modules_rst in tag_to_modules.items():
        doc_contents = f"{tag}\n{'=' * (len(tag) + 4)}\n"
        full_modules_rst = "\n\n".join(modules_rst)
        full_modules_rst = re.sub(
            r"={3,}", lambda match: "-" * len(match.group()), full_modules_rst
        )
        doc_contents += full_modules_rst

        with open(os.path.join(EXPORTDB_SOURCE, f"{tag}.rst"), "w") as f:
            f.write(doc_contents)


def generate_rst():
    # [FUNCTION DOC] Entry point. Creates the exportdb output directory if missing, loads
    # all example cases, generates the rst for each, groups the texts by tag and by
    # support level, then writes the per-tag pages and the index page. Returns nothing.
    if not os.path.exists(EXPORTDB_SOURCE):
        os.makedirs(EXPORTDB_SOURCE)

    example_cases = all_examples()
    tag_to_modules = {}
    support_level_to_modules = {}
    for example_case in example_cases.values():
        doc_contents = generate_example_rst(example_case)

        for tag in example_case.tags:
            tag_to_modules.setdefault(tag, []).append(doc_contents)

        support_level_to_modules.setdefault(example_case.support_level, []).append(
            doc_contents
        )

    generate_tag_rst(tag_to_modules)
    generate_index_rst(example_cases, tag_to_modules, support_level_to_modules)


if __name__ == "__main__":
    generate_rst()
