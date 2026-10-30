# [MODULE DOC] Script that generates old-format lite-interpreter model files for operator upgrader
# tests. It holds a table mapping fixture module instances (star-imported from fixtures_src) to
# the operator each must contain, and helpers that find existing fixtures, read a saved model's
# version and operator list, and export missing models. Import-time side effects: configures root
# logging to standard output and sets this module's logger to debug. Running as a script calls
# main, which writes model files under the fixtures directory.
import io
import logging
import sys
import zipfile
from pathlib import Path

# Use asterisk symbol so developer doesn't need to import here when they add tests for upgraders.
from test.jit.fixtures_srcs.fixtures_src import *  # noqa: F403
from typing import Set

import torch
from torch.jit.mobile import _export_operator_list, _load_for_lite_interpreter


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

"""
This file is used to generate model for test operator change. Please refer to
https://github.com/pytorch/rfcs/blob/master/RFC-0017-PyTorch-Operator-Versioning.md for more details.

A systematic workflow to change operator is needed to ensure
Backwards Compatibility (BC) / Forwards Compatibility (FC) for operator changes. For BC-breaking operator change,
an upgrader is needed. Here is the flow to properly land a BC-breaking operator change.

1. Write an upgrader in caffe2/torch/csrc/jit/operator_upgraders/upgraders_entry.cpp file. The softly enforced
naming format is <operator_name>_<operator_overload>_<start>_<end>. For example, the below example means that
div.Tensor at version from 0 to 3 needs to be replaced by this upgrader.

```
/*
div_Tensor_0_3 is added for a change of operator div in pr xxxxxxx.
Create date: 12/02/2021
Expire date: 06/02/2022
*/
     {"div_Tensor_0_3", R"SCRIPT(
def div_Tensor_0_3(self: Tensor, other: Tensor) -> Tensor:
  if (self.is_floating_point() or other.is_floating_point()):
    return self.true_divide(other)
  return self.divide(other, rounding_mode='trunc')
)SCRIPT"},
```

2. In caffe2/torch/csrc/jit/operator_upgraders/version_map.h, add changes like below.
You will need to make sure that the entry is SORTED according to the version bump number.
```
    {"div.Tensor",
      {{4,
        "div_Tensor_0_3",
        "aten::div.Tensor(Tensor self, Tensor other) -> Tensor"}}},
```

3. After rebuild PyTorch, run the following command and it will auto generate a change to
fbcode/caffe2/torch/csrc/jit/mobile/upgrader_mobile.cpp

```
python pytorch/torchgen/operator_versions/gen_mobile_upgraders.py
```

4. Generate the test to cover upgrader.

4.1 Switch the commit before the operator change, and add a module in
`test/jit/fixtures_srcs/fixtures_src.py`. The reason why switching to commit is that,
an old model with the old operator before the change is needed to ensure the upgrader
is working as expected. In `test/jit/fixtures_srcs/generate_models.py`, add the module and
it's corresponding changed operator like following
```
ALL_MODULES = {
    TestVersionedDivTensorExampleV7(): "aten::div.Tensor",
}
```
This module should includes the changed operator. If the operator isn't covered in the model,
the model export process in step 4.2 will fail.

4.2 Export the model to `test/jit/fixtures` by running
```
python /Users/chenlai/pytorch/test/jit/fixtures_src/generate_models.py
```

4.3 In `test/jit/test_save_load_for_op_version.py`, add a test to cover the old models and
ensure the result is equivalent between current module and old module + upgrader.

4.4 Save all change in 4.1, 4.2 and 4.3, as well as previous changes made in step 1, 2, 3.
Submit a pr

"""

"""
A map of test modules and it's according changed operator
key: test module
value: changed operator
"""
ALL_MODULES = {
    TestVersionedDivTensorExampleV7(): "aten::div.Tensor",
    TestVersionedLinspaceV7(): "aten::linspace",
    TestVersionedLinspaceOutV7(): "aten::linspace.out",
    TestVersionedLogspaceV8(): "aten::logspace",
    TestVersionedLogspaceOutV8(): "aten::logspace.out",
    TestVersionedGeluV9(): "aten::gelu",
    TestVersionedGeluOutV9(): "aten::gelu.out",
    TestVersionedRandomV10(): "aten::random_.from",
    TestVersionedRandomFuncV10(): "aten::random.from",
    TestVersionedRandomOutV10(): "aten::random.from_out",
}

"""
Get the path to `test/jit/fixtures`, where all test models for operator changes
(upgrader/downgrader) are stored
"""


def get_fixtures_path() -> Path:
    # [FUNCTION DOC] Returns the path of the fixtures directory (test/jit/fixtures), computed as
    # three directory levels above this file's directory, joined with the test and jit parts.
    pytorch_dir = Path(__file__).resolve().parents[3]
    fixtures_path = pytorch_dir / "test" / "jit" / "fixtures"
    return fixtures_path


"""
Get all models' name in `test/jit/fixtures`
"""


def get_all_models(model_directory_path: Path) -> Set[str]:
    # [FUNCTION DOC] Recursively globs every entry under the given directory and returns the set
    # of file stems (names without extension) of the regular files. Reads only.
    files_in_fixtures = model_directory_path.glob("**/*")
    all_models_from_fixtures = [
        fixture.stem for fixture in files_in_fixtures if fixture.is_file()
    ]
    return set(all_models_from_fixtures)


"""
Check if a given model already exist in `test/jit/fixtures`
"""


def model_exist(model_file_name: str, all_models: Set[str]) -> bool:
    # [FUNCTION DOC] Returns whether the given model name is a member of the given set of names.
    return model_file_name in all_models


"""
Get the operator list given a module
"""


def get_operator_list(script_module: torch) -> Set[str]:
    # [FUNCTION DOC] Serializes the scripted module for the lite interpreter into an in-memory
    # buffer, reloads it as a mobile module and returns the exported set of operator names.
    buffer = io.BytesIO(script_module._save_to_buffer_for_lite_interpreter())
    buffer.seek(0)
    mobile_module = _load_for_lite_interpreter(buffer)
    operator_list = _export_operator_list(mobile_module)
    return operator_list


"""
Get the output model operator version, given a module
"""


def get_output_model_version(script_module: torch.nn.Module) -> int:
    # [FUNCTION DOC] Saves the scripted module into an in-memory archive and returns the integer
    # model version read from the version entry. If that entry is absent from the archive root,
    # falls back to the version entry under the data subdirectory.
    buffer = io.BytesIO()
    torch.jit.save(script_module, buffer)
    buffer.seek(0)
    zipped_model = zipfile.ZipFile(buffer)
    try:
        version = int(zipped_model.read("archive/version").decode("utf-8"))
        return version
    except KeyError:
        version = int(zipped_model.read("archive/.data/version").decode("utf-8"))
        return version


"""
Loop through all test modules. If the corresponding model doesn't exist in
`test/jit/fixtures`, generate one. For the following reason, a model won't be exported:

1. The test module doesn't cover the changed operator. For example, test_versioned_div_tensor_example_v4
is supposed to test the operator aten::div.Tensor. If the model doesn't include this operator, it will fail.
The error message includes the actual operator list from the model.

2. The output model version is not the same as expected version. For example, test_versioned_div_tensor_example_v4
is used to test an operator change aten::div.Tensor, and the operator version will be bumped to v5. This script is
supposed to run before the operator change (before the commit to make the change). If the actual model version is v5,
likely this script is running with the commit to make the change.

3. The model already exists in `test/jit/fixtures`.

"""


def generate_models(model_directory_path: Path):
    # [FUNCTION DOC] For each registered test module, derives the model file name by converting
    # the class name to lower snake case, and skips it if a fixture with that name already exists.
    # Otherwise scripts the module and checks the model version against the runtime maximum
    # operator version (skipping with an error log if it is not below that maximum plus one), and
    # checks the expected operator appears in the module's operator list (skipping with an error
    # log otherwise). Passing models are saved for the lite interpreter into the given directory
    # with the lite-model extension and logged.
    # KNOWN ISSUE: a table key that is not a torch module is only logged as an error; the loop
    # does not skip it, so the following scripting step then fails for it.
    all_models = get_all_models(model_directory_path)
    for a_module, expect_operator in ALL_MODULES.items():
        # For example: TestVersionedDivTensorExampleV7
        torch_module_name = type(a_module).__name__

        if not isinstance(a_module, torch.nn.Module):
            logger.error(
                "The module %s "
                "is not a torch.nn.module instance. "
                "Please ensure it's a subclass of torch.nn.module in fixtures_src.py"
                "and it's registered as an instance in ALL_MODULES in generated_models.py",
                torch_module_name,
            )

        # The corresponding model name is: test_versioned_div_tensor_example_v4
        model_name = "".join(
            [
                "_" + char.lower() if char.isupper() else char
                for char in torch_module_name
            ]
        ).lstrip("_")

        # Some models may not compile anymore, so skip the ones
        # that already has pt file for them.
        logger.info("Processing %s", torch_module_name)
        if model_exist(model_name, all_models):
            logger.info("Model %s already exists, skipping", model_name)
            continue

        script_module = torch.jit.script(a_module)
        actual_model_version = get_output_model_version(script_module)

        current_operator_version = torch._C._get_max_operator_version()
        if actual_model_version >= current_operator_version + 1:
            logger.error(
                "Actual model version %s "
                "is equal or larger than %s + 1. "
                "Please run the script before the commit to change operator.",
                actual_model_version,
                current_operator_version,
            )
            continue

        actual_operator_list = get_operator_list(script_module)
        if expect_operator not in actual_operator_list:
            logger.error(
                "The model includes operator: %s, "
                "however it doesn't cover the operator %s."
                "Please ensure the output model includes the tested operator.",
                actual_operator_list,
                expect_operator,
            )
            continue

        export_model_path = str(model_directory_path / (str(model_name) + ".ptl"))
        script_module._save_for_lite_interpreter(export_model_path)
        logger.info(
            "Generating model %s and it's save to %s", model_name, export_model_path
        )


def main() -> None:
    # [FUNCTION DOC] Entry point: resolves the fixtures directory and generates any missing
    # models.
    model_directory_path = get_fixtures_path()
    generate_models(model_directory_path)


if __name__ == "__main__":
    main()
