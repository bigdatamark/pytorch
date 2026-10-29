# [MODULE DOC] Unit tests for the vLLM build and test helpers in cli.lib.core.vllm.vllm_build
# and vllm_test. Covers VllmBuildParameters validation (env var driven, with patched path and
# image existence checks), the docker build command text produced by VllmBuildRunner, the
# runner's clone/prepare/build sequence, and the environment variables set by VllmTestRunner.
# Tests patch os.environ and collaborators; no import-time side effects beyond imports.
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cli.lib.core.vllm.vllm_build as vllm_build
import cli.lib.core.vllm.vllm_test as vllm_test


_VLLM_BUILD_MODULE = "cli.lib.core.vllm.vllm_build"


class TestVllmBuildParameters(unittest.TestCase):
    # [CLASS DOC] Tests for VllmBuildParameters: successful normalization of paths and image
    # from patched environment variables, and error cases for missing wheels path, missing
    # local base image, missing Dockerfile and an empty output directory.
    @patch(f"{_VLLM_BUILD_MODULE}.local_image_exists", return_value=True)
    @patch(f"{_VLLM_BUILD_MODULE}.is_path_exist", return_value=True)
    @patch(
        "cli.lib.common.envs_helper.env_path_optional",
        side_effect=lambda name, default=None, resolve=True: {
            "DOCKERFILE_PATH": Path("/abs/vllm/Dockerfile"),
            "TORCH_WHEELS_PATH": Path("/abs/dist"),
            "OUTPUT_DIR": Path("/abs/shared"),
        }.get(name, Path(default) if default is not None else None),
    )
    @patch.dict(
        os.environ,
        {
            "USE_TORCH_WHEEL": "1",
            "USE_LOCAL_BASE_IMAGE": "1",
            "USE_LOCAL_DOCKERFILE": "1",
            "BASE_IMAGE": "my/image:tag",
            "DOCKERFILE_PATH": "vllm/Dockerfile",
            "TORCH_WHEELS_PATH": "dist",
            "OUTPUT_DIR": "shared",
        },
        clear=True,
    )
    def test_params_success_normalizes_and_validates(
        self, mock_env_path, mock_is_path, mock_local_img
    ):
        # [FUNCTION DOC] With all use-local flags on and existence checks patched to succeed,
        # builds the parameters and asserts the wheels path, Dockerfile path and output dir are
        # the normalized absolute paths and the base image comes from the environment.
        params = vllm_build.VllmBuildParameters()
        self.assertEqual(params.torch_whls_path, Path("/abs/dist"))
        self.assertEqual(params.dockerfile_path, Path("/abs/vllm/Dockerfile"))
        self.assertEqual(params.output_dir, Path("/abs/shared"))
        self.assertEqual(params.base_image, "my/image:tag")

    @patch(f"{_VLLM_BUILD_MODULE}.is_path_exist", return_value=False)
    @patch.dict(
        os.environ, {"USE_TORCH_WHEEL": "1", "TORCH_WHEELS_PATH": "dist"}, clear=True
    )
    def test_params_missing_torch_whls_raises(self, _is_path):
        # [FUNCTION DOC] With the path-exists check patched to fail, expects construction to
        # raise a value error whose message names the wheels path variable.
        # KNOWN ISSUE: changes the process working directory into a temporary directory that is
        # deleted afterwards and never restores it.
        with tempfile.TemporaryDirectory() as td:
            os.chdir(td)
            with self.assertRaises(ValueError) as cm:
                vllm_build.VllmBuildParameters(
                    use_local_base_image=False,
                    use_local_dockerfile=False,
                )
        err = cm.exception
        self.assertIn("TORCH_WHEELS_PATH", str(err))

    @patch(f"{_VLLM_BUILD_MODULE}.local_image_exists", return_value=False)
    @patch.dict(
        os.environ, {"USE_LOCAL_BASE_IMAGE": "1", "BASE_IMAGE": "img:tag"}, clear=True
    )
    def test_params_missing_local_base_image_raises(self, _local_img):
        # [FUNCTION DOC] With the local image check patched to fail, expects a value error whose
        # message names the base image variable.
        # KNOWN ISSUE: changes the process working directory into a temporary directory that is
        # deleted afterwards and never restores it.
        with tempfile.TemporaryDirectory() as td:
            os.chdir(td)
            with self.assertRaises(ValueError) as cm:
                vllm_build.VllmBuildParameters(
                    use_torch_whl=False,
                    use_local_dockerfile=False,
                )
        err = cm.exception
        self.assertIn("BASE_IMAGE", str(err))

    @patch(f"{_VLLM_BUILD_MODULE}.is_path_exist", return_value=False)
    @patch.dict(
        os.environ,
        {"USE_LOCAL_DOCKERFILE": "1", "DOCKERFILE_PATH": "Dockerfile"},
        clear=True,
    )
    def test_params_missing_dockerfile_raises(self, _is_path):
        # [FUNCTION DOC] With the path-exists check patched to fail, expects a value error whose
        # message names the Dockerfile path variable.
        # KNOWN ISSUE: changes the process working directory into a temporary directory that is
        # deleted afterwards and never restores it.
        with tempfile.TemporaryDirectory() as td:
            os.chdir(td)
            with self.assertRaises(ValueError) as cm:
                vllm_build.VllmBuildParameters(
                    use_torch_whl=False,
                    use_local_base_image=False,
                )
        err = cm.exception
        self.assertIn("DOCKERFILE_PATH", str(err))

    @patch(f"{_VLLM_BUILD_MODULE}.is_path_exist", return_value=False)
    @patch.dict(
        os.environ,
        {"OUTPUT_DIR": ""},
        clear=True,
    )
    def test_params_missing_output_dir(self, _is_path):
        # [FUNCTION DOC] With an empty output directory and the path check failing, expects
        # construction to raise a file-not-found error.
        with self.assertRaises(FileNotFoundError):
            vllm_build.VllmBuildParameters()


class TestBuildCmdAndRun(unittest.TestCase):
    # [CLASS DOC] Tests for VllmBuildRunner: the generated docker build command text and the
    # run method's orchestration (clone, ensure output dir, generate command, execute).
    @patch(f"{_VLLM_BUILD_MODULE}.local_image_exists", return_value=True)
    def test_generate_docker_build_cmd_includes_bits(self, _exists):
        # [FUNCTION DOC] Fills a mock parameter object (output dir, local base image, wheels
        # path, job count, CUDA/Python versions, sccache settings, arch list, target stage, tag)
        # and asserts the whitespace-normalized command contains the expected output, Dockerfile,
        # pull, build-arg, target and tag options, with the wheels path reduced to its last part.
        runner = vllm_build.VllmBuildRunner()
        inputs = MagicMock()
        inputs.output_dir = Path("/abs/out")
        inputs.use_local_base_image = True
        inputs.base_image = "img:tag"
        inputs.torch_whls_path = Path("./vllm/tmp")
        inputs.max_jobs = 64
        inputs.cuda_version = "12.8.1"
        inputs.python_version = "3.12"
        inputs.sccache_bucket = "my-bucket"
        inputs.sccache_region = "us-west-2"
        inputs.torch_cuda_arch_list = "8.0;9.0"
        inputs.target_stage = "export-wheels"
        inputs.tag_name = "vllm-wheels"

        cmd = runner._generate_docker_build_cmd(inputs)
        squashed = " ".join(cmd.split())

        self.assertIn("--output type=local,dest=/abs/out", squashed)
        self.assertIn("-f docker/Dockerfile.nightly_torch", squashed)
        self.assertIn("--pull=false", squashed)
        self.assertIn("--build-arg TORCH_WHEELS_PATH=tmp", squashed)
        self.assertIn("--build-arg BUILD_BASE_IMAGE=img:tag", squashed)
        self.assertIn("--build-arg FINAL_BASE_IMAGE=img:tag", squashed)
        self.assertIn("--build-arg max_jobs=64", squashed)
        self.assertIn("--build-arg CUDA_VERSION=12.8.1", squashed)
        self.assertIn("--build-arg PYTHON_VERSION=3.12", squashed)
        self.assertIn("--build-arg USE_SCCACHE=1", squashed)
        self.assertIn("--build-arg SCCACHE_BUCKET_NAME=my-bucket", squashed)
        self.assertIn("--build-arg SCCACHE_REGION_NAME=us-west-2", squashed)
        self.assertIn("--build-arg torch_cuda_arch_list='8.0;9.0'", squashed)
        self.assertIn("--target export-wheels", squashed)
        self.assertIn("-t vllm-wheels", squashed)

    @patch(f"{_VLLM_BUILD_MODULE}.run_command")
    @patch(f"{_VLLM_BUILD_MODULE}.ensure_dir_exists")
    @patch(f"{_VLLM_BUILD_MODULE}.clone_vllm")
    @patch.object(
        vllm_build.VllmBuildRunner,
        "_generate_docker_build_cmd",
        return_value="docker buildx ...",
    )
    @patch.dict(
        os.environ,
        {
            "USE_TORCH_WHEEL": "0",
            "USE_LOCAL_BASE_IMAGE": "0",
            "USE_LOCAL_DOCKERFILE": "0",
            "OUTPUT_DIR": "shared",
        },
        clear=True,
    )
    def test_run_calls_clone_prepare_and_build(
        self, mock_gen, mock_clone, mock_ensure, mock_run
    ):
        # [FUNCTION DOC] With parameters mocked to not use a local Dockerfile or wheels, runs
        # the runner and asserts clone was called once, the output dir was ensured, the build
        # command was generated from the parameters, and the command ran once with the working
        # directory set to vllm (raises an assertion error otherwise).
        params = MagicMock()
        params.output_dir = Path("shared")
        params.use_local_dockerfile = False
        params.use_torch_whl = False

        with patch(f"{_VLLM_BUILD_MODULE}.VllmBuildParameters", return_value=params):
            runner = vllm_build.VllmBuildRunner()
            runner.run()

        mock_clone.assert_called_once()
        mock_ensure.assert_called_once_with(Path("shared"))
        mock_gen.assert_called_once_with(params)
        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        if kwargs.get("cwd") != "vllm":
            raise AssertionError(f"Expected cwd='vllm', got cwd={kwargs.get('cwd')!r}")


class TestVllmTestRunner(unittest.TestCase):
    # [CLASS DOC] Tests for VllmTestRunner environment preparation.
    @patch.dict(
        os.environ,
        {
            "CI": "1",
            "HF_HOME": "/tmp/hf",
            "VLLM_TEST_HUGGING_FACE_TOKEN": "token",
        },
        clear=True,
    )
    def test_set_envs_disables_custom_op_aliasing_errors(self):
        # [FUNCTION DOC] Builds a runner from namespace arguments, calls its env-setting method
        # with parameters carrying an arch list, and asserts the CUDA arch list, HF_TOKEN (copied
        # from the Hugging Face token variable) and the custom-op aliasing error switch (set to
        # zero) are present in os.environ. The name is slightly misleading: it covers all three.
        args = SimpleNamespace(
            shard_id=1,
            num_shards=1,
            test_plan="vllm_basic_correctness_test",
        )
        runner = vllm_test.VllmTestRunner(args)
        params = SimpleNamespace(torch_cuda_arch_list="8.0")

        runner._set_envs(params)

        self.assertEqual(os.environ["TORCH_CUDA_ARCH_LIST"], "8.0")
        self.assertEqual(os.environ["HF_TOKEN"], "token")
        self.assertEqual(os.environ["TORCHINDUCTOR_ERROR_ON_CUSTOM_OP_ALIASING"], "0")
