# [MODULE DOC] Tests for the Vulkan SPIR-V shader generator (tools.gen_vulkan_spv). Defines an
# inline GLSL template and a YAML parameter file, and checks which shader variants the
# generator produces. Import time: only defines the two data strings.
import tempfile
import unittest

from tools.gen_vulkan_spv import DEFAULT_ENV, SPVGenerator


####################
# Data for testing #
####################

test_shader = """
#version 450 core

#define FORMAT ${FORMAT}
#define PRECISION ${PRECISION}
#define OP(X) ${OPERATOR}

$def is_int(dtype):
$   return dtype in {"int", "int32", "int8"}

$def is_uint(dtype):
$   return dtype in {"uint", "uint32", "uint8"}

$if is_int(DTYPE):
  #define VEC4_T ivec4
$elif is_uint(DTYPE):
  #define VEC4_T uvec4
$else:
  #define VEC4_T vec4

$if not INPLACE:
  $if is_int(DTYPE):
    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly iimage3D uOutput;
    layout(set = 0, binding = 1) uniform PRECISION isampler3D uInput;
  $elif is_uint(DTYPE):
    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly uimage3D uOutput;
    layout(set = 0, binding = 1) uniform PRECISION usampler3D uInput;
  $else:
    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
    layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
$else:
  $if is_int(DTYPE):
    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict iimage3D uOutput;
  $elif is_uint(DTYPE):
    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict uimage3D uOutput;
  $else:
    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict image3D uOutput;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  $if not INPLACE:
    VEC4_T v = texelFetch(uInput, pos, 0);
  $else:
    VEC4_T v = imageLoad(uOutput, pos);
  $for i in range(ITER[0]):
    for (int i = 0; i < ${ITER[1]}; ++i) {
        v = OP(v + i);
    }
  imageStore(uOutput, pos, OP(v));
}

"""

test_params_yaml = """
test_shader:
  parameter_names_with_default_values:
    DTYPE: float
    INPLACE: false
    OPERATOR: X + 3
    ITER: !!python/tuple [3, 5]
  generate_variant_forall:
    INPLACE:
      - VALUE: false
        SUFFIX: ""
      - VALUE: true
        SUFFIX: inplace
    DTYPE:
      - VALUE: int8
      - VALUE: float
  shader_variants:
    - NAME: test_shader_1
    - NAME: test_shader_3
      OPERATOR: X - 1
      ITER: !!python/tuple [3, 2]
      generate_variant_forall:
        DTYPE:
        - VALUE: float
        - VALUE: int

"""

##############
# Unit Tests #
##############


class TestVulkanSPVCodegen(unittest.TestCase):
    # [CLASS DOC] Test case for SPVGenerator variant enumeration. State: tmpdir (source dir),
    # tmpoutdir (output dir, unused by the test) and generator built over tmpdir.
    def setUp(self) -> None:
        # [FUNCTION DOC] Creates temporary source and output directories, writes the shader
        # template and parameter YAML into the source directory, and builds a generator
        # with the default environment and no shader compiler path.
        # KNOWN ISSUE: the shader file name has a trailing comma in its name, so it is not a
        # glsl file by extension; the test still passes since variants come from the YAML.
        self.tmpdir = tempfile.TemporaryDirectory()

        with open(f"{self.tmpdir.name}/test_shader.glsl,", "w") as f:
            f.write(test_shader)

        with open(f"{self.tmpdir.name}/test_params.yaml", "w") as f:
            f.write(test_params_yaml)

        self.tmpoutdir = tempfile.TemporaryDirectory()

        self.generator = SPVGenerator(
            src_dir_paths=self.tmpdir.name, env=DEFAULT_ENV, glslc_path=None
        )

    def cleanUp(self) -> None:
        # [FUNCTION DOC] HELPER: Intended to remove both temporary directories.
        # KNOWN ISSUE: unittest calls tearDown, not cleanUp, so this is never run
        # automatically and the directories are left until garbage collection.
        self.tmpdir.cleanup()
        self.tmpoutdir.cleanup()

    def testOutputMap(self) -> None:
        # [FUNCTION DOC] Checks the set of generated shader names equals the six expected
        # variants (two names crossed with dtype and in-place options, with the second name
        # overriding the dtype list).
        # Each shader variant will produce variants generated based on all possible combinations
        # of the DTYPE and INPLACE parameters. test_shader_3 has fewer generated variants due to
        # a custom specified generate_variant_forall field.
        expected_output_shaders = {
            "test_shader_1_float",
            "test_shader_1_inplace_float",
            "test_shader_1_inplace_int8",
            "test_shader_1_int8",
            "test_shader_3_float",
            "test_shader_3_int",
        }

        actual_output_shaders = set(self.generator.output_shader_map.keys())

        self.assertEqual(expected_output_shaders, actual_output_shaders)
