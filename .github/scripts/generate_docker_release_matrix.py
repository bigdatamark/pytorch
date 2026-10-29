#!/usr/bin/env python3

# [MODULE DOC] Script that prints, as JSON, the build matrix for docker release images: one
# runtime and one devel amd64 entry per supported CUDA version (with full CUDA and cuDNN
# versions from generate_binary_build_matrix) plus one cpu arm64 runtime entry. No import-time
# side effects beyond imports.
"""Generates a matrix for docker releases through github actions

Will output a condensed version of the matrix. Will include fllowing:
    * CUDA version short
    * CUDA full version
    * CUDNN version short
    * Image type either runtime or devel
    * Platform linux/arm64,linux/amd64

"""

import json

import generate_binary_build_matrix


DOCKER_IMAGE_TYPES = ["runtime", "devel"]


def generate_docker_matrix() -> dict[str, list[dict[str, str]]]:
    # [FUNCTION DOC] Returns a mapping with one key holding the list of matrix entries. Each CUDA
    # version yields an entry per image type (runtime, devel) with the short and full CUDA
    # versions, cuDNN version, image type and the amd64 platform; a final entry is the cpu
    # runtime image for arm64 with empty version fields. Raises a lookup error if a CUDA version
    # lacks a cuDNN version.
    ret: list[dict[str, str]] = []
    # CUDA amd64 Docker images are available as both runtime and devel while
    # CPU arm64 image is only available as runtime.
    for cuda, version in generate_binary_build_matrix.CUDA_ARCHES_FULL_VERSION.items():
        for image in DOCKER_IMAGE_TYPES:
            ret.append(
                {
                    "cuda": cuda,
                    "cuda_full_version": version,
                    "cudnn_version": generate_binary_build_matrix.CUDA_ARCHES_CUDNN_VERSION[
                        cuda
                    ],
                    "image_type": image,
                    "platform": "linux/amd64",
                }
            )
    ret.append(
        {
            "cuda": "cpu",
            "cuda_full_version": "",
            "cudnn_version": "",
            "image_type": "runtime",
            "platform": "linux/arm64",
        }
    )

    return {"include": ret}


if __name__ == "__main__":
    build_matrix = generate_docker_matrix()
    print(json.dumps(build_matrix))
