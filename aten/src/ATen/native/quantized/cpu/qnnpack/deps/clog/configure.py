#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# [MODULE DOC] Configuration script for the clog logging library, using the confu build
# configuration framework. Builds a static library from the clog source and a unit test
# target (linking googletest, and the log library on Android). Run as a script, it builds
# the configuration from the command-line arguments and generates the build files.
# Import-time side effects: creates the standard argument parser.
import confu


parser = confu.standard_parser("clog configuration script")


def main(args):
    # [FUNCTION DOC] Parse the arguments, create a confu build, export the clog public header
    # directory, define the static library clog from the source file and the clog unit test
    # executable. Returns the build object; the caller generates the build files from it.
    options = parser.parse_args(args)
    build = confu.Build.from_options(options)

    build.export_cpath("include", ["clog.h"])

    with build.options(source_dir="src", extra_include_dirs="src"):
        build.static_library("clog", build.cc("clog.c"))

    with build.options(
        source_dir="test",
        deps={(build, build.deps.googletest): all, "log": build.target.is_android},
    ):
        build.unittest("clog-test", build.cxx("clog.cc"))

    return build


if __name__ == "__main__":
    import sys

    main(sys.argv[1:]).generate()
