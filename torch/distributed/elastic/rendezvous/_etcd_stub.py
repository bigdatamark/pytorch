# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# [MODULE DOC] HELPER: Placeholder replacements for the optional etcd client library so importing
# modules succeed without it. Every class here raises EtcdStubError (an import error) when
# instantiated or when its methods are called. No import-time side effects.

from typing import Any


"""
This file is not meant to be used directly. It serves as a stub to allow
other files to be safely imported without requiring the installation of
the 'etcd' library. The classes and methods here raise exceptions to
indicate that the real 'etcd' module is needed.
"""


class EtcdStubError(ImportError):
    # [CLASS DOC] HELPER: Import error signalling that the real etcd module is required.
    """Custom exception to indicate that the real etcd module is required."""

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Initializes with a fixed message saying etcd is not installed.
        super().__init__("The 'etcd' module is required but not installed.")


class EtcdAlreadyExist(Exception):
    # [CLASS DOC] HELPER: Stub for the etcd exception of the same name; cannot be instantiated.
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # [FUNCTION DOC] HELPER: Always raises EtcdStubError.
        raise EtcdStubError


class EtcdCompareFailed(Exception):
    # [CLASS DOC] HELPER: Stub for the etcd exception of the same name; cannot be instantiated.
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # [FUNCTION DOC] HELPER: Always raises EtcdStubError.
        raise EtcdStubError


class EtcdKeyNotFound(Exception):
    # [CLASS DOC] HELPER: Stub for the etcd exception of the same name; cannot be instantiated.
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # [FUNCTION DOC] HELPER: Always raises EtcdStubError.
        raise EtcdStubError


class EtcdWatchTimedOut(Exception):
    # [CLASS DOC] HELPER: Stub for the etcd exception of the same name; cannot be instantiated.
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # [FUNCTION DOC] HELPER: Always raises EtcdStubError.
        raise EtcdStubError


class EtcdEventIndexCleared(Exception):
    # [CLASS DOC] HELPER: Stub for the etcd exception of the same name; cannot be instantiated.
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # [FUNCTION DOC] HELPER: Always raises EtcdStubError.
        raise EtcdStubError


class EtcdException(Exception):
    # [CLASS DOC] HELPER: Stub for the etcd exception of the same name; cannot be instantiated.
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # [FUNCTION DOC] HELPER: Always raises EtcdStubError.
        raise EtcdStubError


class EtcdResult:
    # [CLASS DOC] HELPER: Stub for the etcd result type; cannot be instantiated.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Always raises EtcdStubError.
        raise EtcdStubError


class Client:
    # [CLASS DOC] HELPER: Stub for the etcd client; construction and all methods raise EtcdStubError.
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # [FUNCTION DOC] HELPER: Always raises EtcdStubError.
        raise EtcdStubError

    def read(self, key: str) -> None:
        # [FUNCTION DOC] HELPER: Stub; always raises EtcdStubError.
        raise EtcdStubError

    def write(
        self, key: str, value: Any, ttl: int | None = None, **kwargs: Any
    ) -> None:
        # [FUNCTION DOC] HELPER: Stub; always raises EtcdStubError.
        raise EtcdStubError

    def test_and_set(
        self, key: str, value: Any, prev_value: Any, ttl: int | None = None
    ) -> None:
        # [FUNCTION DOC] HELPER: Stub; always raises EtcdStubError.
        raise EtcdStubError
