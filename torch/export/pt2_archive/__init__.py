# [PACKAGE DOC] Public entry point of the PT2 archive format package. Re-exports the archive
# writer and reader classes and the predicate that tells whether a path is a PT2 archive, all
# implemented in the private _package module. No import-time side effects beyond that import.
from ._package import is_pt2_package, PT2ArchiveReader, PT2ArchiveWriter


__all__ = ["PT2ArchiveWriter", "PT2ArchiveReader", "is_pt2_package"]
