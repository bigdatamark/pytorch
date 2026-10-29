#!/usr/bin/env python3
# mypy: allow-untyped-defs
# [MODULE DOC] Command line tool and helpers to print the contents of a pickle file (or a
# member of a zip archive) without importing the classes it references. Unknown classes are
# replaced by placeholder objects. Exports FakeObject, FakeClass, DumpUnpickler and main.
import sys
import pickle
import struct
import pprint
import zipfile
import fnmatch
from typing import Any, IO

__all__ = ["FakeObject", "FakeClass", "DumpUnpickler", "main"]

class FakeObject:
    # [CLASS DOC] Placeholder for an object of an unimportable class: records module, class name,
    # constructor args and state (None if never set, so unset and None are indistinguishable).
    # Has a repr and a pretty-printer formatting hook (pp_format).
    def __init__(self, module, name, args) -> None:
        # [FUNCTION DOC] Store the module name, class name and args; state starts as None.
        self.module = module
        self.name = name
        self.args = args
        # NOTE: We don't distinguish between state never set and state set to None.
        self.state = None

    def __repr__(self) -> str:
        # [FUNCTION DOC] Return module.name followed by the args tuple repr and, if state is set,
        # a state suffix.
        state_str = "" if self.state is None else f"(state={self.state!r})"
        return f"{self.module}.{self.name}{self.args!r}{state_str}"

    def __setstate__(self, state):
        # [FUNCTION DOC] Pickle protocol hook: record the state without interpreting it.
        self.state = state

    @staticmethod
    def pp_format(printer, obj, stream, indent, allowance, context, level) -> None:
        # [FUNCTION DOC] Pretty-printer hook that writes obj to stream: plain repr when it has no
        # args and no state, name then formatted args when no state, name plus an indented state
        # when it has state but no args. Raises Exception when both args and state exist.
        if not obj.args and obj.state is None:
            stream.write(repr(obj))
            return
        if obj.state is None:
            stream.write(f"{obj.module}.{obj.name}")
            printer._format(obj.args, stream, indent + 1, allowance + 1, context, level)
            return
        if not obj.args:
            stream.write(f"{obj.module}.{obj.name}()(state=\n")
            indent += printer._indent_per_level
            stream.write(" " * indent)
            printer._format(obj.state, stream, indent, allowance + 1, context, level + 1)
            stream.write(")")
            return
        raise Exception("Need to implement")  # noqa: TRY002


class FakeClass:
    # [CLASS DOC] Placeholder for an unimportable class: calling it, or its replaced __new__,
    # returns a FakeObject carrying the module and class name and the call arguments.
    def __init__(self, module, name) -> None:
        # [FUNCTION DOC] Store module and name and install fake_new as the instance __new__.
        self.module = module
        self.name = name
        self.__new__ = self.fake_new  # type: ignore[assignment]

    def __repr__(self) -> str:
        # [FUNCTION DOC] Return module and class name joined by a dot.
        return f"{self.module}.{self.name}"

    def __call__(self, *args):
        # [FUNCTION DOC] Return a FakeObject for this class with the given positional args.
        return FakeObject(self.module, self.name, args)

    def fake_new(self, *args):
        # [FUNCTION DOC] Stand-in for __new__: drops the first argument (the class) and returns
        # a FakeObject with the remaining args.
        return FakeObject(self.module, self.name, args[1:])


class DumpUnpickler(pickle._Unpickler):  # type: ignore[name-defined]
    # [CLASS DOC] Pure Python unpickler that never imports classes: find_class yields FakeClass,
    # persistent ids yield FakeObject, and string loading optionally tolerates invalid UTF-8
    # (flag catch_invalid_utf8). Uses a copy of the base dispatch table with a patched
    # binary-unicode loader. The dump classmethod loads and pretty-prints a stream.
    def __init__(
            self,
            file,
            *,
            catch_invalid_utf8=False,
            **kwargs) -> None:
        # [FUNCTION DOC] Initialize the base unpickler with file and extra keyword arguments and
        # remember whether invalid UTF-8 strings are replaced rather than raising.
        super().__init__(file, **kwargs)
        self.catch_invalid_utf8 = catch_invalid_utf8

    def find_class(self, module, name):
        # [FUNCTION DOC] Return a FakeClass for module and name without importing anything.
        return FakeClass(module, name)

    def persistent_load(self, pid):
        # [FUNCTION DOC] Return a placeholder FakeObject (module pers, name obj) holding pid.
        return FakeObject("pers", "obj", (pid,))

    dispatch = dict(pickle._Unpickler.dispatch)  # type: ignore[attr-defined]

    # Custom objects in TorchScript are able to return invalid UTF-8 strings
    # from their pickle (__getstate__) functions.  Install a custom loader
    # for strings that catches the decode exception and replaces it with
    # a sentinel object.
    def load_binunicode(self) -> None:
        # [FUNCTION DOC] Unpickler opcode handler: read a 4 byte length and that many bytes and
        # push the decoded string. Raises Exception if the length exceeds the maximum size. On
        # invalid UTF-8, re-raises unless catch_invalid_utf8 is set, in which case a FakeObject
        # describing the decode error is pushed instead.
        strlen, = struct.unpack("<I", self.read(4))  # type: ignore[attr-defined]
        if strlen > sys.maxsize:
            raise Exception("String too long.")  # noqa: TRY002
        str_bytes = self.read(strlen)  # type: ignore[attr-defined]
        obj: Any
        try:
            obj = str(str_bytes, "utf-8", "surrogatepass")
        except UnicodeDecodeError as exn:
            if not self.catch_invalid_utf8:
                raise
            obj = FakeObject("builtin", "UnicodeDecodeError", (str(exn),))
        self.append(obj)  # type: ignore[attr-defined]
    dispatch[pickle.BINUNICODE[0]] = load_binunicode  # type: ignore[assignment]

    @classmethod
    def dump(cls, in_stream, out_stream):
        # [FUNCTION DOC] Unpickle one value from in_stream, pretty-print it to out_stream (stdout
        # if None) and return the value.
        value = cls(in_stream).load()
        pprint.pprint(value, stream=out_stream)
        return value


def main(argv, output_stream=None) -> int | None:
    # [FUNCTION DOC] CLI entry: argv must have exactly two items (program, target) where target
    # is a pickle path, archive@member, or archive@glob (first match). Dumps it to output_stream.
    # With wrong argc, prints usage to stderr and returns 2 (or raises Exception if an output
    # stream was supplied). Raises Exception if no archive member matches. Returns None on success.
    if len(argv) != 2:
        # Don't spam stderr if not using stdout.
        if output_stream is not None:
            raise Exception("Pass argv of length 2.")  # noqa: TRY002
        sys.stderr.write("usage: show_pickle PICKLE_FILE\n")
        sys.stderr.write("  PICKLE_FILE can be any of:\n")
        sys.stderr.write("    path to a pickle file\n")
        sys.stderr.write("    file.zip@member.pkl\n")
        sys.stderr.write("    file.zip@*/pattern.*\n")
        sys.stderr.write("      (shell glob pattern for members)\n")
        sys.stderr.write("      (only first match will be shown)\n")
        return 2

    fname = argv[1]
    handle: IO[bytes]
    if "@" not in fname:
        with open(fname, "rb") as handle:
            DumpUnpickler.dump(handle, output_stream)
    else:
        zfname, mname = fname.split("@", 1)
        with zipfile.ZipFile(zfname) as zf:
            if "*" not in mname:
                with zf.open(mname) as handle:
                    DumpUnpickler.dump(handle, output_stream)
            else:
                found = False
                for info in zf.infolist():
                    if fnmatch.fnmatch(info.filename, mname):
                        with zf.open(info) as handle:
                            DumpUnpickler.dump(handle, output_stream)
                        found = True
                        break
                if not found:
                    raise Exception(f"Could not find member matching {mname} in {zfname}")  # noqa: TRY002


if __name__ == "__main__":
    # Script mode: registers the FakeObject formatter with the pretty printer dispatch
    # table (only in this mode) and exits with the status from main.
    # This hack works on every version of Python I've tested.
    # I've tested on the following versions:
    #   3.7.4
    if True:
        pprint.PrettyPrinter._dispatch[FakeObject.__repr__] = FakeObject.pp_format  # type: ignore[attr-defined]

    sys.exit(main(sys.argv))
