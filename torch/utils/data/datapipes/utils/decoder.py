# mypy: allow-untyped-defs
# [MODULE DOC]
# Decoding of raw bytes into Python objects by file extension, used by the routed decoder
# DataPipe. Provides handler functions (basic types, images, video, audio, mat files), the
# Decoder class that tries a list of handlers in order, and the extension extraction helper.
# Optional dependencies (numpy, PIL, torchvision, torchaudio, scipy) are imported lazily only
# when a handler needs them.
# This file takes partial of the implementation from NVIDIA's webdataset at here:
# https://github.com/tmbdev/webdataset/blob/master/webdataset/autodecode.py

import io
import json
import os.path
import pickle
import tempfile

import torch
from torch.utils.data.datapipes.utils.common import StreamWrapper


__all__ = [
    "Decoder",
    "ImageHandler",
    "MatHandler",
    "audiohandler",
    "basichandlers",
    "extension_extract_fn",
    "handle_extension",
    "imagehandler",
    "mathandler",
    "videohandler",
]


################################################################
# handle basic datatypes
################################################################
def basichandlers(extension: str, data):
    # [FUNCTION DOC] Decodes bytes by extension: text decoded as utf-8; integer-like
    # extensions parsed as int (None if parsing fails); json parsed; pickle loaded; "pt"
    # loaded with torch.load from an in-memory buffer. Returns None for unknown extensions.
    # Side effects: unpickling and torch.load can execute arbitrary code from untrusted data.
    # KNOWN ISSUE: the text and json checks test substring membership in a space-separated
    # string, so partial names (for example "t" or "js") also match; the docstring example
    # uses an extension not handled as shown.
    """Transforms raw data (byte stream) into python objects.

    Looks at the extension and loads the data into a python object supporting
    the corresponding extension.

    Args:
        extension (str): The file extension
        data (byte stream): Data to load into a python object.

    Returns:
        object: The data loaded into a corresponding python object
            supporting the extension.

    Example:
        >>> import pickle
        >>> data = pickle.dumps("some data")
        >>> new_data = basichandlers("pickle", data)
        >>> new_data
        some data

    The transformation of data for extensions are:
        - txt, text, transcript: utf-8 decoded data of str format
        - cls, cls2, class, count, index, inx, id: int
        - json, jsn: json loaded data
        - pickle, pyd: pickle loaded data
        - pt: torch loaded data
    """

    if extension in "txt text transcript":
        return data.decode("utf-8")

    if extension in ["cls", "cls2", "class", "count", "index", "inx", "id"]:
        try:
            return int(data)
        except ValueError:
            return None

    if extension in "json jsn":
        return json.loads(data)

    if extension in ["pyd", "pickle"]:
        return pickle.loads(data)

    if extension == "pt":
        stream = io.BytesIO(data)
        return torch.load(stream)

    # if extension in "ten tb".split():
    #     from . import tenbin
    #     return tenbin.decode_buffer(data)

    # if extension in "mp msgpack msg".split():
    #     import msgpack
    #     return msgpack.unpackb(data)

    return None


################################################################
# handle images
################################################################
imagespecs = {
    "l8": ("numpy", "uint8", "l"),
    "rgb8": ("numpy", "uint8", "rgb"),
    "rgba8": ("numpy", "uint8", "rgba"),
    "l": ("numpy", "float", "l"),
    "rgb": ("numpy", "float", "rgb"),
    "rgba": ("numpy", "float", "rgba"),
    "torchl8": ("torch", "uint8", "l"),
    "torchrgb8": ("torch", "uint8", "rgb"),
    "torchrgba8": ("torch", "uint8", "rgba"),
    "torchl": ("torch", "float", "l"),
    "torchrgb": ("torch", "float", "rgb"),
    "torch": ("torch", "float", "rgb"),
    "torchrgba": ("torch", "float", "rgba"),
    "pill": ("pil", None, "l"),
    "pil": ("pil", None, "rgb"),
    "pilrgb": ("pil", None, "rgb"),
    "pilrgba": ("pil", None, "rgba"),
}


def handle_extension(extensions, f):
    # [FUNCTION DOC] Builds a handler taking (key, data) that calls f on the data when the
    # key's trailing dotted components match an extension from the space-separated,
    # case-insensitive list; otherwise the handler returns None.
    # KNOWN ISSUE: the inner handler returns None after testing only the first listed
    # extension (the early return sits inside the loop), so later extensions never match.
    """
    Return a decoder handler function for the list of extensions.

    Extensions can be a space separated list of extensions.
    Extensions can contain dots, in which case the corresponding number
    of extension components must be present in the key given to f.
    Comparisons are case insensitive.
    Examples:
    handle_extension("jpg jpeg", my_decode_jpg)  # invoked for any file.jpg
    handle_extension("seg.jpg", special_case_jpg)  # invoked only for file.seg.jpg
    """
    extensions = extensions.lower().split()

    def g(key, data):
        # [FUNCTION DOC] HELPER: Handler closure returned by handle_extension; splits the key
        # on dots, compares its tail with the first target extension and returns f of the data
        # on a match, else None.
        extension = key.lower().split(".")

        for target in extensions:
            target = target.split(".")
            if len(target) > len(extension):
                continue

            if extension[-len(target) :] == target:
                return f(data)
            return None

    return g


class ImageHandler:
    # [CLASS DOC]
    # Callable decoder handler for image bytes. State: the lower-cased imagespec naming the
    # output kind (numpy, torch or PIL), element type and color mode, looked up in the
    # imagespecs table. Calling it with (extension, data) returns a decoded image or None.
    """
    Decode image data using the given `imagespec`.

    The `imagespec` specifies whether the image is decoded
    to numpy/torch/pi, decoded to uint8/float, and decoded
    to l/rgb/rgba:

    - l8: numpy uint8 l
    - rgb8: numpy uint8 rgb
    - rgba8: numpy uint8 rgba
    - l: numpy float l
    - rgb: numpy float rgb
    - rgba: numpy float rgba
    - torchl8: torch uint8 l
    - torchrgb8: torch uint8 rgb
    - torchrgba8: torch uint8 rgba
    - torchl: torch float l
    - torchrgb: torch float rgb
    - torch: torch float rgb
    - torchrgba: torch float rgba
    - pill: pil None l
    - pil: pil None rgb
    - pilrgb: pil None rgb
    - pilrgba: pil None rgba
    """

    def __init__(self, imagespec) -> None:
        # [FUNCTION DOC] Validates the image spec against the known table and stores it in
        # lower case. Raises AssertionError for an unknown spec.
        # KNOWN ISSUE: the membership check is case-sensitive, so the later lower-casing
        # never matters.
        if imagespec not in list(imagespecs.keys()):
            raise AssertionError(f"unknown image specification: {imagespec}")
        self.imagespec = imagespec.lower()

    def __call__(self, extension, data):
        # [FUNCTION DOC] Returns None for non-image extensions. Otherwise imports numpy and
        # PIL (raising ModuleNotFoundError with an install hint if missing), decodes the bytes,
        # converts to the spec's color mode and returns a PIL image, a numpy array, or a torch
        # tensor (channels first), as uint8 or scaled to float in [0, 1]. Raises
        # AssertionError if the decoded numpy array is not uint8.
        # KNOWN ISSUE: the torch path transposes three axes, which fails for single-channel
        # (2-D) images from the grayscale specs.
        if extension.lower() not in ["jpg", "jpeg", "png", "ppm", "pgm", "pbm", "pnm"]:
            return None

        try:
            import numpy as np
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Package `numpy` is required to be installed for default image decoder."
                "Please use `pip install numpy` to install the package"
            ) from e

        try:
            import PIL.Image
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Package `PIL` is required to be installed for default image decoder."
                "Please use `pip install Pillow` to install the package"
            ) from e

        imagespec = self.imagespec
        atype, etype, mode = imagespecs[imagespec]

        with io.BytesIO(data) as stream:
            img = PIL.Image.open(stream)
            img.load()
            img = img.convert(mode.upper())
            if atype == "pil":
                return img
            elif atype == "numpy":
                result = np.asarray(img)
                if result.dtype != np.uint8:
                    raise AssertionError(
                        f"numpy image array should be type uint8, but got {result.dtype}"
                    )
                if etype == "uint8":
                    return result
                else:
                    return result.astype("f") / 255.0
            elif atype == "torch":
                result = np.asarray(img)
                if result.dtype != np.uint8:
                    raise AssertionError(
                        f"numpy image array should be type uint8, but got {result.dtype}"
                    )

                if etype == "uint8":
                    result = np.array(result.transpose(2, 0, 1))
                    return torch.tensor(result)
                else:
                    result = np.array(result.transpose(2, 0, 1))
                    return torch.tensor(result) / 255.0
            return None


def imagehandler(imagespec):
    # [FUNCTION DOC] Factory returning an ImageHandler for the given image spec.
    return ImageHandler(imagespec)


################################################################
# torch video
################################################################
def videohandler(extension, data):
    # [FUNCTION DOC] For video extensions, writes the bytes to a temporary file and returns
    # what torchvision's video reader gives for it (the temporary directory is removed
    # afterwards); returns None for other extensions. Raises ModuleNotFoundError if
    # torchvision cannot be imported.
    if extension not in [
        "mp4",
        "ogv",
        "mjpeg",
        "avi",
        "mov",
        "h264",
        "mpg",
        "webm",
        "wmv",
    ]:
        return None

    try:
        import torchvision.io
    except ImportError as e:
        raise ModuleNotFoundError(
            "Package `torchvision` is required to be installed for default video file loader."
            "Please use `pip install torchvision`"
            "to install the package"
        ) from e

    with tempfile.TemporaryDirectory() as dirname:
        fname = os.path.join(dirname, f"file.{extension}")
        with open(fname, "wb") as stream:
            stream.write(data)
            return torchvision.io.read_video(fname)


################################################################
# torchaudio
################################################################
def audiohandler(extension, data):
    # [FUNCTION DOC] For audio extensions, writes the bytes to a temporary file and returns
    # what torchaudio loads from it (the temporary directory is removed afterwards); returns
    # None for other extensions. Raises ModuleNotFoundError if torchaudio cannot be imported.
    if extension not in ["flac", "mp3", "sox", "wav", "m4a", "ogg", "wma"]:
        return None

    try:
        import torchaudio  # type: ignore[import]
    except ImportError as e:
        raise ModuleNotFoundError(
            "Package `torchaudio` is required to be installed for default audio file loader."
            "Please use `pip install torchaudio`"
            "to install the package"
        ) from e

    with tempfile.TemporaryDirectory() as dirname:
        fname = os.path.join(dirname, f"file.{extension}")
        with open(fname, "wb") as stream:
            stream.write(data)
            return torchaudio.load(fname)


################################################################
# mat
################################################################
class MatHandler:
    # [CLASS DOC]
    # Callable decoder handler for MATLAB "mat" files. State: the scipy.io module and the
    # keyword arguments forwarded to its loader. Calling it returns the loaded contents for
    # the "mat" extension, else None.
    def __init__(self, **loadmat_kwargs) -> None:
        # [FUNCTION DOC] Imports scipy.io (raising ModuleNotFoundError with an install hint
        # if missing) and stores the keyword arguments for later loading.
        try:
            import scipy.io as sio
        except ImportError as e:
            raise ModuleNotFoundError(
                "Package `scipy` is required to be installed for mat file."
                "Please use `pip install scipy`"
                "to install the package"
            ) from e
        self.sio = sio
        self.loadmat_kwargs = loadmat_kwargs

    def __call__(self, extension, data):
        # [FUNCTION DOC] Returns None unless the extension is "mat"; otherwise loads the bytes
        # from an in-memory buffer using the scipy loader and the stored keyword arguments.
        if extension != "mat":
            return None
        with io.BytesIO(data) as stream:
            return self.sio.loadmat(stream, **self.loadmat_kwargs)


def mathandler(**loadmat_kwargs):
    # [FUNCTION DOC] Factory returning a MatHandler built with the given keyword arguments.
    return MatHandler(**loadmat_kwargs)


################################################################
# a sample decoder
################################################################
# Extract extension from pathname
def extension_extract_fn(pathname):
    # [FUNCTION DOC] Returns the file extension of the pathname without the leading dot
    # (empty string if there is none).
    ext = os.path.splitext(pathname)[1]
    # Remove dot
    if ext:
        ext = ext[1:]
    return ext


class Decoder:
    # [CLASS DOC]
    # Applies an ordered list of handlers to (key, data) pairs; the first handler returning
    # a non-None result wins, otherwise the data is returned unchanged. State: the handler
    # list (earlier means higher priority) and key_fn that maps a pathname to the key
    # given to handlers (default: file extension).
    """
    Decode key/data sets using a list of handlers.

    For each key/data item, this iterates through the list of
    handlers until some handler returns something other than None.
    """

    def __init__(self, *handler, key_fn=extension_extract_fn) -> None:
        # [FUNCTION DOC] Stores the handlers as a list (empty if none) and the key function.
        self.handlers = list(handler) if handler else []
        self.key_fn = key_fn

    # Insert new handler from the beginning of handlers list to make sure the new
    # handler having the highest priority
    def add_handler(self, *handler) -> None:
        # [FUNCTION DOC] Prepends the given handlers (keeping their order) to the handler list
        # so they take priority; does nothing if none are given.
        if not handler:
            return
        self.handlers = list(handler) + self.handlers

    @staticmethod
    def _is_stream_handle(data):
        # [FUNCTION DOC] HELPER: True if data (or the object inside a StreamWrapper) is a
        # buffered or raw binary IO stream.
        obj_to_check = data.file_obj if isinstance(data, StreamWrapper) else data
        return isinstance(obj_to_check, (io.BufferedIOBase, io.RawIOBase))

    def decode1(self, key, data):
        # [FUNCTION DOC] Decodes one item. Falsy data is returned as is. A stream is fully
        # read into bytes and closed (side effect). Then each handler is called with (key,
        # data) in order and the first non-None result is returned; if none match, returns
        # the (possibly read) data.
        if not data:
            return data

        # if data is a stream handle, we need to read all the content before decoding
        if Decoder._is_stream_handle(data):
            ds = data
            # The behavior of .read can differ between streams (e.g. HTTPResponse), hence this is used instead
            data = b"".join(data)
            ds.close()

        for f in self.handlers:
            result = f(key, data)
            if result is not None:
                return result
        return data

    def decode(self, data):
        # [FUNCTION DOC] Decodes a (pathname, stream) tuple or an iterable of such pairs into
        # a mapping from pathname to decoded value (empty for None). Pathnames starting with
        # an underscore keep bytes values as utf-8 text without handler decoding; others go
        # through decode1 with the key from key_fn.
        # KNOWN ISSUE: an empty pathname raises IndexError; underscore keys with non-bytes
        # values are still decoded normally.
        result = {}
        # single data tuple(pathname, data stream)
        if isinstance(data, tuple):
            data = [data]

        if data is not None:
            for k, v in data:
                # TODO: xinyu, figure out why Nvidia do this?
                if k[0] == "_":
                    if isinstance(v, bytes):
                        v = v.decode("utf-8")
                        result[k] = v
                        continue
                result[k] = self.decode1(self.key_fn(k), v)
        return result

    def __call__(self, data):
        # [FUNCTION DOC] Same as decode.
        return self.decode(data)
