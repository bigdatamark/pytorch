# mypy: allow-untyped-defs
# [MODULE DOC] Helpers that write TensorBoard embedding projector files: metadata tsv, sprite
# image, tensors tsv, projector config text and the EmbeddingInfo message. All output goes
# through tensorboard's tf file API. Import-time: computes whether gfile has a join function.
import math
import numpy as np
from ._convert_np import make_np
from ._utils import make_grid
from tensorboard.compat import tf
from tensorboard.plugins.projector.projector_config_pb2 import EmbeddingInfo


_HAS_GFILE_JOIN = hasattr(tf.io.gfile, "join")


def _gfile_join(a, b):
    # [FUNCTION DOC] HELPER: Join two path pieces using gfile.join when available, otherwise
    # the join method of the filesystem object for the first path. Returns the joined path.
    # The join API is different between tensorboard's TF stub and TF:
    # https://github.com/tensorflow/tensorboard/issues/6080
    # We need to try both because `tf` may point to either the stub or the real TF.
    if _HAS_GFILE_JOIN:
        return tf.io.gfile.join(a, b)
    else:
        fs = tf.io.gfile.get_filesystem(a)
        return fs.join(a, b)


def make_tsv(metadata, save_path, metadata_header=None) -> None:
    # [FUNCTION DOC] Write metadata.tsv under save_path. Without a header, each entry becomes
    # one line; with a header, metadata is a list of rows, each joined by tabs, and the header
    # is the first line. Raises AssertionError if header length differs from the column count.
    # Side effect: creates or overwrites the file.
    if not metadata_header:
        metadata = [str(x) for x in metadata]
    else:
        if len(metadata_header) != len(
            metadata[0]
        ):
            raise AssertionError("len of header must be equal to the number of columns in metadata")
        metadata = ["\t".join(str(e) for e in l) for l in [metadata_header] + metadata]

    metadata_bytes = tf.compat.as_bytes("\n".join(metadata) + "\n")
    with tf.io.gfile.GFile(_gfile_join(save_path, "metadata.tsv"), "wb") as f:
        f.write(metadata_bytes)


# https://github.com/tensorflow/tensorboard/issues/44 image label will be squared
def make_sprite(label_img, save_path) -> None:
    # [FUNCTION DOC] Tile a batch of label images into one square grid image and save it as
    # sprite.png under save_path. The grid has ceil(sqrt(batch)) columns, is padded to a square
    # with zeros, and pixel values are scaled by 255 and clipped. Requires PIL. Side effect:
    # file write.
    from PIL import Image
    from io import BytesIO

    # this ensures the sprite image has correct dimension as described in
    # https://www.tensorflow.org/get_started/embedding_viz
    nrow = math.ceil((label_img.size(0)) ** 0.5)
    arranged_img_CHW = make_grid(make_np(label_img), ncols=nrow)

    # augment images so that #images equals nrow*nrow
    arranged_augment_square_HWC = np.zeros(
        (arranged_img_CHW.shape[2], arranged_img_CHW.shape[2], 3)
    )
    arranged_img_HWC = arranged_img_CHW.transpose(1, 2, 0)  # chw -> hwc
    arranged_augment_square_HWC[: arranged_img_HWC.shape[0], :, :] = arranged_img_HWC
    im = Image.fromarray(np.uint8((arranged_augment_square_HWC * 255).clip(0, 255)))

    with BytesIO() as buf:
        im.save(buf, format="PNG")
        im_bytes = buf.getvalue()

    with tf.io.gfile.GFile(_gfile_join(save_path, "sprite.png"), "wb") as f:
        f.write(im_bytes)


def get_embedding_info(metadata, label_img, subdir, global_step, tag):
    # [FUNCTION DOC] Build an EmbeddingInfo message named tag plus the step zero-padded to five
    # digits, with the tensors path inside subdir. Adds a metadata path if metadata is given and
    # a sprite path and image dimensions (width, height from label_img) if label_img is given.
    # Returns the message; no I/O.
    info = EmbeddingInfo()
    info.tensor_name = f"{tag}:{str(global_step).zfill(5)}"
    info.tensor_path = _gfile_join(subdir, "tensors.tsv")
    if metadata is not None:
        info.metadata_path = _gfile_join(subdir, "metadata.tsv")
    if label_img is not None:
        info.sprite.image_path = _gfile_join(subdir, "sprite.png")
        info.sprite.single_image_dim.extend([label_img.size(3), label_img.size(2)])
    return info


def write_pbtxt(save_path, contents) -> None:
    # [FUNCTION DOC] Write the text contents to projector_config.pbtxt under save_path
    # (side effect: creates or overwrites the file).
    config_path = _gfile_join(save_path, "projector_config.pbtxt")
    with tf.io.gfile.GFile(config_path, "wb") as f:
        f.write(tf.compat.as_bytes(contents))


def make_mat(matlist, save_path) -> None:
    # [FUNCTION DOC] Write tensors.tsv under save_path with one line per row of matlist; each
    # row is an iterable of scalar tensors whose values are tab separated. Side effect: file write.
    with tf.io.gfile.GFile(_gfile_join(save_path, "tensors.tsv"), "wb") as f:
        for x in matlist:
            x = [str(i.item()) for i in x]
            f.write(tf.compat.as_bytes("\t".join(x) + "\n"))
