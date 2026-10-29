# mypy: allow-untyped-defs
# [MODULE DOC] HELPER: Numpy image/video layout helpers for the TensorBoard writer.
# Renders matplotlib figures to arrays, tiles image batches into a grid, converts tensors with
# an arbitrary dimension-order string to height-width-channel layout, and packs a batch of
# videos into a single grid-of-frames array. Imports numpy at import time; matplotlib is
# imported lazily inside figure_to_image only.
import numpy as np
import numpy.typing as npt


# Functions for converting
def figure_to_image(figures, close=True):
    # [FUNCTION DOC] Render one matplotlib figure, or a list of figures, to uint8 numpy arrays.
    # Each figure is drawn with the Agg backend and returned as RGB in channel-first order
    # (alpha dropped). A list yields a stacked array with a leading batch dimension. If close
    # is true the figures are closed (pyplot state mutated). Needs matplotlib installed;
    # raises ImportError otherwise.
    """Render matplotlib figure to numpy format.

    Note that this requires the ``matplotlib`` package.

    Args:
        figures (matplotlib.pyplot.figure or list of figures): figure or a list of figures
        close (bool): Flag to automatically close the figure

    Returns:
        numpy.array: image in [CHW] order
    """
    import matplotlib.pyplot as plt
    import matplotlib.backends.backend_agg as plt_backend_agg

    def render_to_rgb(figure):
        # [FUNCTION DOC] HELPER: Draw a single figure on an Agg canvas and return its pixels
        # as a channel-first RGB array; closes the figure when the enclosing close flag is set.
        canvas = plt_backend_agg.FigureCanvasAgg(figure)
        canvas.draw()
        data: npt.NDArray = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        w, h = figure.canvas.get_width_height()
        image_hwc = data.reshape([h, w, 4])[:, :, 0:3]
        image_chw = np.moveaxis(image_hwc, source=2, destination=0)
        if close:
            plt.close(figure)
        return image_chw

    if isinstance(figures, list):
        images = [render_to_rgb(figure) for figure in figures]
        return np.stack(images)
    else:
        image = render_to_rgb(figures)
        return image


def _prepare_video(V):
    # [FUNCTION DOC] HELPER: Pack a 5D batch-time-channel-height-width array into a 4D
    # time-height-width-channel array. Converts uint8 input to float scaled to 0..1, pads the
    # batch with zero videos up to a power of two, then tiles the batch into a rows-by-columns
    # grid per frame. Returns the new array; the input is not modified.
    # KNOWN ISSUE: when the batch size is already a power of two, the row count is derived from
    # the original size, and the padded zero-filled videos use float64, promoting the dtype.
    """
    Convert a 5D tensor into 4D tensor.

    Conversion is done from [batchsize, time(frame), channel(color), height, width]  (5D tensor)
    to [time(frame), new_width, new_height, channel] (4D tensor).

    A batch of images are spread to a grid, which forms a frame.
    e.g. Video with batchsize 16 will have a 4x4 grid.
    """
    b, t, c, h, w = V.shape

    if V.dtype == np.uint8:
        V = np.float32(V) / 255.0

    def is_power2(num):
        # [FUNCTION DOC] HELPER: Return true if num is a nonzero power of two.
        return num != 0 and ((num & (num - 1)) == 0)

    # pad to nearest power of 2, all at once

    if not is_power2(V.shape[0]):

        len_addition = int(2 ** V.shape[0].bit_length() - V.shape[0])
        V = np.concatenate((V, np.zeros(shape=(len_addition, t, c, h, w))), axis=0)

    n_rows = 2 ** ((b.bit_length() - 1) // 2)

    n_cols = V.shape[0] // n_rows

    V = np.reshape(V, (n_rows, n_cols, t, c, h, w))
    V = np.transpose(V, axes=(2, 0, 4, 1, 5, 3))
    V = np.reshape(V, (t, n_rows * h, n_cols * w, c))

    return V


def make_grid(I, ncols=8):
    # [FUNCTION DOC] Tile a numpy batch of images (N x 1 or 3 x H x W) into one 3-channel
    # canvas, filling row by row with at most ncols images per row; unused cells stay zero.
    # Single-channel input is replicated to three channels. Raises AssertionError if the input
    # is not a numpy array or is not 4D with 3 channels after replication.
    # I: N1HW or N3HW
    if not isinstance(I, np.ndarray):
        raise AssertionError("plugin error, should pass numpy array here")
    if I.shape[1] == 1:
        I = np.concatenate([I, I, I], 1)
    if I.ndim != 4 or I.shape[1] != 3:
        raise AssertionError("Input should be a 4D numpy array with 3 channels")
    nimg = I.shape[0]
    H = I.shape[2]
    W = I.shape[3]
    ncols = min(nimg, ncols)
    nrows = int(np.ceil(float(nimg) / ncols))
    canvas = np.zeros((3, H * nrows, W * ncols), dtype=I.dtype)
    i = 0
    for y in range(nrows):
        for x in range(ncols):
            if i >= nimg:
                break
            canvas[:, y * H : (y + 1) * H, x * W : (x + 1) * W] = I[i]
            i = i + 1
    return canvas

    # if modality == 'IMG':
    #     if x.dtype == np.uint8:
    #         x = x.astype(np.float32) / 255.0


def convert_to_HWC(tensor, input_format):  # tensor: numpy array
    # [FUNCTION DOC] Convert a numpy array to height-width-channel layout using a dimension
    # string such as NCHW, CHW, HWC or HW (case-insensitive). 4D input is permuted and tiled
    # into a grid; 3D input is permuted with a single channel replicated to three; 2D input is
    # stacked into three identical channels. Raises AssertionError if the format repeats a
    # letter or its length differs from the number of dimensions.
    # KNOWN ISSUE: the duplicate check runs before upper-casing, so mixed-case repeats pass;
    # lengths other than 2, 3 or 4 fall through and return None.
    if len(set(input_format)) != len(input_format):
        raise AssertionError(f"You can not use the same dimension shorthand twice. \
            input_format: {input_format}")
    if len(tensor.shape) != len(input_format):
        raise AssertionError(f"size of input tensor and input format are different. \
        tensor shape: {tensor.shape}, input_format: {input_format}")
    input_format = input_format.upper()

    if len(input_format) == 4:
        index = [input_format.find(c) for c in "NCHW"]
        tensor_NCHW = tensor.transpose(index)
        tensor_CHW = make_grid(tensor_NCHW)
        return tensor_CHW.transpose(1, 2, 0)

    if len(input_format) == 3:
        index = [input_format.find(c) for c in "HWC"]
        tensor_HWC = tensor.transpose(index)
        if tensor_HWC.shape[2] == 1:
            tensor_HWC = np.concatenate([tensor_HWC, tensor_HWC, tensor_HWC], 2)
        return tensor_HWC

    if len(input_format) == 2:
        index = [input_format.find(c) for c in "HW"]
        tensor = tensor.transpose(index)
        tensor = np.stack([tensor, tensor, tensor], 2)
        return tensor
