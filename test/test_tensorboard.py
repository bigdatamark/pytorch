# Owner(s): ["module: unknown"]

# [MODULE DOC] Tests for torch.utils.tensorboard (SummaryWriter, summary builders, numpy
# conversion helpers, graph export and tensor protos). Import time: probes optional packages
# (tensorboard, torchvision, matplotlib) and sets the flags TEST_TENSORBOARD, HAS_TORCHVISION
# and TEST_MATPLOTLIB plus skip decorators; selects the Agg matplotlib backend when no DISPLAY
# is set; imports tensorboard-dependent names only if available, else defines a dummy DataType
# holder so parametrization still works. Golden protobuf text files live in the expect
# directory next to this file, named after the test method. Running the file registers the
# device-generic numpy test class and parametrized tests.
import io
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import expecttest
import numpy as np


TEST_TENSORBOARD = True
try:
    import tensorboard.summary.writer.event_file_writer  # noqa: F401
    from tensorboard.compat.proto.summary_pb2 import Summary
except ImportError:
    TEST_TENSORBOARD = False

HAS_TORCHVISION = True
try:
    import torchvision
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")

TEST_MATPLOTLIB = True
try:
    import matplotlib

    if os.environ.get("DISPLAY", "") == "":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    TEST_MATPLOTLIB = False
skipIfNoMatplotlib = unittest.skipIf(not TEST_MATPLOTLIB, "no matplotlib")

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    IS_MACOS,
    IS_WINDOWS,
    run_tests,
    skipIfTorchDynamo,
    TEST_WITH_CROSSREF,
    TestCase,
    xfailIfS390X,
)


def tensor_N(shape, dtype=float):
    # [FUNCTION DOC] HELPER: Builds a deterministic numpy array of the given shape holding the
    # consecutive values from zero up to the element count minus one, cast to dtype (default
    # float). Used as predictable image/audio/video input for the summary tests.
    numel = np.prod(shape)
    x = (np.arange(numel, dtype=dtype)).reshape(shape)
    return x


class BaseTestCase(TestCase):
    # [CLASS DOC] Base class for all TensorBoard tests. setUp skips the test when tensorboard
    # is missing or when running under crossref, otherwise creates the temp_dirs list that
    # records directories made for writers; tearDown deletes them. Also offers proto comparison
    # helpers against golden expect files, with an accept mode that rewrites them.
    """Base class used for all TensorBoard tests"""

    def setUp(self):
        # [FUNCTION DOC] Runs the parent setUp, then skips the test if tensorboard is not
        # installed or crossref mode is on; otherwise initializes the temp_dirs list to empty.
        super().setUp()
        if not TEST_TENSORBOARD:
            return self.skipTest("Skip the test since TensorBoard is not installed")
        if TEST_WITH_CROSSREF:
            return self.skipTest("Don't run TensorBoard tests with crossref")
        self.temp_dirs = []

    def createSummaryWriter(self):
        # [FUNCTION DOC] Creates a SummaryWriter logging to a freshly generated temporary
        # directory path (only the name is used), records that path in temp_dirs for later
        # cleanup, and returns the writer. Callers typically use it as a context manager.
        # Just to get the name of the directory in a writable place. tearDown()
        # is responsible for clean-ups.
        temp_dir = tempfile.TemporaryDirectory(prefix="test_tensorboard").name
        self.temp_dirs.append(temp_dir)
        return SummaryWriter(temp_dir)

    def tearDown(self):
        # [FUNCTION DOC] Runs the parent tearDown, then removes every directory recorded in
        # temp_dirs that still exists, deleting its whole tree.
        super().tearDown()
        # Remove directories created by SummaryWriter
        for temp_dir in self.temp_dirs:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def assertProto(self, actual_proto):
        # [FUNCTION DOC] Compares a Summary proto against the golden expect file of the running
        # test. In expecttest accept mode it writes the proto to that file and returns True.
        # Otherwise it parses the expected text into a Summary and asserts equality; raises if
        # the expect file is missing or the protos differ.
        if expecttest.ACCEPT:
            write_proto(actual_proto, self)
            return True
        expected_str = read_expected_content(self)
        expected_proto = Summary()
        text_format.Parse(expected_str, expected_proto)
        self.assertEqual(actual_proto, expected_proto)

    def assertImageProto(self, actual_proto):
        # [FUNCTION DOC] Like assertProto but for image summaries. In accept mode it writes the
        # proto text to the expect file and returns True. Otherwise it parses the expected
        # Summary and compares the first value of each: tag, image height, width and colorspace
        # must be equal, and the decoded PIL images must be equal (compared via assertEqual).
        if expecttest.ACCEPT:
            expected_file = get_expected_file(self)
            with open(expected_file, "w") as f:
                f.write(text_format.MessageToString(actual_proto))
            return True
        expected_str = read_expected_content(self)
        expected_proto = Summary()
        text_format.Parse(expected_str, expected_proto)

        [actual, expected] = [actual_proto.value[0], expected_proto.value[0]]
        actual_img = Image.open(io.BytesIO(actual.image.encoded_image_string))
        expected_img = Image.open(io.BytesIO(expected.image.encoded_image_string))

        self.assertEqual(actual.tag, expected.tag)
        self.assertEqual(actual.image.height, expected.image.height)
        self.assertEqual(actual.image.width, expected.image.width)
        self.assertEqual(actual.image.colorspace, expected.image.colorspace)
        self.assertEqual(actual_img, expected_img)


if TEST_TENSORBOARD:
    from google.protobuf import text_format
    from PIL import Image
    from tensorboard.compat.proto.graph_pb2 import GraphDef
    from tensorboard.compat.proto.types_pb2 import DataType

    from torch.utils.tensorboard import summary, SummaryWriter
    from torch.utils.tensorboard._convert_np import make_np
    from torch.utils.tensorboard._pytorch_graph import graph
    from torch.utils.tensorboard._utils import _prepare_video, convert_to_HWC
    from torch.utils.tensorboard.summary import int_to_half, tensor_proto
else:
    # Dummy for parametrization
    class DataType:
    # [CLASS DOC] HELPER: Placeholder used only when tensorboard is not installed, so that
    # decorators referencing DataType members still evaluate. All its members are None.
        DT_FLOAT, DT_HALF, DT_BFLOAT16, DT_INT32 = [None] * 4


class TestTensorBoardPyTorchNumpy(BaseTestCase):
    # [CLASS DOC] Tests for converting PyTorch tensors to numpy for TensorBoard and for the
    # histogram writer entry points (regular and raw buckets).
    def test_pytorch_autograd_np(self):
        # [FUNCTION DOC] Checks that make_np converts a legacy autograd Variable wrapping a
        # tensor into a numpy ndarray.
        x = torch.autograd.Variable(torch.empty(1))
        self.assertIsInstance(make_np(x), np.ndarray)

    def test_pytorch_write(self):
        # [FUNCTION DOC] Smoke test: writes a scalar derived from an autograd Variable through
        # a temporary SummaryWriter; passes if nothing raises.
        with self.createSummaryWriter() as w:
            w.add_scalar("scalar", torch.autograd.Variable(torch.rand(1)), 0)

    def test_pytorch_histogram(self):
        # [FUNCTION DOC] Smoke test: logs float, int and bfloat16 tensors as histograms; passes
        # if nothing raises.
        with self.createSummaryWriter() as w:
            w.add_histogram("float histogram", torch.rand((50,)))
            w.add_histogram("int histogram", torch.randint(0, 100, (50,)))
            w.add_histogram("bfloat16 histogram", torch.rand(50, dtype=torch.bfloat16))

    def test_pytorch_histogram_raw(self):
        # [FUNCTION DOC] Smoke test of the raw histogram API. Builds bucket limits and counts
        # with numpy for float data and int data, and with torch histc for a 0..99 range, and
        # passes min, max, count, sum, sum of squares and buckets to add_histogram_raw three
        # times. Passes if no call raises.
        with self.createSummaryWriter() as w:
            num = 50
            floats = make_np(torch.rand((num,)))
            bins = [0.0, 0.25, 0.5, 0.75, 1.0]
            counts, limits = np.histogram(floats, bins)
            sum_sq = floats.dot(floats).item()
            w.add_histogram_raw(
                "float histogram raw",
                min=floats.min().item(),
                max=floats.max().item(),
                num=num,
                sum=floats.sum().item(),
                sum_squares=sum_sq,
                bucket_limits=limits[1:].tolist(),
                bucket_counts=counts.tolist(),
            )

            ints = make_np(torch.randint(0, 100, (num,)))
            bins = [0, 25, 50, 75, 100]
            counts, limits = np.histogram(ints, bins)
            sum_sq = ints.dot(ints).item()
            w.add_histogram_raw(
                "int histogram raw",
                min=ints.min().item(),
                max=ints.max().item(),
                num=num,
                sum=ints.sum().item(),
                sum_squares=sum_sq,
                bucket_limits=limits[1:].tolist(),
                bucket_counts=counts.tolist(),
            )

            ints = torch.tensor(range(100)).float()
            nbins = 100
            counts = torch.histc(ints, bins=nbins, min=0, max=99)
            limits = torch.tensor(range(nbins))
            sum_sq = ints.dot(ints).item()
            w.add_histogram_raw(
                "int histogram raw",
                min=ints.min().item(),
                max=ints.max().item(),
                num=num,
                sum=ints.sum().item(),
                sum_squares=sum_sq,
                bucket_limits=limits.tolist(),
                bucket_counts=counts.tolist(),
            )


class TestTensorBoardPyTorchNumpyDevice(BaseTestCase):
    # [CLASS DOC] Device-parametrized tests of tensor to numpy conversion. Instantiated for
    # available devices at the bottom of the file, with MPS and XPU allowed.
    def test_pytorch_np(self, device):
        # [FUNCTION DOC] Checks that make_np returns a numpy ndarray for tensors of several
        # ranks created on the device, for Variable-wrapped versions of them, and for plain
        # Python int and float values.
        tensors = [
            torch.rand(3, 10, 10, device=device),
            torch.rand(1, device=device),
            torch.rand(1, 2, 3, 4, 5, device=device),
        ]
        for tensor in tensors:
            self.assertIsInstance(make_np(tensor), np.ndarray)
            self.assertIsInstance(make_np(torch.autograd.Variable(tensor)), np.ndarray)

        # python primitive type (device-agnostic)
        self.assertIsInstance(make_np(0), np.ndarray)
        self.assertIsInstance(make_np(0.1), np.ndarray)


class TestTensorBoardUtils(BaseTestCase):
    # [CLASS DOC] Tests for helper utilities in the tensorboard utils module: layout conversion
    # to height-width-channel and video preparation.
    def test_to_HWC(self):
        # [FUNCTION DOC] Checks convert_to_HWC output shapes: channel-first 3x32x32 becomes
        # 32x32x3, a batch of 16 images in NCHW becomes a tiled 64x256x3 grid, and a 2D
        # grayscale image becomes 32x32x3.
        test_image = np.random.randint(0, 256, size=(3, 32, 32), dtype=np.uint8)
        converted = convert_to_HWC(test_image, "chw")
        self.assertEqual(converted.shape, (32, 32, 3))
        test_image = np.random.randint(0, 256, size=(16, 3, 32, 32), dtype=np.uint8)
        converted = convert_to_HWC(test_image, "nchw")
        self.assertEqual(converted.shape, (64, 256, 3))
        test_image = np.random.randint(0, 256, size=(32, 32), dtype=np.uint8)
        converted = convert_to_HWC(test_image, "hw")
        self.assertEqual(converted.shape, (32, 32, 3))

    def test_convert_to_HWC_dtype_remains_same(self):
        # [FUNCTION DOC] Checks that after convert_to_HWC a uint8 tensor keeps its dtype, so
        # the scale factor computed by the summary module is exactly 1.
        # test to ensure convert_to_HWC restores the dtype of input np array and
        # thus the scale_factor calculated for the image is 1
        test_image = torch.tensor([[[[1, 2, 3], [4, 5, 6]]]], dtype=torch.uint8)
        tensor = make_np(test_image)
        tensor = convert_to_HWC(tensor, "NCHW")
        scale_factor = summary._calc_scale_factor(tensor)
        self.assertEqual(
            scale_factor,
            1,
            msg="Values are already in [0, 255], scale factor should be 1",
        )

    def test_prepare_video(self):
        # [FUNCTION DOC] For several 5D video shapes, runs _prepare_video on a copy and checks
        # that, for each time frame, the sum of all values after preparation matches the sum of
        # the corresponding frame of the original (axes 0 and 1 swapped).
        # At each timeframe, the sum over all other
        # dimensions of the video should be the same.
        shapes = [
            (16, 30, 3, 28, 28),
            (36, 30, 3, 28, 28),
            (19, 29, 3, 23, 19),
            (3, 3, 3, 3, 3),
        ]
        for s in shapes:
            V_input = np.random.random(s)
            V_after = _prepare_video(np.copy(V_input))
            total_frame = s[1]
            V_input = np.swapaxes(V_input, 0, 1)
            for f in range(total_frame):
                x = np.reshape(V_input[f], -1)
                y = np.reshape(V_after[f], -1)
                np.testing.assert_array_almost_equal(np.sum(x), np.sum(y))

    def test_numpy_vid_uint8(self):
        # [FUNCTION DOC] Same frame-sum check as the video preparation test but on a random
        # uint8 video; the prepared output is multiplied by 255 before the sums are compared.
        V_input = np.random.randint(0, 256, (16, 30, 3, 28, 28)).astype(np.uint8)
        V_after = _prepare_video(np.copy(V_input)) * 255
        total_frame = V_input.shape[1]
        V_input = np.swapaxes(V_input, 0, 1)
        for f in range(total_frame):
            x = np.reshape(V_input[f], -1)
            y = np.reshape(V_after[f], -1)
            np.testing.assert_array_almost_equal(np.sum(x), np.sum(y))


freqs = [262, 294, 330, 349, 392, 440, 440, 440, 440, 440, 440]

true_positive_counts = [75, 64, 21, 5, 0]
false_positive_counts = [150, 105, 18, 0, 0]
true_negative_counts = [0, 45, 132, 150, 150]
false_negative_counts = [0, 11, 54, 70, 75]
precision = [0.3333333, 0.3786982, 0.5384616, 1.0, 0.0]
recall = [1.0, 0.8533334, 0.28, 0.0666667, 0.0]


class TestTensorBoardWriter(BaseTestCase):
    # [CLASS DOC] End-to-end smoke test that exercises many SummaryWriter add methods.
    def test_writer(self):
        # [FUNCTION DOC] Smoke test: within a temporary writer, logs hparams, scalars (default,
        # custom walltime, new style), grouped scalars, images, image with boxes, audio, video,
        # text (plain and markdown), histogram, PR curve (computed and raw counts) and a mesh.
        # Uses the module level PR-curve count lists. Passes if nothing raises.
        with self.createSummaryWriter() as writer:
            sample_rate = 44100

            n_iter = 0
            writer.add_hparams(
                {"lr": 0.1, "bsize": 1}, {"hparam/accuracy": 10, "hparam/loss": 10}
            )
            writer.add_scalar("data/scalar_systemtime", 0.1, n_iter)
            writer.add_scalar("data/scalar_customtime", 0.2, n_iter, walltime=n_iter)
            writer.add_scalar("data/new_style", 0.2, n_iter, new_style=True)
            writer.add_scalars(
                "data/scalar_group",
                {
                    "xsinx": n_iter * np.sin(n_iter),
                    "xcosx": n_iter * np.cos(n_iter),
                    "arctanx": np.arctan(n_iter),
                },
                n_iter,
            )
            x = np.zeros((32, 3, 64, 64))  # output from network
            writer.add_images("Image", x, n_iter)  # Tensor
            writer.add_image_with_boxes(
                "imagebox",
                np.zeros((3, 64, 64)),
                np.array([[10, 10, 40, 40], [40, 40, 60, 60]]),
                n_iter,
            )
            x = np.zeros(sample_rate * 2)

            writer.add_audio("myAudio", x, n_iter)
            writer.add_video(
                "myVideo", np.random.rand(16, 48, 1, 28, 28).astype(np.float32), n_iter
            )
            writer.add_text("Text", "text logged at step:" + str(n_iter), n_iter)
            writer.add_text("markdown Text", """a|b\n-|-\nc|d""", n_iter)
            writer.add_histogram("hist", np.random.rand(100, 100), n_iter)
            writer.add_pr_curve(
                "xoxo", np.random.randint(2, size=100), np.random.rand(100), n_iter
            )  # needs tensorboard 0.4RC or later
            writer.add_pr_curve_raw(
                "prcurve with raw data",
                true_positive_counts,
                false_positive_counts,
                true_negative_counts,
                false_negative_counts,
                precision,
                recall,
                n_iter,
            )

            v = np.array(
                [[[1, 1, 1], [-1, -1, 1], [1, -1, -1], [-1, 1, -1]]], dtype=float
            )
            c = np.array(
                [[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 255]]], dtype=int
            )
            f = np.array([[[0, 2, 3], [0, 3, 1], [0, 1, 2], [1, 3, 2]]], dtype=int)
            writer.add_mesh("my_mesh", vertices=v, colors=c, faces=f)


class TestTensorBoardSummaryWriter(BaseTestCase):
    # [CLASS DOC] Tests for SummaryWriter lifecycle: context manager closing, repeated
    # open/close and construction from a pathlib path.
    def test_summary_writer_ctx(self):
        # [FUNCTION DOC] Checks that after leaving the with block the writer's file_writer
        # attribute is None, i.e. the writer was closed.
        # after using a SummaryWriter as a ctx it should be closed
        with self.createSummaryWriter() as writer:
            writer.add_scalar("test", 1)
        self.assertIs(writer.file_writer, None)

    def test_summary_writer_close(self):
        # [FUNCTION DOC] Creates and closes a writer; fails if an OSError (for example too many
        # open files) is raised.
        # Opening and closing SummaryWriter a lot should not run into
        # OSError: [Errno 24] Too many open files
        passed = True
        try:
            writer = self.createSummaryWriter()
            writer.close()
        except OSError:
            passed = False

        self.assertTrue(passed)

    def test_pathlib(self):
        # [FUNCTION DOC] Checks that SummaryWriter accepts a pathlib Path for the log directory
        # and can log a scalar to it inside a temporary directory.
        with tempfile.TemporaryDirectory(prefix="test_tensorboard_pathlib") as d:
            p = Path(d)
            with SummaryWriter(p) as writer:
                writer.add_scalar("test", 1)


class TestTensorBoardEmbedding(BaseTestCase):
    # [CLASS DOC] Tests for SummaryWriter.add_embedding.
    def test_embedding(self):
        # [FUNCTION DOC] Smoke test: adds an embedding of 3 feature vectors with tensor labels
        # and an image tensor, then again with labels paired with dataset strings and a
        # metadata header. No assertions beyond not raising.
        w = self.createSummaryWriter()
        all_features = torch.tensor([[1.0, 2.0, 3.0], [5.0, 4.0, 1.0], [3.0, 7.0, 7.0]])
        all_labels = torch.tensor([33.0, 44.0, 55.0])
        all_images = torch.zeros(3, 3, 5, 5)

        w.add_embedding(
            all_features, metadata=all_labels, label_img=all_images, global_step=2
        )

        dataset_label = ["test"] * 2 + ["train"] * 2
        all_labels = list(zip(all_labels, dataset_label))
        w.add_embedding(
            all_features,
            metadata=all_labels,
            label_img=all_images,
            metadata_header=["digit", "dataset"],
            global_step=2,
        )
        # assert...

    def test_embedding_64(self):
        # [FUNCTION DOC] Same as the embedding test but the label images are float64; passes if
        # nothing raises.
        w = self.createSummaryWriter()
        all_features = torch.tensor([[1.0, 2.0, 3.0], [5.0, 4.0, 1.0], [3.0, 7.0, 7.0]])
        all_labels = torch.tensor([33.0, 44.0, 55.0])
        all_images = torch.zeros((3, 3, 5, 5), dtype=torch.float64)

        w.add_embedding(
            all_features, metadata=all_labels, label_img=all_images, global_step=2
        )

        dataset_label = ["test"] * 2 + ["train"] * 2
        all_labels = list(zip(all_labels, dataset_label))
        w.add_embedding(
            all_features,
            metadata=all_labels,
            label_img=all_images,
            metadata_header=["digit", "dataset"],
            global_step=2,
        )


class TestTensorBoardSummary(BaseTestCase):
    # [CLASS DOC] Tests of the summary module builders (images, video, audio, text, histograms,
    # mesh, scalar). Most compare results to golden expect files via the base class proto
    # assertions.
    def test_uint8_image(self):
        # [FUNCTION DOC] Checks that the scale factor for a uint8 image is 1.
        """
        Tests that uint8 image (pixel values in [0, 255]) is not changed
        """
        test_image = np.random.randint(0, 256, size=(3, 32, 32), dtype=np.uint8)
        scale_factor = summary._calc_scale_factor(test_image)
        self.assertEqual(
            scale_factor,
            1,
            msg="Values are already in [0, 255], scale factor should be 1",
        )

    def test_float32_image(self):
        # [FUNCTION DOC] Checks that the scale factor for a float32 image with values in zero
        # to one is 255.
        """
        Tests that float32 image (pixel values in [0, 1]) are scaled correctly
        to [0, 255]
        """
        test_image = np.random.rand(3, 32, 32).astype(np.float32)
        scale_factor = summary._calc_scale_factor(test_image)
        self.assertEqual(
            scale_factor, 255, msg="Values are in [0, 1], scale factor should be 255"
        )

    def test_list_input(self):
        # [FUNCTION DOC] Checks that building a histogram summary from a plain Python list
        # raises an exception.
        with self.assertRaises(Exception):
            summary.histogram("dummy", [1, 3, 4, 5, 6], "tensorflow")

    def test_empty_input(self):
        # [FUNCTION DOC] Checks that building a histogram summary from an empty numpy array
        # raises an exception.
        with self.assertRaises(Exception):
            summary.histogram("dummy", np.ndarray(0), "tensorflow")

    def test_image_with_boxes(self):
        # [FUNCTION DOC] Compares the image-with-boxes summary of a 3x32x32 array and one box
        # against the golden image proto.
        self.assertImageProto(
            summary.image_boxes(
                "dummy", tensor_N(shape=(3, 32, 32)), np.array([[10, 10, 40, 40]])
            )
        )

    def test_image_with_one_channel(self):
        # [FUNCTION DOC] Compares the summary of a single-channel CHW image against the golden
        # image proto.
        self.assertImageProto(
            summary.image("dummy", tensor_N(shape=(1, 8, 8)), dataformats="CHW")
        )

    def test_image_with_one_channel_batched(self):
        # [FUNCTION DOC] Compares the summary of a batch of two single-channel NCHW images
        # against the golden image proto.
        self.assertImageProto(
            summary.image("dummy", tensor_N(shape=(2, 1, 8, 8)), dataformats="NCHW")
        )

    def test_image_with_3_channel_batched(self):
        # [FUNCTION DOC] Compares the summary of a batch of two 3-channel NCHW images against
        # the golden image proto.
        self.assertImageProto(
            summary.image("dummy", tensor_N(shape=(2, 3, 8, 8)), dataformats="NCHW")
        )

    def test_image_without_channel(self):
        # [FUNCTION DOC] Compares the summary of an 8x8 image with no channel axis, format HW,
        # against the golden image proto.
        self.assertImageProto(
            summary.image("dummy", tensor_N(shape=(8, 8)), dataformats="HW")
        )

    def test_video(self):
        # [FUNCTION DOC] Returns silently if moviepy cannot be imported. Otherwise compares the
        # video summary of a small tensor to its golden proto and builds two more video
        # summaries from random data as a smoke test.
        try:
            import moviepy  # noqa: F401
        except ImportError:
            return

        self.assertProto(summary.video("dummy", tensor_N(shape=(4, 3, 1, 8, 8))))

        summary.video("dummy", np.random.rand(16, 48, 1, 28, 28))
        summary.video("dummy", np.random.rand(20, 7, 1, 8, 8))

    @xfailIfS390X
    def test_audio(self):
        # [FUNCTION DOC] Compares the audio summary of a 42-sample signal against the golden
        # proto. Marked as an expected failure on s390x.
        self.assertProto(summary.audio("dummy", tensor_N(shape=(42,))))

    def test_text(self):
        # [FUNCTION DOC] Compares the text summary of a short string against the golden proto.
        self.assertProto(summary.text("dummy", "text 123"))

    def test_histogram_auto(self):
        # [FUNCTION DOC] Compares a histogram summary with the auto binning rule and at most 5
        # bins against the golden proto.
        self.assertProto(
            summary.histogram("dummy", tensor_N(shape=(1024,)), bins="auto", max_bins=5)
        )

    def test_histogram_fd(self):
        # [FUNCTION DOC] Compares a histogram summary using the fd binning rule and at most 5
        # bins against the golden proto.
        self.assertProto(
            summary.histogram("dummy", tensor_N(shape=(1024,)), bins="fd", max_bins=5)
        )

    def test_histogram_doane(self):
        # [FUNCTION DOC] Compares a histogram summary using the doane binning rule and at most
        # 5 bins against the golden proto.
        self.assertProto(
            summary.histogram(
                "dummy", tensor_N(shape=(1024,)), bins="doane", max_bins=5
            )
        )

    def test_custom_scalars(self):
        # [FUNCTION DOC] Smoke test: builds a custom scalars summary from a nested layout of
        # categories and chart definitions; the output is not compared.
        layout = {
            "Taiwan": {"twse": ["Multiline", ["twse/0050", "twse/2330"]]},
            "USA": {
                "dow": ["Margin", ["dow/aaa", "dow/bbb", "dow/ccc"]],
                "nasdaq": ["Margin", ["nasdaq/aaa", "nasdaq/bbb", "nasdaq/ccc"]],
            },
        }
        summary.custom_scalars(
            layout
        )  # only smoke test. Because protobuf in python2/3 serialize dictionary differently.

    def test_mesh(self):
        # [FUNCTION DOC] Builds a mesh summary from one small set of vertices, colors and faces
        # and compares it with the golden proto.
        v = np.array([[[1, 1, 1], [-1, -1, 1], [1, -1, -1], [-1, 1, -1]]], dtype=float)
        c = np.array(
            [[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 255]]], dtype=int
        )
        f = np.array([[[0, 2, 3], [0, 3, 1], [0, 1, 2], [1, 3, 2]]], dtype=int)
        mesh = summary.mesh("my_mesh", vertices=v, colors=c, faces=f, config_dict=None)
        self.assertProto(mesh)

    def test_scalar_new_style(self):
        # [FUNCTION DOC] Checks that a new-style scalar summary of 1.0 matches the golden
        # proto, and that new-style scalar with a three-element tensor raises AssertionError.
        scalar = summary.scalar("test_scalar", 1.0, new_style=True)
        self.assertProto(scalar)
        with self.assertRaises(AssertionError):
            summary.scalar("test_scalar2", torch.Tensor([1, 2, 3]), new_style=True)


def remove_whitespace(string):
    # [FUNCTION DOC] HELPER: Returns the string with all spaces, tabs and newlines removed.
    return string.replace(" ", "").replace("\t", "").replace("\n", "")


def get_expected_file(function_ptr):
    # [FUNCTION DOC] HELPER: Given a running test case, returns the path of its golden file:
    # the expect directory next to the (symlink-resolved) test source file, with a name made of
    # the fixed prefix TestTensorBoard, the test method name and the expect suffix.
    module_id = function_ptr.__class__.__module__
    test_file = sys.modules[module_id].__file__
    # Look for the .py file (since __file__ could be pyc).
    test_file = ".".join(test_file.split(".")[:-1]) + ".py"

    # Use realpath to follow symlinks appropriately.
    test_dir = os.path.dirname(os.path.realpath(test_file))
    functionName = function_ptr.id().split(".")[-1]
    return os.path.join(
        test_dir, "expect", "TestTensorBoard." + functionName + ".expect"
    )


def read_expected_content(function_ptr):
    # [FUNCTION DOC] HELPER: Reads and returns the golden file text of the given test. Raises
    # AssertionError if the file does not exist.
    expected_file = get_expected_file(function_ptr)
    if not os.path.exists(expected_file):
        raise AssertionError(f"expected file does not exist: {expected_file}")
    with open(expected_file) as f:
        return f.read()


def write_proto(str_to_compare, function_ptr):
    # [FUNCTION DOC] HELPER: Overwrites the test's golden file with the string form of the
    # given object (used in accept mode).
    expected_file = get_expected_file(function_ptr)
    with open(expected_file, "w") as f:
        f.write(str(str_to_compare))


class TestTensorBoardPytorchGraph(BaseTestCase):
    # [CLASS DOC] Tests for graph export through add_graph and the graph conversion function.
    def test_pytorch_graph(self):
        # [FUNCTION DOC] Exports a small linear module via a writer, then converts it with
        # graph and compares against the golden GraphDef: same node count, and per node the
        # same name, op, inputs, device and attribute key set.
        dummy_input = (torch.zeros(1, 3),)

        class myLinear(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one linear layer from 3 to 5 features.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the module and its linear layer.
                super().__init__()
                self.l = torch.nn.Linear(3, 5)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer to the input and returns the
                # result.
                return self.l(x)

        with self.createSummaryWriter() as w:
            w.add_graph(myLinear(), dummy_input)

        actual_proto, _ = graph(myLinear(), dummy_input)

        expected_str = read_expected_content(self)
        expected_proto = GraphDef()
        text_format.Parse(expected_str, expected_proto)

        self.assertEqual(len(expected_proto.node), len(actual_proto.node))
        for i in range(len(expected_proto.node)):
            expected_node = expected_proto.node[i]
            actual_node = actual_proto.node[i]
            self.assertEqual(expected_node.name, actual_node.name)
            self.assertEqual(expected_node.op, actual_node.op)
            self.assertEqual(expected_node.input, actual_node.input)
            self.assertEqual(expected_node.device, actual_node.device)
            self.assertEqual(
                sorted(expected_node.attr.keys()), sorted(actual_node.attr.keys())
            )

    def test_nested_nn_squential(self):
        # [FUNCTION DOC] Exports nested Sequential modules (an outer Sequential of inner ones
        # containing two linears) via a writer, then compares the graph conversion to the
        # golden GraphDef with the same per-node checks as the basic graph test.
        dummy_input = torch.randn(2, 3)

        class InnerNNSquential(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping a Sequential of two linear layers going dim1
            # to dim2 and back.
            def __init__(self, dim1, dim2):
                # [FUNCTION DOC] HELPER: Builds the inner Sequential from the two dimensions.
                super().__init__()
                self.inner_nn_squential = torch.nn.Sequential(
                    torch.nn.Linear(dim1, dim2),
                    torch.nn.Linear(dim2, dim1),
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Runs the input through the inner Sequential and
                # returns it.
                x = self.inner_nn_squential(x)
                return x

        class OuterNNSquential(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a Sequential made of depth copies of the inner
            # module.
            def __init__(self, dim1=3, dim2=4, depth=2):
                # [FUNCTION DOC] HELPER: Builds the list of inner modules (defaults dims 3 and
                # 4, depth 2) and wraps them in a Sequential.
                super().__init__()
                layers = []
                for _ in range(depth):
                    layers.append(InnerNNSquential(dim1, dim2))
                self.outer_nn_squential = torch.nn.Sequential(*layers)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Runs the input through the outer Sequential and
                # returns it.
                x = self.outer_nn_squential(x)
                return x

        with self.createSummaryWriter() as w:
            w.add_graph(OuterNNSquential(), dummy_input)

        actual_proto, _ = graph(OuterNNSquential(), dummy_input)

        expected_str = read_expected_content(self)
        expected_proto = GraphDef()
        text_format.Parse(expected_str, expected_proto)

        self.assertEqual(len(expected_proto.node), len(actual_proto.node))
        for i in range(len(expected_proto.node)):
            expected_node = expected_proto.node[i]
            actual_node = actual_proto.node[i]
            self.assertEqual(expected_node.name, actual_node.name)
            self.assertEqual(expected_node.op, actual_node.op)
            self.assertEqual(expected_node.input, actual_node.input)
            self.assertEqual(expected_node.device, actual_node.device)
            self.assertEqual(
                sorted(expected_node.attr.keys()), sorted(actual_node.attr.keys())
            )

    def test_pytorch_graph_dict_input(self):
        # [FUNCTION DOC] Checks add_graph with a model that returns a tensor (works with and
        # without strict tracing) and a model that returns a dict (raises RuntimeError with
        # strict tracing on, works with strict tracing off).
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one linear layer returning a tensor.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear layer from 3 to 5 features.
                super().__init__()
                self.l = torch.nn.Linear(3, 5)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the linear layer applied to the input.
                return self.l(x)

        class ModelDict(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one linear layer whose output is wrapped in a
            # dict.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear layer from 3 to 5 features.
                super().__init__()
                self.l = torch.nn.Linear(3, 5)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a mapping with key out holding the linear
                # layer output.
                return {"out": self.l(x)}

        dummy_input = torch.zeros(1, 3)

        with self.createSummaryWriter() as w:
            w.add_graph(Model(), dummy_input)

        with self.createSummaryWriter() as w:
            w.add_graph(Model(), dummy_input, use_strict_trace=True)

        # expect error: Encountering a dict at the output of the tracer...
        with self.assertRaises(RuntimeError):
            with self.createSummaryWriter() as w:
                w.add_graph(ModelDict(), dummy_input, use_strict_trace=True)

        with self.createSummaryWriter() as w:
            w.add_graph(ModelDict(), dummy_input, use_strict_trace=False)

    def test_mlp_graph(self):
        # [FUNCTION DOC] Checks that add_graph on an MLP known to fail JIT optimization does
        # not raise (see the linked upstream issue in the comment above the class).
        dummy_input = (torch.zeros(2, 1, 28, 28),)

        # This MLP class with the above input is expected
        # to fail JIT optimizations as seen at
        # https://github.com/pytorch/pytorch/issues/18903
        #
        # However, it should not raise an error during
        # the add_graph call and still continue.
        class myMLP(torch.nn.Module):
            # [CLASS DOC] HELPER: Three layer MLP (flattened 28x28 input, two hidden layers of
            # 1200, 10 outputs) with relu activations; forward takes an unused extra flag.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the flattened input length and three linear
                # layers.
                super().__init__()
                self.input_len = 1 * 28 * 28
                self.fc1 = torch.nn.Linear(self.input_len, 1200)
                self.fc2 = torch.nn.Linear(1200, 1200)
                self.fc3 = torch.nn.Linear(1200, 10)

            def forward(self, x, update_batch_stats=True):
                # [FUNCTION DOC] HELPER: Flattens the input, applies relu after the first two
                # linears and returns the third linear output. The batch statistics flag is
                # unused.
                h = torch.nn.functional.relu(self.fc1(x.view(-1, self.input_len)))
                h = self.fc2(h)
                h = torch.nn.functional.relu(h)
                h = self.fc3(h)
                return h

        with self.createSummaryWriter() as w:
            w.add_graph(myMLP(), dummy_input)

    def test_wrong_input_size(self):
        # [FUNCTION DOC] Checks that add_graph with a linear 3 to 5 model and an input with 9
        # features raises RuntimeError.
        with self.assertRaises(RuntimeError):
            dummy_input = torch.rand(1, 9)
            model = torch.nn.Linear(3, 5)
            with self.createSummaryWriter() as w:
                w.add_graph(model, dummy_input)  # error

    @skipIfNoTorchVision
    def test_torchvision_smoke(self):
        # [FUNCTION DOC] Skipped without torchvision. For several torchvision models, builds
        # each with its default constructor and exports its graph with a zero input of batch 2
        # and 224x224 images; passes if nothing raises.
        model_input_shapes = {
            "alexnet": (2, 3, 224, 224),
            "resnet34": (2, 3, 224, 224),
            "resnet152": (2, 3, 224, 224),
            "densenet121": (2, 3, 224, 224),
            "vgg16": (2, 3, 224, 224),
            "vgg19": (2, 3, 224, 224),
            "vgg16_bn": (2, 3, 224, 224),
            "vgg19_bn": (2, 3, 224, 224),
            "mobilenet_v2": (2, 3, 224, 224),
        }
        for model_name, input_shape in model_input_shapes.items():
            with self.createSummaryWriter() as w:
                model = getattr(torchvision.models, model_name)()
                w.add_graph(model, torch.zeros(input_shape))


class TestTensorBoardFigure(BaseTestCase):
    # [CLASS DOC] Tests for add_figure with matplotlib figures.
    @skipIfNoMatplotlib
    @skipIfTorchDynamo("dynamo fails to trace matplotlib WRITEABLE flag and slice.indices")
    def test_figure(self):
        # [FUNCTION DOC] Skipped without matplotlib or under Dynamo. Draws two circles, logs
        # the figure without closing it and checks it still exists, then logs it again with
        # default closing and checks it is gone (that check is skipped with a message on
        # matplotlib 3.3.0). Closes the writer.
        writer = self.createSummaryWriter()

        figure, axes = plt.figure(), plt.gca()
        circle1 = plt.Circle((0.2, 0.5), 0.2, color="r")
        circle2 = plt.Circle((0.8, 0.5), 0.2, color="g")
        axes.add_patch(circle1)
        axes.add_patch(circle2)
        plt.axis("scaled")
        plt.tight_layout()

        writer.add_figure("add_figure/figure", figure, 0, close=False)
        self.assertTrue(plt.fignum_exists(figure.number))

        writer.add_figure("add_figure/figure", figure, 1)
        if matplotlib.__version__ != "3.3.0":
            self.assertFalse(plt.fignum_exists(figure.number))
        else:
            print(
                "Skipping fignum_exists, see https://github.com/matplotlib/matplotlib/issues/18163"
            )

        writer.close()

    @skipIfNoMatplotlib
    @skipIfTorchDynamo("dynamo fails to trace matplotlib WRITEABLE flag and slice.indices")
    def test_figure_list(self):
        # [FUNCTION DOC] Skipped without matplotlib or under Dynamo. Creates five line plot
        # figures, logs the list without closing and checks all exist, then logs again with
        # default closing and checks none exist (check skipped on matplotlib 3.3.0). Closes the
        # writer.
        writer = self.createSummaryWriter()

        figures = []
        for i in range(5):
            figure = plt.figure()
            plt.plot([i * 1, i * 2, i * 3], label="Plot " + str(i))
            plt.xlabel("X")
            plt.xlabel("Y")
            plt.legend()
            plt.tight_layout()
            figures.append(figure)

        writer.add_figure("add_figure/figure_list", figures, 0, close=False)
        self.assertTrue(
            all(plt.fignum_exists(figure.number) is True for figure in figures)
        )

        writer.add_figure("add_figure/figure_list", figures, 1)
        if matplotlib.__version__ != "3.3.0":
            self.assertTrue(
                all(plt.fignum_exists(figure.number) is False for figure in figures)
            )
        else:
            print(
                "Skipping fignum_exists, see https://github.com/matplotlib/matplotlib/issues/18163"
            )

        writer.close()


class TestTensorBoardNumpy(BaseTestCase):
    # [CLASS DOC] Tests for make_np on Python and numpy scalars and unsupported inputs.
    def test_scalar(self):
        # [FUNCTION DOC] Checks that make_np returns a one-element ndarray for a float, a large
        # integer, a numpy float16, a numpy float128 (not on macOS or Windows) and a numpy
        # int64.
        res = make_np(1.1)
        self.assertIsInstance(res, np.ndarray) and self.assertEqual(res.shape, (1,))
        res = make_np(1 << 64 - 1)  # uint64_max
        self.assertIsInstance(res, np.ndarray) and self.assertEqual(res.shape, (1,))
        res = make_np(np.float16(1.00000087))
        self.assertIsInstance(res, np.ndarray) and self.assertEqual(res.shape, (1,))
        if not IS_MACOS and not IS_WINDOWS:
            res = make_np(np.float128(1.00008 + 9))
            self.assertIsInstance(res, np.ndarray) and self.assertEqual(res.shape, (1,))
        res = make_np(np.int64(100000000000))
        self.assertIsInstance(res, np.ndarray) and self.assertEqual(res.shape, (1,))

    def test_pytorch_np_expect_fail(self):
        # [FUNCTION DOC] Checks that make_np raises NotImplementedError for a dict input.
        with self.assertRaises(NotImplementedError):
            make_np({"pytorch": 1.0})


class TestTensorProtoSummary(BaseTestCase):
    # [CLASS DOC] Tests for tensor_proto conversions of various dtypes. Registered with the
    # parametrized test instantiation at the bottom of the file.
    @parametrize(
        "tensor_type,proto_type",
        [
            (torch.float16, DataType.DT_HALF),
            (torch.bfloat16, DataType.DT_BFLOAT16),
        ],
    )
    @skipIfTorchDynamo("Unsuitable test for Dynamo, behavior changes with version")
    def test_half_tensor_proto(self, tensor_type, proto_type):
        # [FUNCTION DOC] Parametrized over float16 and bfloat16 with matching proto types.
        # Builds the proto of a three-value tensor, decodes the half values with int_to_half,
        # and checks they equal the originals and the proto dtype matches. Skipped under
        # Dynamo.
        float_values = [1.0, 2.0, 3.0]
        actual_proto = (
            tensor_proto(
                "dummy",
                torch.tensor(float_values, dtype=tensor_type),
            )
            .value[0]
            .tensor
        )
        self.assertSequenceEqual(
            [int_to_half(x) for x in actual_proto.half_val],
            float_values,
        )
        self.assertTrue(actual_proto.dtype == proto_type)

    def test_float_tensor_proto(self):
        # [FUNCTION DOC] Checks that a float tensor proto stores the values in the float field
        # and has the float data type.
        float_values = [1.0, 2.0, 3.0]
        actual_proto = tensor_proto("dummy", torch.tensor(float_values)).value[0].tensor
        self.assertEqual(actual_proto.float_val, float_values)
        self.assertTrue(actual_proto.dtype == DataType.DT_FLOAT)

    def test_int_tensor_proto(self):
        # [FUNCTION DOC] Checks that an int32 tensor proto stores the values in the int field
        # and has the int32 data type.
        int_values = [1, 2, 3]
        actual_proto = (
            tensor_proto("dummy", torch.tensor(int_values, dtype=torch.int32))
            .value[0]
            .tensor
        )
        self.assertEqual(actual_proto.int_val, int_values)
        self.assertTrue(actual_proto.dtype == DataType.DT_INT32)

    def test_scalar_tensor_proto(self):
        # [FUNCTION DOC] Checks that a zero-dimension tensor proto holds the value,
        # approximately, as the first float value.
        scalar_value = 0.1
        actual_proto = tensor_proto("dummy", torch.tensor(scalar_value)).value[0].tensor
        self.assertAlmostEqual(actual_proto.float_val[0], scalar_value)

    def test_complex_tensor_proto(self):
        # [FUNCTION DOC] Checks that a complex tensor proto stores interleaved real and
        # imaginary parts in the single-precision complex field.
        real = torch.tensor([1.0, 2.0])
        imag = torch.tensor([3.0, 4.0])
        actual_proto = tensor_proto("dummy", torch.complex(real, imag)).value[0].tensor
        self.assertEqual(actual_proto.scomplex_val, [1.0, 3.0, 2.0, 4.0])

    def test_empty_tensor_proto(self):
        # [FUNCTION DOC] Checks that an empty tensor proto has no float values.
        actual_proto = tensor_proto("dummy", torch.empty(0)).value[0].tensor
        self.assertEqual(actual_proto.float_val, [])


instantiate_parametrized_tests(TestTensorProtoSummary)
instantiate_device_type_tests(
    TestTensorBoardPyTorchNumpyDevice,
    globals(),
    allow_mps=True,
    allow_xpu=True,
)

if __name__ == "__main__":
    run_tests()
