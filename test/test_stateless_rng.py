# Owner(s): ["module: random"]
# [MODULE DOC] Tests for the stateless (functional) random number API in torch.func._random: key
# creation, key splitting, folding data into keys, normal and uniform generation (functional and
# in-place), and torch.compile support. A key is a uint64 tensor of shape (2,) holding seed and
# offset, optionally with leading batch dims. Test classes are instantiated per device (cpu and
# cuda) at import time through instantiate_device_type_tests.

import torch
import torch.func._random as random
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    onlyAccelerator,
    onlyCUDA,
)
from torch.testing._internal.common_dtype import floating_types_and
from torch.testing._internal.common_utils import (
    parametrize,
    run_tests,
    subtest,
    TestCase,
)
from torch.testing._internal.inductor_utils import HAS_TRITON


all_floating_dtypes = floating_types_and(torch.half, torch.bfloat16)


class TestStatelessRNGKey(TestCase):
    # [CLASS DOC] Tests for the key constructor: result shape, dtype and device, seed sensitivity,
    # determinism, and the error for an unsupported generator implementation name.
    def test_basic_shape_and_dtype(self, device):
        # [FUNCTION DOC] Checks that a key from seed 42 has shape of two elements, dtype uint64
        # and lives on the requested device.
        key = random.key(42, device=device)
        self.assertEqual(key.shape, (2,))
        self.assertEqual(key.dtype, torch.uint64)
        self.assertEqual(key.device, torch.device(device))

    def test_different_seeds(self, device):
        # [FUNCTION DOC] Checks that keys built from two different seeds are not equal.
        key1 = random.key(42, device=device)
        key2 = random.key(43, device=device)
        self.assertNotEqual(key1, key2)

    def test_determinism(self, device):
        # [FUNCTION DOC] Checks that building a key twice from the same seed gives equal tensors.
        key1 = random.key(42, device=device)
        key2 = random.key(42, device=device)
        self.assertEqual(key1, key2)

    def test_error_unsupported_impl(self, device):
        # [FUNCTION DOC] Checks that requesting an unknown impl name raises NotImplementedError
        # saying the PRNG impl is not supported.
        with self.assertRaisesRegex(
            NotImplementedError, "does not support PRNG impl 'unsupported'"
        ):
            random.key(42, impl="unsupported", device=device)


class TestStatelessRNGKeySplit(TestCase):
    # [CLASS DOC] Tests for splitting a key into several child keys: output shape and dtype,
    # determinism, uniqueness, sensitivity to seed and offset, batched keys, input validation
    # errors, offset wraparound, and CPU versus accelerator agreement.
    def test_basic_shape_and_dtype(self, device):
        # [FUNCTION DOC] Checks that splitting one key four ways returns a uint64 tensor of shape
        # (4, 2) on the same device as the key.
        key = random.key(42, device=device)
        splits = random.split(key, 4)
        self.assertEqual(splits.shape, (4, 2))
        self.assertEqual(splits.dtype, torch.uint64)
        self.assertEqual(splits.device, key.device)

    def test_single_split(self, device):
        # [FUNCTION DOC] Checks that splitting into one child gives shape (1, 2).
        key = random.key(42, device=device)
        splits = random.split(key, 1)
        self.assertEqual(splits.shape, (1, 2))

    def test_large_num_splits(self, device):
        # [FUNCTION DOC] Checks that splitting into ten thousand children gives the matching
        # shape.
        key = random.key(42, device=device)
        splits = random.split(key, 10000)
        self.assertEqual(splits.shape, (10000, 2))

    def test_determinism(self, device):
        # [FUNCTION DOC] Checks that splitting the same key twice gives equal results.
        key = random.key(42, device=device)
        splits1 = random.split(key, 8)
        splits2 = random.split(key, 8)
        self.assertEqual(splits1, splits2)

    def test_all_keys_unique(self, device):
        # [FUNCTION DOC] Checks that one hundred children of a split are all distinct rows, by
        # counting unique rows.
        key = random.key(42, device=device)
        splits = random.split(key, 100)
        unique_keys = torch.unique(splits, dim=0)
        self.assertEqual(unique_keys.shape[0], 100)

    def test_different_seeds_produce_different_outputs(self, device):
        # [FUNCTION DOC] Checks that splitting keys made from different seeds gives different
        # outputs.
        key1 = random.key(42, device=device)
        key2 = random.key(43, device=device)
        splits1 = random.split(key1, 4)
        splits2 = random.split(key2, 4)
        self.assertNotEqual(splits1, splits2)

    def test_different_offsets_produce_different_outputs(self, device):
        # [FUNCTION DOC] Checks that a key and the same key folded with data one produce different
        # splits, i.e. the offset influences the split.
        key1 = random.key(42, device=device)
        key2 = random.fold_in(key1, 1)
        splits1 = random.split(key1, 4)
        splits2 = random.split(key2, 4)
        self.assertNotEqual(splits1, splits2)

    def test_offset_zero_vs_one_produce_different_splits(self, device):
        # [FUNCTION DOC] Checks that two keys with the same seed but offsets zero and one give
        # different splits.
        key1 = random.key(42, device=device)
        key2 = torch.tensor([42, 1], dtype=torch.uint64, device=device)
        splits1 = random.split(key1, 4)
        splits2 = random.split(key2, 4)
        self.assertNotEqual(splits1, splits2)

    def test_batched(self, device):
        # [FUNCTION DOC] Checks that splitting a batch of four keys into three yields shape (3, 4,
        # 2) and that each entry equals the split of the corresponding single key (split index
        # first, then key index).
        key = random.key(42, device=device)
        keys = random.split(key, 4)  # (4, 2)
        num_splits = 3
        batched = random.split(keys, num_splits)  # (3, 4, 2)
        self.assertEqual(batched.shape, (num_splits, 4, 2))
        for k in range(4):
            individual = random.split(keys[k], num_splits)
            for s in range(num_splits):
                self.assertEqual(batched[s][k], individual[s])

    def test_multi_batch(self, device):
        # [FUNCTION DOC] Same as the batched test but with a key batch of shape (3, 4): the result
        # has shape (5, 3, 4, 2) and every entry matches the split of the individual key.
        key = random.key(42, device=device)
        keys = random.split(key, 12).reshape(3, 4, 2)
        num_splits = 5
        batched = random.split(keys, num_splits)  # (5, 3, 4, 2)
        self.assertEqual(batched.shape, (num_splits, 3, 4, 2))
        for i in range(3):
            for j in range(4):
                individual = random.split(keys[i][j], num_splits)
                for s in range(num_splits):
                    self.assertEqual(batched[s][i][j], individual[s])

    def test_error_wrong_shape(self, device):
        # [FUNCTION DOC] Checks that a key with three elements is rejected with a RuntimeError
        # about the required key shape.
        key = torch.tensor([42, 0, 1], dtype=torch.uint64, device=device)
        with self.assertRaisesRegex(
            RuntimeError, r"key must have shape \(\*batch, 2\)"
        ):
            random.split(key, 4)

    def test_error_wrong_dtype(self, device):
        # [FUNCTION DOC] Checks that a float32 key is rejected with a RuntimeError about the
        # required uint64 dtype.
        key = torch.tensor([42, 0], dtype=torch.float32, device=device)
        with self.assertRaisesRegex(RuntimeError, "key must have dtype uint64"):
            random.split(key, 4)

    def test_error_invalid_num_splits(self, device):
        # [FUNCTION DOC] Checks that a split count of zero or a negative count raises a
        # RuntimeError saying the count must be positive.
        key = random.key(42, device=device)
        with self.assertRaisesRegex(RuntimeError, "num_splits must be positive"):
            random.split(key, 0)
        with self.assertRaisesRegex(RuntimeError, "num_splits must be positive"):
            random.split(key, -1)

    def test_error_batched_last_dim_not_2(self, device):
        # [FUNCTION DOC] Checks that a batch of keys whose last dimension is three instead of two
        # raises the key shape RuntimeError.
        key = torch.tensor([[42, 0, 1], [43, 0, 1]], dtype=torch.uint64, device=device)
        with self.assertRaisesRegex(
            RuntimeError, r"key must have shape \(\*batch, 2\)"
        ):
            random.split(key, 4)

    def test_offset_overflow(self, device):
        # [FUNCTION DOC] Uses a key whose offset is the largest uint64 value. Checks that the
        # second child equals folding data zero into a key with offset zero and the third equals
        # folding data one, i.e. the offset wraps around modulo 2 to the 64.
        near_max = (1 << 64) - 1
        key = torch.tensor([42, near_max], dtype=torch.uint64, device=device)
        splits = random.split(key, 3)
        # split_idx=1 wraps offset to 0, split_idx=2 wraps to 1
        key0 = torch.tensor([42, 0], dtype=torch.uint64, device=device)
        self.assertEqual(splits[1], random.fold_in(key0, 0))
        self.assertEqual(splits[2], random.fold_in(key0, 1))

    @parametrize("batched", [False, True])
    @onlyAccelerator
    def test_cross_device_consistency(self, device, batched):
        # [FUNCTION DOC] Accelerator-only, parametrized on whether the key is batched. Builds the
        # same key on CPU and on the device (optionally split into four) and checks that splitting
        # eight ways gives equal results on both after moving the device result to CPU.
        key_cpu = random.key(42)
        key_dev = random.key(42, device=device)
        if batched:
            # Batched key exercises the multi-key path.
            key_cpu = random.split(key_cpu, 4)  # (4, 2)
            key_dev = random.split(key_dev, 4)
        self.assertEqual(
            random.split(key_cpu, 8),
            random.split(key_dev, 8).cpu(),
        )


class TestStatelessRNGKeyFoldIn(TestCase):
    # [CLASS DOC] Tests for folding an integer or tensor datum into a key: result
    # shape/dtype/device, determinism, consistency with split, batched keys, validation errors,
    # uint64 wraparound and negative data handling, tensor data, CUDA graph replay, and CPU versus
    # accelerator agreement.
    def test_basic_shape_and_dtype(self, device):
        # [FUNCTION DOC] Checks that folding data seven into a key returns a uint64 tensor of
        # shape (2,) on the key's device.
        key = random.key(42, device=device)
        result = random.fold_in(key, 7)
        self.assertEqual(result.shape, (2,))
        self.assertEqual(result.dtype, torch.uint64)
        self.assertEqual(result.device, key.device)

    def test_determinism(self, device):
        # [FUNCTION DOC] Checks that folding the same data into the same key twice gives equal
        # results.
        key = random.key(42, device=device)
        result1 = random.fold_in(key, 7)
        result2 = random.fold_in(key, 7)
        self.assertEqual(result1, result2)

    def test_fold_in_produces_new_key_for_zero_data(self, device):
        # [FUNCTION DOC] Checks that folding data zero still changes the key (the result differs
        # from the input).
        key = random.key(42, device=device)
        folded = random.fold_in(key, 0)
        self.assertNotEqual(folded, key)

    def test_different_data_produces_different_outputs(self, device):
        # [FUNCTION DOC] Checks that folding data zero and data one into one key gives different
        # results.
        key = random.key(42, device=device)
        result1 = random.fold_in(key, 0)
        result2 = random.fold_in(key, 1)
        self.assertNotEqual(result1, result2)

    def test_consistency_with_split(self, device):
        # [FUNCTION DOC] Checks that folding data i into a key equals the i-th child of splitting
        # that key ten ways.
        key = random.key(42, device=device)
        splits = random.split(key, 10)
        for i in range(10):
            folded = random.fold_in(key, i)
            self.assertEqual(folded, splits[i])

    def test_batched(self, device):
        # [FUNCTION DOC] Checks that folding one datum into four batched keys gives shape (4, 2)
        # and each row equals folding into the single key.
        key = random.key(42, device=device)
        keys = random.split(key, 4)  # (4, 2)
        data = 7
        batched = random.fold_in(keys, data)  # (4, 2)
        self.assertEqual(batched.shape, (4, 2))
        for k in range(4):
            individual = random.fold_in(keys[k], data)
            self.assertEqual(batched[k], individual)

    def test_multi_batch(self, device):
        # [FUNCTION DOC] Same as the batched test with key batch shape (3, 4): the result has
        # shape (3, 4, 2) and matches per-key folding.
        key = random.key(42, device=device)
        keys = random.split(key, 12).reshape(3, 4, 2)
        data = 7
        batched = random.fold_in(keys, data)  # (3, 4, 2)
        self.assertEqual(batched.shape, (3, 4, 2))
        for i in range(3):
            for j in range(4):
                individual = random.fold_in(keys[i][j], data)
                self.assertEqual(batched[i][j], individual)

    def test_error_wrong_shape(self, device):
        # [FUNCTION DOC] Checks that a three-element key raises a RuntimeError about the key
        # shape.
        key = torch.tensor([42, 0, 1], dtype=torch.uint64, device=device)
        with self.assertRaisesRegex(
            RuntimeError, r"key must have shape \(\*batch, 2\)"
        ):
            random.fold_in(key, 0)

    def test_error_wrong_dtype(self, device):
        # [FUNCTION DOC] Checks that a float32 key raises a RuntimeError about the uint64 dtype.
        key = torch.tensor([42, 0], dtype=torch.float32, device=device)
        with self.assertRaisesRegex(RuntimeError, "key must have dtype uint64"):
            random.fold_in(key, 0)

    def test_error_batched_last_dim_not_2(self, device):
        # [FUNCTION DOC] Checks that a batch of keys whose last dimension is three raises the key
        # shape RuntimeError.
        key = torch.tensor([[42, 0, 1], [43, 0, 1]], dtype=torch.uint64, device=device)
        with self.assertRaisesRegex(
            RuntimeError, r"key must have shape \(\*batch, 2\)"
        ):
            random.fold_in(key, 0)

    def test_offset_overflow(self, device):
        # [FUNCTION DOC] Uses a key with the maximum uint64 offset. Checks that folding data one
        # wraps the offset to zero, giving the same result as folding data zero into the same seed
        # with offset zero.
        near_max = (1 << 64) - 1
        key = torch.tensor([42, near_max], dtype=torch.uint64, device=device)
        # fold_in(data=1) wraps offset to 0, so it should match fold_in on
        # a key with offset=0 and data=0.
        result = random.fold_in(key, 1)
        key0 = torch.tensor([42, 0], dtype=torch.uint64, device=device)
        self.assertEqual(result, random.fold_in(key0, 0))

    def test_data_above_int64_max(self, device):
        # [FUNCTION DOC] Checks that data equal to the largest uint64 value is accepted and that
        # folding it into a key with offset zero equals folding data zero into a key whose offset
        # is that value.
        # data is interpreted as uint64; values above int64 max must be accepted.
        data = (1 << 64) - 1
        key0 = torch.tensor([42, 0], dtype=torch.uint64, device=device)
        key_shifted = torch.tensor([42, data], dtype=torch.uint64, device=device)
        self.assertEqual(random.fold_in(key0, data), random.fold_in(key_shifted, 0))

    @parametrize(
        "data",
        [
            subtest(-1, name="neg_one"),
            subtest(-12345, name="mid"),
            subtest(-(1 << 63), name="int64_min"),
        ],
    )
    def test_data_negative_reinterpreted_as_uint64(self, device, data):
        # [FUNCTION DOC] Parametrized over three negative integers. Checks that a negative datum
        # is treated as that value plus 2 to the 64, by comparing against folding the explicitly
        # shifted non-negative integer.
        # Negative data is remapped per the docstring: 2**64 + data.
        key = random.key(42, device=device)
        expected = random.fold_in(key, (1 << 64) + data)
        self.assertEqual(random.fold_in(key, data), expected)

    def test_error_invalid_data(self, device):
        # [FUNCTION DOC] Checks the error cases for the data argument: integers just outside the
        # range from minus 2 to the 63 to 2 to the 64 minus 1 raise ValueError; an int64 tensor
        # raises RuntimeError about dtype; a tensor with two values raises RuntimeError about
        # needing a single value.
        key = random.key(42, device=device)
        # int out of the [-2**63, 2**64 - 1] range
        for bad in (1 << 64, -(1 << 63) - 1):
            with self.assertRaisesRegex(ValueError, "data must be in"):
                random.fold_in(key, bad)
        # tensor with the wrong dtype
        with self.assertRaisesRegex(RuntimeError, "data must have dtype uint64"):
            random.fold_in(key, torch.tensor(7, dtype=torch.int64, device=device))
        # tensor with more than one value
        with self.assertRaisesRegex(RuntimeError, "data must be a single value"):
            random.fold_in(key, torch.tensor([1, 2], dtype=torch.uint64, device=device))

    @onlyAccelerator
    def test_error_data_wrong_device(self, device):
        # [FUNCTION DOC] Accelerator-only. Checks that a CPU data tensor combined with a key on
        # the accelerator raises a RuntimeError about tensors on different devices.
        key = random.key(42, device=device)
        # A CPU data tensor with an accelerator key is a device mismatch.
        with self.assertRaisesRegex(
            RuntimeError, "Expected all tensors to be on the same device"
        ):
            random.fold_in(key, torch.tensor(7, dtype=torch.uint64))  # CPU

    @parametrize(
        "data",
        [
            subtest(0, name="zero"),
            subtest(7, name="small"),
            subtest(1 << 63, name="int64_max_plus_one"),
            subtest((1 << 64) - 1, name="uint64_max"),
        ],
    )
    def test_tensor_data_matches_int(self, device, data):
        # [FUNCTION DOC] Parametrized over integer values including those above the int64 maximum.
        # With a batched key, checks that a zero-dimensional uint64 tensor and a one-element
        # uint64 tensor give the same result as passing the plain integer.
        key = random.split(random.key(42, device=device), 4)  # (4, 2) batched
        expected = random.fold_in(key, data)
        # Both a 0-dim scalar and a (1,) tensor are accepted as a single value.
        scalar = torch.tensor(data, dtype=torch.uint64, device=device)
        one_d = torch.tensor([data], dtype=torch.uint64, device=device)
        self.assertEqual(random.fold_in(key, scalar), expected)
        self.assertEqual(random.fold_in(key, one_d), expected)

    @onlyCUDA
    def test_tensor_data_cuda_graph(self, device):
        # [FUNCTION DOC] CUDA-only. Warms up on a side stream, captures a fold_in with a tensor
        # datum into a CUDA graph, then for two new values writes the value into the datum tensor
        # in place, replays the graph, and checks the captured output equals folding that integer,
        # showing the datum is read at replay time.
        # A tensor data is not baked into the graph: mutating it and replaying
        # produces the result for the new value.
        key = random.key(42, device=device)
        data = torch.zeros((), dtype=torch.uint64, device=device)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                random.fold_in(key, data)
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = random.fold_in(key, data)

        for value in (5, 99):
            data.fill_(value)
            g.replay()
            torch.cuda.synchronize()
            self.assertEqual(out, random.fold_in(key, value))

    @parametrize("batched", [False, True])
    @parametrize("tensor_data", [False, True])
    @onlyAccelerator
    def test_cross_device_consistency(self, device, batched, tensor_data):
        # [FUNCTION DOC] Accelerator-only, parametrized on batched keys and tensor versus integer
        # data. Folds the same datum seven into equivalent CPU and device keys and checks the
        # results match after moving the device result to CPU.
        key_cpu = random.key(42)
        key_dev = random.key(42, device=device)
        if batched:
            # Batched key exercises the multi-key path.
            key_cpu = random.split(key_cpu, 4)  # (4, 2)
            key_dev = random.split(key_dev, 4)
        if tensor_data:
            # Tensor data exercises the .Tensor overload.
            data_cpu = torch.tensor(7, dtype=torch.uint64)
            data_dev = torch.tensor(7, dtype=torch.uint64, device=device)
        else:
            data_cpu = data_dev = 7
        self.assertEqual(
            random.fold_in(key_cpu, data_cpu),
            random.fold_in(key_dev, data_dev).cpu(),
        )


class TestStatelessRNGDistribution(TestCase):
    # [CLASS DOC] Tests for the normal and uniform generators (functional and in-place forms)
    # across floating dtypes: shape and dtype, determinism, batched key semantics and
    # broadcasting, key validation errors, offset behavior, small and empty outputs, statistics,
    # and cross-device agreement. Most tests are parametrized on the generator name.
    def _gen(self, gen_fn_name, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Looks up the generator function with the given name in the random
        # module and calls it with the remaining positional and keyword arguments, returning its
        # result.
        return getattr(random, gen_fn_name)(*args, **kwargs)

    @parametrize("gen_fn_name", ["normal", "uniform"])
    @dtypes(*all_floating_dtypes)
    def test_basic_shape(self, device, dtype, gen_fn_name):
        # [FUNCTION DOC] Checks that generating one hundred values for either generator returns
        # the requested shape and dtype.
        key = random.key(42, device=device)
        result = self._gen(gen_fn_name, key, (100,), dtype=dtype)
        self.assertEqual(result.shape, (100,))
        self.assertEqual(result.dtype, dtype)

    @parametrize("gen_fn_name", ["normal", "uniform"])
    @dtypes(*all_floating_dtypes)
    def test_determinism(self, device, dtype, gen_fn_name):
        # [FUNCTION DOC] Checks that two generations with the same key give equal tensors.
        key = random.key(42, device=device)
        a = self._gen(gen_fn_name, key, (1000,), dtype=dtype)
        b = self._gen(gen_fn_name, key, (1000,), dtype=dtype)
        self.assertEqual(a, b)

    @parametrize("gen_fn_name", ["normal", "uniform"])
    @dtypes(*all_floating_dtypes)
    def test_different_keys(self, device, dtype, gen_fn_name):
        # [FUNCTION DOC] Checks that two different keys give different samples.
        key1 = random.key(42, device=device)
        key2 = random.key(43, device=device)
        a = self._gen(gen_fn_name, key1, (1000,), dtype=dtype)
        b = self._gen(gen_fn_name, key2, (1000,), dtype=dtype)
        self.assertNotEqual(a, b)

    @parametrize("gen_fn_name", ["normal", "uniform"])
    @dtypes(*all_floating_dtypes)
    def test_batched_keys(self, device, dtype, gen_fn_name):
        # [FUNCTION DOC] Uses four keys of shape (4, 1, 2) and output shape (4, 100). Checks each
        # output row equals generation from the matching single key.
        key = random.key(42, device=device)
        keys = random.split(key, 4).unsqueeze(-2)  # (4, 1, 2)
        result = self._gen(gen_fn_name, keys, (4, 100), dtype=dtype)
        for i in range(4):
            individual = self._gen(gen_fn_name, keys[i], (100,), dtype=dtype)
            self.assertEqual(result[i], individual)

    @parametrize("gen_fn_name", ["normal", "uniform"])
    @dtypes(*all_floating_dtypes)
    def test_batched_keys_large(self, device, dtype, gen_fn_name):
        # [FUNCTION DOC] Same as the batched keys test but with ten thousand elements per key, to
        # cover the large-size batched kernel path.
        # Large event_numel to exercise the multi-key tiled kernel path.
        key = random.key(42, device=device)
        keys = random.split(key, 4).unsqueeze(-2)  # (4, 1, 2)
        result = self._gen(gen_fn_name, keys, (4, 10000), dtype=dtype)
        for i in range(4):
            individual = self._gen(gen_fn_name, keys[i], (10000,), dtype=dtype)
            self.assertEqual(result[i], individual)

    @parametrize("gen_fn_name", ["normal", "uniform"])
    @dtypes(*all_floating_dtypes)
    def test_multi_batch(self, device, dtype, gen_fn_name):
        # [FUNCTION DOC] Uses key batch shape (2, 3, 1, 2) and output (2, 3, 50). Checks every (i,
        # j) slice equals generation from the single key at that position.
        key = random.key(42, device=device)
        keys = random.split(key, 6).view(2, 3, 1, 2)
        result = self._gen(gen_fn_name, keys, (2, 3, 50), dtype=dtype)
        for i in range(2):
            for j in range(3):
                individual = self._gen(gen_fn_name, keys[i][j], (50,), dtype=dtype)
                self.assertEqual(result[i][j], individual)

    @parametrize("gen_fn_name", ["normal", "uniform"])
    @dtypes(*all_floating_dtypes)
    def test_key_broadcasting_semantics(self, device, dtype, gen_fn_name):
        # [FUNCTION DOC] Checks key broadcasting rules: size-one key dims replicate the same
        # stream across that output dim while other dims use different keys; an all-size-one key
        # matches the unbatched key of the flattened output; several trailing size-one dims
        # broadcast; and a key batch covering the entire output gives one element per key.
        key = random.key(42, device=device)

        # Broadcast key dim: size-1 dims replicate, other dims index keys.
        keys = random.split(key, 3).unsqueeze(0).unsqueeze(-2)  # (1, 3, 1, 2)
        result = self._gen(gen_fn_name, keys, (4, 3, 100), dtype=dtype)
        for i in range(1, 4):
            self.assertEqual(result[0], result[i])
        for j in range(1, 3):
            self.assertNotEqual(result[0][0], result[0][j])

        # All-broadcast key matches unbatched.
        batched = self._gen(gen_fn_name, key.view(1, 1, 2), (4, 100), dtype=dtype)
        unbatched = self._gen(gen_fn_name, key, (400,), dtype=dtype)
        self.assertEqual(batched.flatten(), unbatched)

        # Multiple trailing size-1 dims to broadcast over.
        keys = random.split(key, 4).view(4, 1, 1, 2)
        result = self._gen(gen_fn_name, keys, (4, 10, 100), dtype=dtype)
        for i in range(4):
            individual = self._gen(gen_fn_name, keys[i], (10, 100), dtype=dtype)
            self.assertEqual(result[i], individual)
        keys_flat = random.split(key, 4).unsqueeze(-2)  # (4, 1, 2)
        flat = self._gen(gen_fn_name, keys_flat, (4, 1000), dtype=dtype)
        self.assertEqual(result.view(4, 1000), flat)

        # No generation dims: every element gets its own key.
        keys = random.split(key, 12).view(4, 3, 2)
        result = self._gen(gen_fn_name, keys, (4, 3), dtype=dtype)
        for i in range(4):
            for j in range(3):
                individual = self._gen(gen_fn_name, keys[i][j], (1,), dtype=dtype)
                self.assertEqual(result[i][j], individual.squeeze())

    @parametrize("gen_fn_name", ["normal", "uniform"])
    def test_error_wrong_key_dtype(self, device, gen_fn_name):
        # [FUNCTION DOC] Checks that a float32 key makes either generator raise a RuntimeError
        # about the uint64 dtype.
        key = torch.tensor([42, 0], dtype=torch.float32, device=device)
        with self.assertRaisesRegex(RuntimeError, "key must have dtype uint64"):
            self._gen(gen_fn_name, key, (100,))

    @parametrize("gen_fn_name", ["normal", "uniform"])
    def test_error_key_shape(self, device, gen_fn_name):
        # [FUNCTION DOC] Checks key validation errors: a three-element key; a batched key whose
        # ndim is too small or too large relative to the output ndim; and a key batch that cannot
        # be broadcast against the output shape. Each raises a RuntimeError with the matching
        # message.
        key = random.key(42, device=device)
        # Last dim must be 2.
        bad_key = torch.tensor([42, 0, 1], dtype=torch.uint64, device=device)
        with self.assertRaisesRegex(
            RuntimeError, r"key must have shape \(2,\) or \(\*batch, 2\)"
        ):
            self._gen(gen_fn_name, bad_key, (100,))
        # Key batch ndim must equal output ndim (too few).
        with self.assertRaisesRegex(
            RuntimeError, "batched key must have ndim == output ndim \\+ 1"
        ):
            self._gen(gen_fn_name, random.split(key, 3), (3, 4, 100))
        # Key batch ndim must equal output ndim (too many).
        with self.assertRaisesRegex(
            RuntimeError, "batched key must have ndim == output ndim \\+ 1"
        ):
            self._gen(gen_fn_name, random.split(key, 3).view(3, 1, 1, 2), (3, 100))
        # Key batch dims must be broadcastable with output.
        with self.assertRaisesRegex(
            RuntimeError, "is not broadcastable with output shape"
        ):
            self._gen(gen_fn_name, random.split(key, 5).unsqueeze(-2), (3, 100))

    @parametrize("gen_fn_name", ["normal", "uniform"])
    @dtypes(*all_floating_dtypes)
    def test_offset_shift_consistency(self, device, dtype, gen_fn_name):
        # [FUNCTION DOC] Checks that increasing a key offset by k shifts the output stream by k
        # times the number of values per Philox call (two for float64, four otherwise), by
        # comparing slices of the offset-zero output and the shifted output.
        seed = 42
        n = 100
        key0 = random.key(seed, device=device)
        ref = self._gen(gen_fn_name, key0, (n,), dtype=dtype)

        # as a key's offset shifts, we expect the stream to shift by
        # the number of elements per philox call (2 for double; 4 otherwise)
        for offset in range(1, 4):
            key = torch.tensor([seed, offset], dtype=torch.uint64, device=device)
            elems_per_call = 2 if dtype == torch.float64 else 4
            expected_shift = offset * elems_per_call
            result = self._gen(gen_fn_name, key, (n,), dtype=dtype)
            self.assertEqual(ref[expected_shift:], result[:-expected_shift])

    @parametrize("gen_fn_name", ["normal", "uniform"])
    @dtypes(*all_floating_dtypes)
    def test_offset_overflow(self, device, dtype, gen_fn_name):
        # [FUNCTION DOC] Uses a key with the maximum uint64 offset and checks the offset wraps to
        # zero: the output after the first Philox call's worth of values matches the offset-zero
        # output shifted by that many elements (two for float64, four otherwise).
        seed = 42
        n = 100
        last_offset_before_wrap = (1 << 64) - 1
        key = torch.tensor(
            [seed, last_offset_before_wrap], dtype=torch.uint64, device=device
        )
        result = self._gen(gen_fn_name, key, (n,), dtype=dtype)

        # ensure offset wraps around to 0 by comparing with 0-offset key results
        key0 = random.key(seed, device=device)
        result0 = self._gen(gen_fn_name, key0, (n,), dtype=dtype)
        elems_per_call = 2 if dtype == torch.float64 else 4
        self.assertEqual(result[elems_per_call:], result0[:-elems_per_call])

    @parametrize("gen_fn_name", ["normal", "uniform"])
    @dtypes(*all_floating_dtypes)
    def test_small_output_sizes(self, device, dtype, gen_fn_name):
        # [FUNCTION DOC] For sizes zero through seven (some not multiples of the Philox width),
        # checks shape, determinism, and that the result equals the prefix of a larger one hundred
        # element generation.
        key = random.key(42, device=device)
        large = self._gen(gen_fn_name, key, (100,), dtype=dtype)
        for n in [0, 1, 2, 3, 4, 5, 7]:
            result = self._gen(gen_fn_name, key, (n,), dtype=dtype)
            self.assertEqual(result.shape, (n,))
            # Determinism.
            result2 = self._gen(gen_fn_name, key, (n,), dtype=dtype)
            self.assertEqual(result, result2)
            # Prefix consistency: first n elements of a larger output.
            if n > 0:
                self.assertEqual(result, large[:n])

    @parametrize("gen_fn_name", ["normal", "uniform"])
    @parametrize("layout", ["contiguous", "noncontiguous", "unaligned"])
    @dtypes(*all_floating_dtypes)
    def test_inplace(self, device, dtype, gen_fn_name, layout):
        # [FUNCTION DOC] Parametrized on generator and memory layout (contiguous, strided
        # non-contiguous, and contiguous with unaligned data pointer). Calls the in-place variant
        # on a pre-allocated tensor, checks it returns that same tensor object and that the
        # contents equal the functional result of the same size.
        key = random.key(42, device=device)
        if layout == "contiguous":
            result = torch.empty(1000, dtype=dtype, device=device)
        elif layout == "noncontiguous":
            result = torch.empty(2000, dtype=dtype, device=device)[::2]
        else:
            # Contiguous but data pointer is not aligned to vectorized write width.
            result = torch.empty(1001, dtype=dtype, device=device)[1:]
        inplace_fn = getattr(random, gen_fn_name + "_")
        out = inplace_fn(key, result)
        self.assertIs(out, result)
        functional = self._gen(gen_fn_name, key, (1000,), dtype=dtype)
        self.assertEqual(result, functional)

    @parametrize("gen_fn_name", ["normal", "uniform"])
    @dtypes(*all_floating_dtypes)
    def test_empty_output(self, device, dtype, gen_fn_name):
        # [FUNCTION DOC] Checks that outputs with zero elements, of shapes (0,) and (3, 0), are
        # produced with the right shape and dtype and without error.
        key = random.key(42, device=device)
        result = self._gen(gen_fn_name, key, (0,), dtype=dtype)
        self.assertEqual(result.shape, (0,))
        self.assertEqual(result.dtype, dtype)
        result = self._gen(gen_fn_name, key, (3, 0), dtype=dtype)
        self.assertEqual(result.shape, (3, 0))
        self.assertEqual(result.dtype, dtype)

    # Distribution-specific tests

    @dtypes(*all_floating_dtypes)
    def test_standard_normal_statistics(self, device, dtype):
        # [FUNCTION DOC] Checks one hundred thousand standard normal samples have mean within 0.05
        # of zero and standard deviation within 0.05 of one.
        key = random.key(42, device=device)
        result = random.normal(key, (100000,), dtype=dtype)
        self.assertTrue(abs(result.mean().item()) < 0.05)
        self.assertTrue(abs(result.std().item() - 1.0) < 0.05)

    @dtypes(*all_floating_dtypes)
    def test_custom_mean_std(self, device, dtype):
        # [FUNCTION DOC] Checks normal samples with mean five and standard deviation two have
        # sample mean and standard deviation within 0.1 of those values.
        key = random.key(42, device=device)
        result = random.normal(key, (100000,), mean=5.0, std=2.0, dtype=dtype)
        self.assertTrue(abs(result.mean().item() - 5.0) < 0.1)
        self.assertTrue(abs(result.std().item() - 2.0) < 0.1)

    @dtypes(*all_floating_dtypes)
    def test_standard_uniform_statistics(self, device, dtype):
        # [FUNCTION DOC] Checks standard uniform samples have mean near one half, minimum at least
        # zero and maximum strictly below one.
        key = random.key(42, device=device)
        result = random.uniform(key, (100000,), dtype=dtype)
        self.assertTrue(abs(result.mean().item() - 0.5) < 0.05)
        self.assertTrue(result.min().item() >= 0.0)
        self.assertTrue(result.max().item() < 1.0)

    @dtypes(*all_floating_dtypes)
    def test_custom_low_high(self, device, dtype):
        # [FUNCTION DOC] Checks uniform samples with bounds two and five have mean near 3.5 within
        # 0.1 and all values within the closed bounds.
        key = random.key(42, device=device)
        result = random.uniform(key, (100000,), low=2.0, high=5.0, dtype=dtype)
        self.assertTrue(abs(result.mean().item() - 3.5) < 0.1)
        self.assertTrue(result.min().item() >= 2.0)
        self.assertTrue(result.max().item() <= 5.0)

    @dtypes(*all_floating_dtypes)
    @parametrize("batched", [False, True])
    @onlyAccelerator
    def test_cross_device_uniform_consistency(self, device, dtype, batched):
        # [FUNCTION DOC] Accelerator-only, parametrized on batched keys. Generates uniform values
        # on CPU and on the device from equal keys and requires exact (zero tolerance) equality,
        # since uniform generation uses no transcendental functions.
        if batched:
            # Batched key exercises the multi-key path.
            key_cpu = random.split(random.key(42), 4).unsqueeze(-2)  # (4, 1, 2)
            key_dev = random.split(random.key(42, device=device), 4).unsqueeze(-2)
            shape = (4, 100)
        else:
            key_cpu = random.fold_in(random.key(42), 7)
            key_dev = random.fold_in(random.key(42, device=device), 7)
            shape = (1000,)
        # Uniform generation uses no transcendentals, so results must be bitwise identical.
        self.assertEqual(
            self._gen("uniform", key_cpu, shape, dtype=dtype),
            self._gen("uniform", key_dev, shape, dtype=dtype).cpu(),
            atol=0,
            rtol=0,
        )

    @dtypes(*all_floating_dtypes)
    @onlyAccelerator
    def test_cross_device_normal_consistency(self, device, dtype):
        # [FUNCTION DOC] Accelerator-only. Generates one thousand normal values from equal CPU and
        # device keys and compares with default tolerance, because the Box-Muller transform uses
        # math functions that differ slightly between CPU and CUDA.
        key_cpu = random.fold_in(random.key(42), 7)
        key_dev = random.fold_in(random.key(42, device=device), 7)
        # Normal generation uses Box-Muller (log, sin, cos), and CUDA uses fast-math
        # intrinsics (__logf, __sincosf) that differ slightly from CPU std::log / std::sin /
        # std::cos. Results are approximately but not bitwise equal. assertEqual() by default
        # allows for some tolerance in the comparisons.
        self.assertEqual(
            self._gen("normal", key_cpu, (1000,), dtype=dtype),
            self._gen("normal", key_dev, (1000,), dtype=dtype).cpu(),
        )


class TestStatelessRNGCompile(TestCase):
    # [CLASS DOC] Tests that the stateless RNG functions work inside torch.compile with the
    # aot_eager backend in full-graph mode (and a couple of inductor tests), matching eager
    # results.
    def test_split_fullgraph(self, device):
        # [FUNCTION DOC] Compiles a function calling split with four children as a full graph and
        # checks it equals the eager split.
        key = random.key(42, device=device)

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(key):
            # [FUNCTION DOC] HELPER: Compiled function returning the four-way split of the key.
            return random.split(key, 4)

        self.assertEqual(f(key), random.split(key, 4))

    def test_fold_in_fullgraph(self, device):
        # [FUNCTION DOC] Compiles a function folding in a constant datum and checks the result
        # equals eager.
        key = random.key(42, device=device)

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(key):
            # [FUNCTION DOC] HELPER: Compiled function returning the key folded with data seven.
            return random.fold_in(key, 7)

        self.assertEqual(f(key), random.fold_in(key, 7))

    def test_fold_in_tensor_fullgraph(self, device):
        # [FUNCTION DOC] Compiles fold_in where the datum is a tensor graph input, and checks the
        # result equals eager.
        key = random.key(42, device=device)
        # data as a graph input (not a constant) exercises the Tensor overload.
        data = torch.tensor(7, dtype=torch.uint64, device=device)

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(key, data):
            # [FUNCTION DOC] HELPER: Compiled function returning the key folded with the given
            # tensor datum.
            return random.fold_in(key, data)

        self.assertEqual(f(key, data), random.fold_in(key, data))

    def test_uniform_fullgraph(self, device):
        # [FUNCTION DOC] Compiles uniform generation of one hundred values and checks it equals
        # eager.
        key = random.key(42, device=device)

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(key):
            # [FUNCTION DOC] HELPER: Compiled function returning uniform samples of shape (100,)
            # for the key.
            return random.uniform(key, (100,))

        self.assertEqual(f(key), random.uniform(key, (100,)))

    def test_normal_fullgraph(self, device):
        # [FUNCTION DOC] Compiles normal generation of one hundred values and checks it equals
        # eager.
        key = random.key(42, device=device)

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(key):
            # [FUNCTION DOC] HELPER: Compiled function returning normal samples of shape (100,)
            # for the key.
            return random.normal(key, (100,))

        self.assertEqual(f(key), random.normal(key, (100,)))

    def test_batched_normal_fullgraph(self, device):
        # [FUNCTION DOC] Compiles normal generation with four batched keys and output (4, 50) and
        # checks it equals eager.
        key = random.key(42, device=device)
        keys = random.split(key, 4).unsqueeze(-2)  # (4, 1, 2)

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(keys):
            # [FUNCTION DOC] HELPER: Compiled function returning batched normal samples of shape
            # (4, 50) for the given keys.
            return random.normal(keys, (4, 50))

        self.assertEqual(f(keys), random.normal(keys, (4, 50)))

    def test_split_then_normal_fullgraph(self, device):
        # [FUNCTION DOC] Compiles a function that splits a key and generates normal samples from
        # the batched children, and checks it equals the eager composition.
        key = random.key(42, device=device)

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(key):
            # [FUNCTION DOC] HELPER: Compiled function splitting the key four ways, adding a
            # broadcast dim, and returning normal samples of shape (4, 100).
            keys = random.split(key, 4).unsqueeze(-2)
            return random.normal(keys, (4, 100))

        self.assertEqual(
            f(key), random.normal(random.split(key, 4).unsqueeze(-2), (4, 100))
        )

    def test_fold_in_then_uniform_fullgraph(self, device):
        # [FUNCTION DOC] Compiles a function that folds data three into a key and generates
        # uniform values from the result, and checks it equals eager.
        key = random.key(42, device=device)

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(key):
            # [FUNCTION DOC] HELPER: Compiled function folding data three into the key and
            # returning uniform samples of shape (100,).
            k = random.fold_in(key, 3)
            return random.uniform(k, (100,))

        self.assertEqual(f(key), random.uniform(random.fold_in(key, 3), (100,)))

    @onlyAccelerator
    @parametrize("op", ["uniform", "normal"])
    def test_generation_no_extra_clone(self, device, op):
        # [FUNCTION DOC] Accelerator-only, parametrized on generator. Skips on CUDA without
        # Triton. Compiles generation of a 2048 by 2048 float32 tensor with the default backend,
        # then measures peak accelerator memory above baseline during a run and requires it be
        # below 1.5 times the output size, so no extra full-size copy is allocated. Also checks
        # the result equals eager.
        # Out-of-place uniform()/normal() fully overwrite their output; ensure
        # generation in torch.compile doesn't allocate an extra full-size buffer
        # (i.e. ensure peak ~= output size).
        if torch.device(device).type == "cuda" and not HAS_TRITON:
            self.skipTest("CUDA inductor codegen requires triton")
        gen = getattr(random, op)
        key = random.key(42, device=device)
        shape = (2048, 2048)  # 16 MiB fp32; an extra clone would ~double peak
        out_bytes = shape[0] * shape[1] * torch.float32.itemsize

        @torch.compile(fullgraph=True)
        def f(key):
            # [FUNCTION DOC] HELPER: Compiled function calling the selected generator on the key
            # with the closed-over shape.
            return gen(key, shape)

        f(key)  # compile + warm up the allocator
        torch.accelerator.synchronize(device)
        base = torch.accelerator.memory_allocated(device)
        torch.accelerator.reset_peak_memory_stats(device)
        result = f(key)
        torch.accelerator.synchronize(device)
        extra = torch.accelerator.max_memory_allocated(device) - base

        self.assertEqual(result, gen(key, shape))
        self.assertLess(extra, 1.5 * out_bytes)  # no extra full-size clone

    @onlyAccelerator
    def test_generation_no_corruption_from_buffer_reuse(self, device):
        # [FUNCTION DOC] Accelerator-only regression test. Skips on CUDA without Triton. Compiles
        # a function that keeps four uniform generations alive and sums them, and checks equality
        # with eager to catch buffer reuse corrupting in-place generation.
        # Regression test for Inductor buffer reuse corrupting generation.
        if torch.device(device).type == "cuda" and not HAS_TRITON:
            self.skipTest("CUDA inductor codegen requires triton")

        # Keep all generations live until the final sum, exercising whether
        # Inductor reuses an in-place generation's buffer while it is still live.
        def f(key, x):
            # [FUNCTION DOC] HELPER: Generates four uniform tensors shaped like the input from
            # keys folded with indices zero through three, adds them all to the input one by one,
            # and returns the sum.
            rs = [random.uniform(random.fold_in(key, i), x.shape) for i in range(4)]
            for r in rs:
                x = x + r
            return x

        key = random.key(0, device=device)
        x = torch.randn(16, device=device)
        self.assertEqual(torch.compile(f, dynamic=False)(key, x), f(key, x))


instantiate_device_type_tests(TestStatelessRNGKey, globals(), only_for=("cpu", "cuda"))
instantiate_device_type_tests(
    TestStatelessRNGKeySplit, globals(), only_for=("cpu", "cuda")
)
instantiate_device_type_tests(
    TestStatelessRNGKeyFoldIn, globals(), only_for=("cpu", "cuda")
)
instantiate_device_type_tests(
    TestStatelessRNGDistribution, globals(), only_for=("cpu", "cuda")
)
instantiate_device_type_tests(
    TestStatelessRNGCompile, globals(), only_for=("cpu", "cuda")
)


if __name__ == "__main__":
    run_tests()
