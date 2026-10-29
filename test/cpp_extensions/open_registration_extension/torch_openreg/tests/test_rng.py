# Owner(s): ["module: PrivateUse1"]
# [MODULE DOC] RNG tests for the openreg backend: generators, global seeds and RNG
# state. Tests needing true per-device RNG are skipped. No import-time side effects.

import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestRNG(TestCase):
    # [CLASS DOC] Test cases for generators and seeding on openreg.
    def test_generator(self):
        # [FUNCTION DOC] A generator on openreg:1 has type openreg and index one.
        """Test generator creation on openreg device"""
        generator = torch.Generator(device="openreg:1")
        self.assertEqual(generator.device.type, "openreg")
        self.assertEqual(generator.device.index, 1)

    def test_rng_state(self):
        # [FUNCTION DOC] Gets the RNG state of device zero and sets it back; passes if no error.
        """Test RNG state get and set"""
        state = torch.openreg.get_rng_state(0)
        torch.openreg.set_rng_state(state, 0)

    def test_manual_seed(self):
        # [FUNCTION DOC] manual_seed_all with 2024 makes initial_seed return 2024.
        """Test manual seed setting"""
        torch.openreg.manual_seed_all(2024)
        self.assertEqual(torch.openreg.initial_seed(), 2024)

    def test_generator_seed(self):
        # [FUNCTION DOC] Manually seeding generators on devices
        # zero and one is read back by initial_seed.
        """Test generator seed setting"""
        generator = torch.Generator(device="openreg:0")
        generator.manual_seed(42)
        self.assertEqual(generator.initial_seed(), 42)

        generator = torch.Generator(device="openreg:1")
        generator.manual_seed(100)
        self.assertEqual(generator.initial_seed(), 100)

    @unittest.skip("openreg backend does not implement per-device RNG yet")
    def test_generator_state(self):
        # [FUNCTION DOC] Skipped (no per-device RNG): restoring a saved generator state should
        # replay the same random numbers.
        """Test generator state get/set"""
        generator = torch.Generator(device="openreg:0")
        state = generator.get_state()

        # Generate some random numbers
        x1 = torch.randn(10, device="openreg:0", generator=generator)

        # Set state back
        generator.set_state(state)
        x2 = torch.randn(10, device="openreg:0", generator=generator)

        # Should produce same sequence
        self.assertEqual(x1, x2)

    @unittest.skip("openreg backend does not implement per-device RNG yet")
    def test_rng_state_consistency(self):
        # [FUNCTION DOC] Skipped (no per-device RNG): states of devices zero and one should
        # differ, and a state set on device zero should be read back.
        """Test RNG state consistency across devices"""
        state0 = torch.openreg.get_rng_state(0)
        state1 = torch.openreg.get_rng_state(1)

        # States should be different for different devices
        self.assertNotEqual(state0, state1)

        # Setting state should work
        torch.openreg.set_rng_state(state0, 0)
        restored_state = torch.openreg.get_rng_state(0)
        self.assertEqual(state0, restored_state)

    def test_manual_seed_all(self):
        # [FUNCTION DOC] manual_seed_all with 1234 then
        # 5678 is reflected by initial_seed each time.
        """Test manual_seed_all sets seed for all devices"""
        torch.openreg.manual_seed_all(1234)

        # Check that seed is set
        seed = torch.openreg.initial_seed()
        self.assertEqual(seed, 1234)

        # Test with different seed
        torch.openreg.manual_seed_all(5678)
        seed = torch.openreg.initial_seed()
        self.assertEqual(seed, 5678)

    @unittest.skip("openreg backend does not implement per-device RNG yet")
    def test_generator_different_devices(self):
        # [FUNCTION DOC] Skipped (no per-device RNG): equally seeded generators on two devices
        # should give the same random numbers.
        """Test generators on different devices"""
        gen0 = torch.Generator(device="openreg:0")
        gen1 = torch.Generator(device="openreg:1")

        gen0.manual_seed(1)
        gen1.manual_seed(1)

        x0 = torch.randn(10, device="openreg:0", generator=gen0)
        x1 = torch.randn(10, device="openreg:1", generator=gen1)

        # Should produce same sequence with same seed
        self.assertEqual(x0.cpu(), x1.cpu())


if __name__ == "__main__":
    run_tests()
