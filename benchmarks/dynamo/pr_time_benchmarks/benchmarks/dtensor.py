# [MODULE DOC] PR-time benchmarks measuring raw instruction count of DTensor dispatch for
# several operators (detach, to/from local, collectives, add with backward, in-place add, view,
# random, custom handler). Uses a fake process group with world size 256 so no real
# communication happens. Run as a script with one argument, the CSV result path. main
# initializes and destroys the global default process group.
import sys

from benchmark_base import BenchmarkBase

import torch
from torch.distributed._tensor import DTensor, Partial, Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore


class BenchmarkDTensorDispatch(BenchmarkBase):
    # [CLASS DOC] Base class for the DTensor dispatch benchmarks. Category is a dtensor_dispatch
    # prefix plus the operator name. Subclasses supply the measured work; this class builds a
    # device mesh over the world size and two replicated 10 by 10 DTensors (a and b). State:
    # world size, mesh, a, b.
    def __init__(self, operator, world_size, device="cuda") -> None:
        # [FUNCTION DOC] Calls the base constructor with the operator-derived category and the
        # device, and stores the world size.
        super().__init__(
            category=f"dtensor_dispatch_{operator}",
            device=device,
        )
        self.world_size = world_size

    def name(self) -> str:
        # [FUNCTION DOC] Returns the result name, equal to the category.
        prefix = f"{self.category()}"
        return prefix

    def description(self) -> str:
        # [FUNCTION DOC] Returns a description string naming the category and device.
        return f"DTensor dispatch instructions for {self.category()} on {self.device()}"

    def _prepare_once(self) -> None:
        # [FUNCTION DOC] One-time setup: creates a one-dimensional device mesh named dp over the
        # world size on the device, and two DTensors a and b built from local 10 by 10 ones
        # tensors with replicated placement.
        self.mesh = torch.distributed.device_mesh.init_device_mesh(
            self.device(), (self.world_size,), mesh_dim_names=("dp",)
        )
        self.a = DTensor.from_local(
            torch.ones(10, 10, device=self.device()), self.mesh, [Replicate()]
        )
        self.b = DTensor.from_local(
            torch.ones(10, 10, device=self.device()), self.mesh, [Replicate()]
        )

    def _prepare(self) -> None:
        # [FUNCTION DOC] Per-iteration setup: does nothing.
        pass


class BenchmarkDetach(BenchmarkDTensorDispatch):
    # [CLASS DOC] Benchmark of DTensor detach on the meta device.
    def __init__(self, world_size) -> None:
        # [FUNCTION DOC] Configures operator name detach with the meta device.
        super().__init__(operator="detach", world_size=world_size, device="meta")

    def _work(self) -> None:
        # [FUNCTION DOC] Measured work: detaches tensor a.
        self.a.detach()


class BenchmarkToFromLocal(BenchmarkDTensorDispatch):
    # [CLASS DOC] Benchmark of converting a DTensor to its local tensor and back, on the
    # meta device.
    def __init__(self, world_size) -> None:
        # [FUNCTION DOC] Configures operator name to_from_local with the meta device.
        super().__init__(operator="to_from_local", world_size=world_size, device="meta")

    def _work(self) -> None:
        # [FUNCTION DOC] Measured work: takes the local tensor of a and rebuilds a DTensor from
        # it on the mesh with replicated placement.
        local = self.a.to_local()
        DTensor.from_local(local, self.mesh, [Replicate()])


class BenchmarkCollectives(BenchmarkDTensorDispatch):
    # [CLASS DOC] Benchmark of redistribute collectives on the default cuda device. Adds a third
    # DTensor c with partial placement.
    def __init__(self, world_size) -> None:
        # [FUNCTION DOC] Configures operator name collectives with the default device.
        super().__init__(operator="collectives", world_size=world_size)

    def _prepare_once(self) -> None:
        # [FUNCTION DOC] One-time setup: runs the base setup and additionally creates DTensor c
        # from a 10 by 10 ones local tensor with partial placement.
        super()._prepare_once()
        self.c = DTensor.from_local(
            torch.ones(10, 10, device=self.device()), self.mesh, [Partial()]
        )

    def _work(self) -> None:
        # [FUNCTION DOC] Measured work: redistributes a to shard on dim 0, then dim 1
        # (all-to-all), then replicated (all-gather); then redistributes c to replicated
        # (all-reduce) and to shard on dim 0 (reduce-scatter). Results are discarded.
        # shard
        a = self.a.redistribute(placements=[Shard(0)])
        # alltoall
        a = a.redistribute(placements=[Shard(1)])
        # allgather
        a = a.redistribute(placements=[Replicate()])
        # allreduce
        self.c.redistribute(placements=[Replicate()])
        # reducescatter
        self.c.redistribute(placements=[Shard(0)])


class BenchmarkAddBackward(BenchmarkDTensorDispatch):
    # [CLASS DOC] Benchmark of an add followed by a backward pass on large (512 by 512)
    # replicated DTensors that require grad.
    def __init__(self, world_size) -> None:
        # [FUNCTION DOC] Configures operator name add_backward with the default device.
        super().__init__(operator="add_backward", world_size=world_size)

    def _prepare_once(self) -> None:
        # [FUNCTION DOC] One-time setup: runs the base setup, then replaces a and b with 512 by
        # 512 ones DTensors that require grad, replicated on the mesh.
        super()._prepare_once()
        self.a = DTensor.from_local(
            torch.ones(512, 512, device=self.device(), requires_grad=True),
            self.mesh,
            [Replicate()],
        )
        self.b = DTensor.from_local(
            torch.ones(512, 512, device=self.device(), requires_grad=True),
            self.mesh,
            [Replicate()],
        )

    def _work(self) -> None:
        # [FUNCTION DOC] Measured work: adds a and b, sums the result and runs backward.
        out = self.a + self.b
        out.sum().backward()


class BenchmarkInplace(BenchmarkDTensorDispatch):
    # [CLASS DOC] Benchmark of an in-place add on DTensors.
    def __init__(self, world_size) -> None:
        # [FUNCTION DOC] Configures operator name inplace with the default device.
        super().__init__(operator="inplace", world_size=world_size)

    def _work(self) -> None:
        # [FUNCTION DOC] Measured work: adds b into a in place.
        self.a.add_(self.b)


class BenchmarkView(BenchmarkDTensorDispatch):
    # [CLASS DOC] Benchmark of viewing a DTensor, on the meta device.
    def __init__(self, world_size) -> None:
        # [FUNCTION DOC] Configures operator name view with the meta device.
        super().__init__(operator="view", world_size=world_size, device="meta")

    def _work(self) -> None:
        # [FUNCTION DOC] Measured work: views a as a flat tensor of 100 elements.
        self.a.view(100)


class BenchmarkRandom(BenchmarkDTensorDispatch):
    # [CLASS DOC] Benchmark of an in-place random fill on a DTensor.
    def __init__(self, world_size) -> None:
        # [FUNCTION DOC] Configures operator name random with the default device.
        super().__init__(operator="random", world_size=world_size)

    def _work(self) -> None:
        # [FUNCTION DOC] Measured work: fills a in place using uniform random values.
        self.a.uniform_()


class BenchmarkCustomHandler(BenchmarkDTensorDispatch):
    # [CLASS DOC] Benchmark of a DTensor op handled by a custom handler (same-size check), on
    # the meta device.
    def __init__(self, world_size) -> None:
        # [FUNCTION DOC] Configures operator name custom_handler with the meta device.
        super().__init__(
            operator="custom_handler", world_size=world_size, device="meta"
        )

    def _work(self) -> None:
        # [FUNCTION DOC] Measured work: calls the is_same_size aten op on a and b.
        torch.ops.aten.is_same_size(self.a, self.b)


def main():
    # [FUNCTION DOC] Script entry point. Initializes the default process group with the fake
    # backend (world size 256, rank 0, fake store), then runs each of the eight benchmarks
    # with plain instruction counting and appends results to the CSV path given as first
    # argument. Destroys the process group at the end (not in a finally block, so an error
    # leaves it initialized).
    world_size = 256
    fake_store = FakeStore()
    torch.distributed.init_process_group(
        "fake", store=fake_store, rank=0, world_size=world_size
    )
    result_path = sys.argv[1]
    BenchmarkDetach(world_size).enable_instruction_count().collect_all().append_results(
        result_path
    )
    BenchmarkToFromLocal(
        world_size
    ).enable_instruction_count().collect_all().append_results(result_path)
    BenchmarkCollectives(
        world_size
    ).enable_instruction_count().collect_all().append_results(result_path)
    BenchmarkAddBackward(
        world_size
    ).enable_instruction_count().collect_all().append_results(result_path)
    BenchmarkInplace(
        world_size
    ).enable_instruction_count().collect_all().append_results(result_path)
    BenchmarkView(world_size).enable_instruction_count().collect_all().append_results(
        result_path
    )
    BenchmarkRandom(world_size).enable_instruction_count().collect_all().append_results(
        result_path
    )
    BenchmarkCustomHandler(
        world_size
    ).enable_instruction_count().collect_all().append_results(result_path)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
