# mypy: allow-untyped-defs

# [MODULE DOC] Test suite and helpers for combining DistributedDataParallel with distributed
# autograd over RPC. A hybrid model with remote embedding and remote linear parts (living on one
# remote worker) plus local linear layers is trained by several trainer workers coordinated by a
# master, in modes with no DDP, DDP around the whole model, or DDP around one inner layer,
# optionally with uneven inputs. Also compares distributed-autograd gradients against local
# autograd gradients, including CUDA and RemoteModule variants. Import-time side effects: creates
# a module logger with a console handler (debug level if the debug environment variable is set)
# and a module-level threading condition used as a shutdown signal. Constants define world layout:
# four trainers, one remote worker, one master.

import contextlib
import enum
import logging
import os
import threading
from typing import NamedTuple

import torch
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
import torch.nn as nn
from torch.distributed import rpc
from torch.distributed.nn import RemoteModule
from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.common_distributed import (
    requires_gloo,
    requires_nccl,
    skip_if_lt_x_gpu,
    skip_if_rocm_multiprocess,
)
from torch.testing._internal.dist_utils import dist_init, INIT_METHOD_TEMPLATE
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
    RpcAgentTestFixture,
)


NUM_EM_ROW = 2
D_SPARSE = 3
D_DENSE = 2
D_HID = 3
D_OUT = 1
NUM_TRAINERS = 4
# Trainers + the master + the remote worker
WORLD_SIZE = NUM_TRAINERS + 2
TRAINER_RANKS = list(range(NUM_TRAINERS))
REMOTE_WORKER_RANK = TRAINER_RANKS[-1] + 1
MASTER_RANK = REMOTE_WORKER_RANK + 1


class DdpMode(enum.Enum):
    # [CLASS DOC] Enumeration of how DDP is applied to the hybrid model: NONE (no DDP), OUTSIDE
    # (wrap the whole top-level module), INSIDE (wrap only an inner layer).
    # Don't apply DDP
    NONE = enum.auto()
    # Apply DDP to the top level nn.Module
    OUTSIDE = enum.auto()
    # Embed DDP inside the top level nn.Module
    INSIDE = enum.auto()


def init_logger():
    # [FUNCTION DOC] HELPER: Builds the module logger with a console handler and a formatter
    # including time, file, line, level, process and thread. Level is debug if the debug
    # environment variable is present, else info. Disables propagation to parent loggers. Returns
    # the logger.
    logger = logging.getLogger(__name__)
    level = logging.DEBUG if "debug" in os.environ else logging.INFO
    logger.setLevel(level)
    console = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(filename)s:%(lineno)s %(levelname)s p:%(processName)s t:%(threadName)s: %(message)s"
    )
    console.setFormatter(formatter)
    console.setLevel(level)
    # add the handlers to the logger
    logger.addHandler(console)
    logger.propagate = False
    return logger


gLogger = init_logger()


class FeatureSet(NamedTuple):
    # [CLASS DOC] Named tuple holding one batch: dense features, sparse (long index) features and
    # per-example values (the sign weights used to form the loss).
    """A feature set has 2 types of features"""

    dense_features: torch.Tensor
    sparse_features: torch.LongTensor
    values: torch.Tensor


def _call_method(method, rref, *args, **kwargs):
    # [FUNCTION DOC] HELPER: RPC target that applies the given method to the local value of an
    # RRef (which must be owned by the current worker) with extra arguments, and returns the
    # result.
    return method(rref.local_value(), *args, **kwargs)


def _remote_method(method, rref, *args, **kwargs):
    # [FUNCTION DOC] HELPER: Synchronously runs a method on the worker owning the RRef through a
    # blocking RPC and returns its result.
    args_tup = tuple([method, rref] + list(args))
    return rpc.rpc_sync(rref.owner(), _call_method, args=args_tup, kwargs=kwargs)


def _remote_method_async(method, rref, *args, **kwargs):
    # [FUNCTION DOC] HELPER: Same as _remote_method but asynchronous: returns a future for the RPC
    # result.
    args_tup = tuple([method, rref] + list(args))
    return rpc.rpc_async(rref.owner(), _call_method, args=args_tup, kwargs=kwargs)


class RemoteEM(nn.Module):
    # [CLASS DOC] Remote-side embedding model: wraps an EmbeddingBag initialized to constant 0.5
    # weights. Constructed on the remote worker via an RRef.
    def __init__(self, num_embeddings: int, embedding_dim: int):
        # [FUNCTION DOC] Logs and creates the embedding bag with the given number of embeddings
        # and dimension, all weights 0.5.
        gLogger.info("Initing RemoteEM with %s %s", num_embeddings, embedding_dim)
        super().__init__()
        init_em = [0.5] * embedding_dim
        self.em = nn.EmbeddingBag(
            num_embeddings,
            embedding_dim,
            _weight=torch.tensor([init_em] * num_embeddings),
        )

    def forward(self, input: torch.Tensor):
        # [FUNCTION DOC] Runs the embedding bag over the input with offsets that make each element
        # its own bag, so the output has one row per input element. Returns the embedding tensor.
        gLogger.debug("Running RemoteEM.forward() on: %s", input)
        return self.em(input, offsets=torch.LongTensor(range(input.shape[0])))


# Return a linear module with predefined parameters.
def getLinear(d_in, d_out):
    # [FUNCTION DOC] HELPER: Creates a bias-free linear layer whose weight is all ones except the
    # first entry which is minus one, with grad enabled, giving deterministic parameters. Returns
    # the layer.
    l = nn.Linear(d_in, d_out, bias=False)
    w = torch.ones((d_out, d_in))
    w[0][0] = -1
    w.requires_grad_()
    l.weight.data = w
    return l


class RemoteNet(nn.Module):
    # [CLASS DOC] Remote-side network: a deterministic linear layer followed by a ReLU.
    # Constructed on the remote worker via an RRef.
    def __init__(self, d_in: int, d_out: int):
        # [FUNCTION DOC] Logs and builds the linear layer (d_in to d_out) and the ReLU.
        gLogger.info("Initing RemoteNet with %s %s", d_in, d_out)
        super().__init__()
        self.fc = getLinear(d_in, d_out)
        self.relu = nn.ReLU()

    def forward(self, input: torch.Tensor):
        # [FUNCTION DOC] Applies the linear layer and ReLU to the input and returns the result.
        gLogger.debug("Running RemoteNet.forward() on: %s", input)
        return self.relu(self.fc(input))


class HybridModel(nn.Module):
    # [CLASS DOC] Local model combining local linear layers with remote modules reached through
    # RRefs. Attributes: the two RRefs, fc1 (dense features), fc2 (final layer, optionally wrapped
    # in DDP), and tuples non_ddp_params and ddp_params listing parameters that are or are not
    # synchronized by DDP.
    def __init__(
        self,
        remote_em_rref: rpc.RRef,
        remote_net_rref: rpc.RRef,
        process_group_for_ddp: dist.ProcessGroup = None,
    ):
        # [FUNCTION DOC] Stores the RRefs and creates fc1 and fc2. By default all parameters are
        # non-DDP. If a process group is given, fc2 is wrapped in DDP with it and its parameters
        # move to the DDP parameter tuple. Logs the number of parameter groups.
        super().__init__()
        self.remote_em_rref = remote_em_rref
        self.remote_net_rref = remote_net_rref
        self.fc1 = getLinear(D_DENSE, D_DENSE)
        self.fc2 = getLinear(D_HID, D_OUT)

        self.non_ddp_params = tuple(self.fc1.parameters()) + tuple(
            self.fc2.parameters()
        )
        self.ddp_params = ()

        if process_group_for_ddp is not None:
            self.non_ddp_params, self.ddp_params = (
                tuple(self.fc1.parameters()),
                tuple(self.fc2.parameters()),
            )
            gLogger.info("Use DDP for the second local net.")
            self.fc2 = DistributedDataParallel(
                self.fc2, check_reduction=True, process_group=process_group_for_ddp
            )

        gLogger.info(
            "HybridModel has %s groups of parameters.", len(list(self.parameters()))
        )

    def forward(self, input: FeatureSet):
        # [FUNCTION DOC] Forward pass: remotely runs the embedding on the sparse features, checks
        # the batch size matches the dense features (raises AssertionError if not), applies fc1 to
        # the dense features, concatenates dense and sparse outputs, runs the remote network on
        # that, then applies fc2. Returns the result.
        gLogger.debug("Running HybridModel.forward on %s", input)
        sparse = _remote_method(
            RemoteEM.forward, self.remote_em_rref, input.sparse_features
        )
        # The same size of mini batch.
        if sparse.shape[0] != input.dense_features.shape[0]:
            raise AssertionError(
                f"Expected sparse.shape[0] == input.dense_features.shape[0], "
                f"got {sparse.shape[0]} != {input.dense_features.shape[0]}"
            )
        dense = self.fc1(input.dense_features)
        x = torch.cat((dense, sparse), 1)
        gLogger.debug("Concatenated feature: %s", x)
        x = _remote_method(RemoteNet.forward, self.remote_net_rref, x)
        return self.fc2(x)


class Trainer:
    # [CLASS DOC] Per-trainer object created remotely on each trainer worker. Holds the rank, the
    # DDP process group over trainer ranks (only when DDP is used), the HybridModel (wrapped in
    # DDP in OUTSIDE mode), and the tuples of DDP and non-DDP parameters whose gradients are
    # reported.
    def __init__(
        self,
        remote_em_rref: rpc.RRef,
        remote_net_rref: rpc.RRef,
        ddp_mode: DdpMode,
        rank: int,
    ):
        # [FUNCTION DOC] Creates the trainer process group when DDP is used (a collective call),
        # builds the hybrid model (passing the group for INSIDE mode), and records the parameter
        # tuples. In OUTSIDE mode all parameters count as DDP parameters and the whole module is
        # wrapped in DDP. Logs the counts.
        self.rank = rank
        self.trainer_group = (
            dist.new_group(TRAINER_RANKS)
            if ddp_mode in (DdpMode.INSIDE, DdpMode.OUTSIDE)
            else None
        )
        self.remote_em_rref = remote_em_rref
        self.remote_net_rref = remote_net_rref
        self.hybrid_module = HybridModel(
            self.remote_em_rref,
            self.remote_net_rref,
            self.trainer_group if ddp_mode == DdpMode.INSIDE else None,
        )
        self.ddp_params, self.non_ddp_params = (
            self.hybrid_module.ddp_params,
            self.hybrid_module.non_ddp_params,
        )
        if ddp_mode == DdpMode.OUTSIDE:
            gLogger.info("Wrapping the whole hybrid module into DDP.")
            self.ddp_params += self.non_ddp_params
            self.non_ddp_params = ()
            self.hybrid_module = DistributedDataParallel(
                self.hybrid_module,
                check_reduction=True,
                process_group=self.trainer_group,
            )
        gLogger.info(
            "Succeeded in creating a HybridModel instance with "
            "%s ddp params and %s other local params.",
            len(self.ddp_params),
            len(self.non_ddp_params),
        )

    def destroy_pg(self):
        # [FUNCTION DOC] Destroys the trainer process group if one exists. Side effect on global
        # distributed state.
        if self.trainer_group:
            dist.destroy_process_group(self.trainer_group)

    def train_batch(
        self,
        mini_batch: FeatureSet,
        trainer_has_less_inputs: bool,
        simulate_uneven_inputs: bool,
    ):
        # [FUNCTION DOC] Runs distributed forward and backward for a batch. If simulating uneven
        # inputs, splits the batch into microbatches of two and, for trainers flagged with less
        # inputs, keeps only the first half. Runs each batch inside a distributed autograd context
        # (under the DDP join context manager when uneven), computes a loss, runs distributed
        # backward and fetches gradients. Returns a pair of tuples: gradients of DDP parameters
        # and of non-DDP parameters from the last context.
        # KNOWN ISSUE: the loss multiplies by the values of the whole mini batch rather than the
        # current microbatch, relying on broadcasting; with microbatches it mixes shapes and does
        # not use microbatch values.
        grads_dict = None

        if not simulate_uneven_inputs:
            input_batches = [mini_batch]
        else:
            # Split into microbatches, and trim to simulate uneven inputs.
            dense_features = mini_batch.dense_features
            sparse_features = mini_batch.sparse_features
            values = mini_batch.values

            dense_microbatch = torch.split(dense_features, 2)
            sparse_microbatch = torch.split(sparse_features, 2)
            values_microbatch = torch.split(values, 2)
            batches = []
            for d, s, v in zip(
                dense_microbatch, sparse_microbatch, values_microbatch, strict=True
            ):
                feature_set = FeatureSet(dense_features=d, sparse_features=s, values=v)
                batches.append(feature_set)

            if trainer_has_less_inputs:
                input_batches = batches[: len(batches) // 2]
                gLogger.info(
                    "Trainer reduced input patches from %s "
                    "to %s to simulate uneven inputs.",
                    len(batches),
                    len(input_batches),
                )
            else:
                input_batches = batches

        with (
            self.hybrid_module.join()
            if simulate_uneven_inputs
            else contextlib.nullcontext()
        ):
            for b in input_batches:
                with dist_autograd.context() as context_id:
                    output = self.hybrid_module.forward(b)
                    loss = (output * mini_batch.values).sum()
                    dist_autograd.backward(context_id, [loss])
                    grads_dict = dist_autograd.get_gradients(context_id)
                    gLogger.info(
                        "Loss is %s for mini batch: %s. Grads dict has %s entries: %s",
                        loss,
                        mini_batch,
                        len(grads_dict),
                        grads_dict,
                    )
        return (
            tuple(grads_dict[param] for param in self.ddp_params),
            tuple(grads_dict[param] for param in self.non_ddp_params),
        )


def get_training_examples():
    # [FUNCTION DOC] Builds sixteen training examples where each example has a partner with
    # identical features and opposite value, so gradients cancel under averaging, then splits them
    # evenly among the trainers. Raises AssertionError if the count is not divisible by the
    # trainer count. Returns a list of per-trainer feature sets.
    n = 16
    training_examples = FeatureSet(
        dense_features=torch.zeros((n, D_DENSE)),
        sparse_features=torch.zeros(n, dtype=torch.long),
        values=torch.zeros(n),
    )
    idx = 0
    # Every example has another one that has exactly the same features but an
    # opposite value. Therefore, their grads cancel each other in all-reduce.
    for value in (-1, 1):
        for x in (-1.0 * value, 1.0 * value):
            for y in (1.0 * value, -1.0 * value):
                for z in (0, 1):
                    training_examples.dense_features[idx, :] = torch.tensor((x, y))
                    training_examples.sparse_features[idx] = z
                    training_examples.values[idx] = value
                    idx += 1

    # Split the examples among NUM_TRAINERS trainers
    if n % NUM_TRAINERS != 0:
        raise AssertionError(
            f"Expected n % NUM_TRAINERS == 0, got {n} % {NUM_TRAINERS} = {n % NUM_TRAINERS}"
        )
    examples_per_trainer = int(n / NUM_TRAINERS)
    return [
        FeatureSet(
            dense_features=training_examples.dense_features[
                start : start + examples_per_trainer, :
            ],
            sparse_features=training_examples.sparse_features[
                start : start + examples_per_trainer
            ],
            values=training_examples.values[start : start + examples_per_trainer],
        )
        for start in range(0, n, examples_per_trainer)
    ]


shutdown_signal = threading.Condition()


def set_shutdown_signal():
    # [FUNCTION DOC] Notifies one waiter on the module shutdown condition; invoked via RPC to let
    # workers exit.
    global shutdown_signal
    with shutdown_signal:
        shutdown_signal.notify()


class DdpUnderDistAutogradTest(RpcAgentTestFixture):
    # [CLASS DOC] Distributed test case with a master, a remote worker and trainers. The master
    # creates remote modules and trainers, runs several training rounds over RPC and checks
    # gradients, then sends shutdown signals; other ranks wait for the signal. Test methods cover
    # no DDP, DDP outside, DDP outside with uneven inputs, and DDP inside.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] Returns the total number of processes (trainers, remote worker and
        # master).
        return WORLD_SIZE

    def remote_worker_name(self) -> str:
        # [FUNCTION DOC] Returns the RPC worker name of the remote worker.
        # The name has to be consistent with that in 'dist_init' decorator.
        return f"worker{REMOTE_WORKER_RANK}"

    def trainer_name(self, rank):
        # [FUNCTION DOC] Returns the RPC worker name for the given trainer rank.
        # The name has to be consistent with that in 'dist_init' decorator.
        return f"worker{rank}"

    def _init_process_group(self):
        # [FUNCTION DOC] HELPER: Initializes the default gloo process group from the file-based
        # init method and this test's rank and world size, then waits on a barrier.
        dist.init_process_group(
            backend="gloo",
            init_method=INIT_METHOD_TEMPLATE.format(file_name=self.file_name),
            world_size=self.world_size,
            rank=self.rank,
        )
        dist.barrier()

    def _remote_worker_process(self, ddp_mode):
        # [FUNCTION DOC] HELPER: Entry point for the remote worker rank: initializes the process
        # group, joins the trainer group creation when DDP is used (the call must happen on every
        # rank), blocks until the shutdown signal, then destroys the default group.
        gLogger.info("The remote worker is running.")
        self._init_process_group()

        if ddp_mode in (DdpMode.INSIDE, DdpMode.OUTSIDE):
            # new_group needs to be called on ranks.
            dist.new_group(TRAINER_RANKS)

        global shutdown_signal
        with shutdown_signal:
            shutdown_signal.wait()
        gLogger.info("Exiting remote worker.")
        dist.destroy_process_group()

    def _trainer_process(self, rank: int):
        # [FUNCTION DOC] HELPER: Entry point for a trainer rank: initializes the process group,
        # blocks until the shutdown signal arrives, then destroys the default group. The trainer's
        # work happens through RPC calls from the master.
        gLogger.info("Running the trainer #%s...", rank)
        gLogger.info(
            "Initing trainer process group by trainer #%s with ranks %s",
            rank,
            TRAINER_RANKS,
        )
        self._init_process_group()

        gLogger.info("Waiting for shutdown signal on trainer #%s...", rank)

        global shutdown_signal
        with shutdown_signal:
            shutdown_signal.wait()
        gLogger.info("Exiting the trainer #%s...", rank)
        dist.destroy_process_group()

    def _master_process(self, ddp_mode: DdpMode, simulate_uneven_inputs: bool):
        # [FUNCTION DOC] HELPER: Entry point for the master rank: initializes the process group,
        # creates remote RRefs for the embedding and network on the remote worker, and runs the
        # test body.
        gLogger.info("Running the master process...")
        self._init_process_group()

        remote_em_rref = rpc.remote(
            self.remote_worker_name(), RemoteEM, args=(NUM_EM_ROW, D_SPARSE)
        )
        remote_net_rref = rpc.remote(
            self.remote_worker_name(), RemoteNet, args=(D_DENSE + D_SPARSE, D_HID)
        )
        gLogger.info("Created remote rrefs on master")
        self.do_test_on_master(
            ddp_mode, simulate_uneven_inputs, remote_em_rref, remote_net_rref
        )

    def do_test_on_master(
        self,
        ddp_mode: DdpMode,
        simulate_uneven_inputs: bool,
        remote_em_rref: rpc.RRef,
        remote_net_rref: rpc.RRef,
    ):
        # [FUNCTION DOC] Test body on the master. Creates a Trainer on every trainer worker, joins
        # group creation if DDP is used, then for three rounds asynchronously trains one batch per
        # trainer (half the trainers get fewer inputs when simulating uneven inputs) and checks
        # gradients: DDP parameter gradients must be zero unless inputs are uneven, and non-DDP
        # gradients must be non-zero. Finally destroys trainer groups and sends shutdown signals
        # to trainers and the remote worker. Failed checks raise assertion errors.
        if simulate_uneven_inputs:
            gLogger.info(
                "Running DDP + RPC test with simulating uneven inputs across trainers."
            )

        trainer_rrefs = []
        for rank in TRAINER_RANKS:
            trainer = self.trainer_name(rank)
            trainer_rrefs.append(
                rpc.remote(
                    trainer,
                    Trainer,
                    args=(remote_em_rref, remote_net_rref, ddp_mode, rank),
                )
            )

        if ddp_mode in (DdpMode.INSIDE, DdpMode.OUTSIDE):
            # new_group needs to be called on ranks.
            dist.new_group(TRAINER_RANKS)

        training_examples = get_training_examples()
        for _ in range(3):
            futures = []
            num_trainers = len(trainer_rrefs)
            for idx, trainer_rref in enumerate(trainer_rrefs):
                # Half the trainers will deplete inputs earlier than the rest.
                trainer_has_less_inputs = (
                    simulate_uneven_inputs and idx < num_trainers // 2
                )
                futures.append(
                    _remote_method_async(
                        Trainer.train_batch,
                        trainer_rref,
                        training_examples[idx],
                        trainer_has_less_inputs,
                        simulate_uneven_inputs,
                    )
                )

            for future in futures:
                ddp_grads, non_ddp_grads = future.wait()
                # When there are uneven inputs, it is not necessary that grads
                # cancel each other out, since some trainers contribute 0 grad.
                if not simulate_uneven_inputs:
                    for grad in ddp_grads:
                        self.assertEqual(
                            grad,
                            torch.zeros_like(grad),
                            msg=f"The grad for any ddp parameter should be zeros, because "
                            "the training examples' grads cancel each other. Received "
                            f"gradient {grad}",
                        )
                for grad in non_ddp_grads:
                    self.assertNotEqual(
                        grad,
                        torch.zeros_like(grad),
                        msg="The grad for any non-ddp parameter shouldn't be zeros",
                    )

        # Destroy process groups
        for trainer_rref in trainer_rrefs:
            _remote_method_async(Trainer.destroy_pg, trainer_rref).wait()

        # Send shutdown signals.
        for rank in TRAINER_RANKS:
            trainer = self.trainer_name(rank)
            rpc.rpc_sync(trainer, set_shutdown_signal, args=())

        rpc.rpc_sync(self.remote_worker_name(), set_shutdown_signal, args=())

    def _do_test(self, ddp_mode, simulate_uneven_inputs=False):
        # [FUNCTION DOC] HELPER: Dispatches by rank to the master, remote worker or trainer
        # routine; raises RuntimeError for an unknown rank.
        if self.rank == MASTER_RANK:
            self._master_process(ddp_mode, simulate_uneven_inputs)
        elif self.rank == REMOTE_WORKER_RANK:
            self._remote_worker_process(ddp_mode)
        elif self.rank in TRAINER_RANKS:
            self._trainer_process(self.rank)
        else:
            raise RuntimeError(f"Unknown process rank: {self.rank}")

    @requires_gloo()
    @dist_init
    def test_backward_no_ddp(self):
        # [FUNCTION DOC] Test: distributed autograd with no DDP.
        self._do_test(DdpMode.NONE)

    @requires_gloo()
    @dist_init
    def test_backward_ddp_outside(self):
        # [FUNCTION DOC] Test: distributed autograd with DDP wrapping the whole hybrid model.
        self._do_test(DdpMode.OUTSIDE)

    @requires_gloo()
    @dist_init
    def test_backward_ddp_outside_uneven_inputs(self):
        # [FUNCTION DOC] Test: DDP wrapping the whole model with trainers given uneven numbers of
        # inputs.
        self._do_test(DdpMode.OUTSIDE, simulate_uneven_inputs=True)

    @requires_gloo()
    @dist_init
    def test_backward_ddp_inside(self):
        # [FUNCTION DOC] Test: distributed autograd with DDP wrapped around an inner layer.
        self._do_test(DdpMode.INSIDE)


# Common utils for both CPU and CUDA test suites
class CommonDdpComparisonTest(RpcAgentTestFixture):
    # [CLASS DOC] Base class for DDP versus local autograd comparison tests; world size is the
    # number of trainers.
    @property
    def world_size(self) -> int:
        # [FUNCTION DOC] Returns the number of trainers as the world size.
        return NUM_TRAINERS

    def trainer_name(self, rank):
        # [FUNCTION DOC] Returns the RPC worker name for the given rank.
        # The name has to be consistent with that in 'dist_init' decorator.
        return f"worker{rank}"

    @staticmethod
    def get_remote_grads(rref, context_id):
        # [FUNCTION DOC] HELPER: Static RPC target: returns the gradient of the weight of the
        # RRef's local module from the given distributed autograd context.
        return dist_autograd.get_gradients(context_id)[rref.local_value().weight]


class DdpComparisonTest(CommonDdpComparisonTest):
    # [CLASS DOC] CPU tests comparing gradients from distributed autograd with DDP against
    # gradients from local autograd.
    def _run_test_ddp_comparision(self, simulate_uneven_inputs=False):
        # [FUNCTION DOC] HELPER: Seeds by rank, initializes a gloo group on a separate file, wraps
        # a small linear layer in DDP and, per input (even ranks get extra inputs when simulating
        # uneven inputs, run under the DDP join context), computes gradients with distributed
        # autograd and with local autograd and asserts they are equal for every parameter.
        # Destroys the process group at the end.
        gLogger.info("Running trainer rank: %s", self.rank)
        # Each trainer uses a different random seed. Otherwise, they are going
        # to have exactly the same initial model parameters, input, and
        # therefore grads. That means the grads will be the same before and
        # after DDP's all-reduce.
        torch.manual_seed(self.rank)
        dist.init_process_group(
            backend="gloo",
            # Postfix file_name with "pg" since file_name is also used by RPC agent
            init_method=INIT_METHOD_TEMPLATE.format(file_name=f"{self.file_name}_pg"),
            world_size=self.world_size,
            rank=self.rank,
        )
        net = nn.Linear(2, 3)
        ddp_net = DistributedDataParallel(net)

        # Odd ranks join early if simulate_uneven_inputs.
        num_inputs = 1
        if simulate_uneven_inputs:
            if self.rank % 2 == 0:
                num_inputs += 2
        inputs_list = [torch.rand((3, 2)) for _ in range(num_inputs)]

        if simulate_uneven_inputs:
            gLogger.info(
                "Rank %s training with %s inputs.", self.rank, len(inputs_list)
            )

        # Use distributed autograd. The gradients will be in RPC context map.
        grads_dict = {}
        with ddp_net.join(simulate_uneven_inputs):
            for i, inputs in enumerate(inputs_list):
                with dist_autograd.context() as context_id:
                    loss = ddp_net(inputs).norm()
                    dist_autograd.backward(context_id, [loss])
                    grads_dict = dist_autograd.get_gradients(context_id)
                gLogger.info("Trainer #%s got grad dict: %s", self.rank, grads_dict)

                # Use local autograd. The gradients will be in each variable's '.grad'.
                ddp_net.zero_grad()
                loss = ddp_net(inputs).norm()
                loss.backward()

                # The gradients should be the same
                for param in net.parameters():
                    self.assertTrue(
                        param in grads_dict,
                        msg=f"Param {param} is not in dist_auto grad dict {grads_dict} for iteration {i}",
                    )
                    self.assertEqual(
                        grads_dict[param],
                        param.grad,
                        msg=f"The grads for param {param} are different under local "
                        f"and dist autograd: {param.grad} \n---\n {grads_dict[param]} for iteration {i}",
                    )
        dist.destroy_process_group()

    @requires_gloo()
    @dist_init
    def test_ddp_comparison(self):
        # [FUNCTION DOC] Test: DDP gradient comparison with even inputs.
        self._run_test_ddp_comparision()

    @requires_gloo()
    @dist_init
    def test_ddp_comparison_uneven_inputs(self):
        # [FUNCTION DOC] Test: DDP gradient comparison with uneven inputs.
        # test with simulating uneven inputs in DDP
        self._run_test_ddp_comparision(simulate_uneven_inputs=True)

    @requires_gloo()
    @dist_init
    def test_ddp_dist_autograd_sparse_grads(self):
        # [FUNCTION DOC] Test: with a sparse-gradient embedding bag under DDP, distributed
        # autograd yields exactly one gradient entry equal to the local gradient. Initializes a
        # process group (not destroyed here).
        # Each trainer uses a different random seed. Otherwise, they are going
        # to have exactly the same initial model parameters, input, and
        # therefore grads. That means the grads will be the same before and
        # after DDP's all-reduce.
        torch.manual_seed(self.rank)
        dist.init_process_group(
            backend="gloo",
            init_method=INIT_METHOD_TEMPLATE.format(file_name=self.file_name),
            world_size=self.world_size,
            rank=self.rank,
        )

        model = nn.EmbeddingBag(10, 3, sparse=True)
        ddp_model = DistributedDataParallel(model)

        # Different inputs for each
        input = torch.LongTensor(10).random_(0, 10)
        offsets = torch.LongTensor([0, 4])

        # Run local.
        loss = ddp_model(input, offsets).sum()
        loss.backward()

        with dist_autograd.context() as context_id:
            loss = ddp_model(input, offsets).sum()
            dist_autograd.backward(context_id, [loss])
            grads_dict = dist_autograd.get_gradients(context_id)
            self.assertEqual(1, len(grads_dict))
            self.assertEqual(model.weight.grad, grads_dict[model.weight])

    @requires_gloo()
    @dist_init
    def test_ddp_dist_autograd_local_vs_remote(self):
        # [FUNCTION DOC] Test: for remote device strings with and without an explicit cpu device,
        # compares a remote linear module plus DDP local layer under distributed autograd against
        # equivalent all-local autograd, checking local and remote gradients. Initializes a
        # process group (not destroyed here).
        # Each trainer uses a different random seed. Otherwise, they are going
        # to have exactly the same initial model parameters, input, and
        # therefore grads. That means the grads will be the same before and
        # after DDP's all-reduce.
        torch.manual_seed(self.rank)
        dist.init_process_group(
            backend="gloo",
            init_method=INIT_METHOD_TEMPLATE.format(file_name=self.file_name),
            world_size=self.world_size,
            rank=self.rank,
        )

        # Use two different remote device input string, w/ and w/o the default
        # device string "cpu", respectively.
        for remote_device in ["worker0/cpu", "worker0"]:
            remote_layer1 = RemoteModule(
                remote_device=remote_device, module_cls=nn.Linear, args=(10, 5, False)
            )
            layer1 = nn.Linear(10, 5, False)
            # Start with the same parameters for remote and local
            layer1.weight = remote_layer1.module_rref.to_here().weight

            # Run local case.
            layer2 = nn.Linear(5, 1)
            inputs = torch.rand((10, 10))
            ddp_model = DistributedDataParallel(layer2)
            loss = ddp_model(layer1(inputs)).sum()
            loss.backward()

            # Run remote case.
            with dist_autograd.context() as context_id:
                loss = ddp_model(remote_layer1(inputs)).sum()
                dist_autograd.backward(context_id, [loss])
                grads_dict = dist_autograd.get_gradients(context_id)
                dist.barrier()
                self.assertEqual(layer2.weight.grad, grads_dict[layer2.weight])
                self.assertEqual(
                    layer1.weight.grad,
                    rpc.rpc_sync(
                        "worker0",
                        CommonDdpComparisonTest.get_remote_grads,
                        args=(remote_layer1.module_rref, context_id),
                    ),
                )


class CudaDdpComparisonTest(CommonDdpComparisonTest):
    # [CLASS DOC] CUDA test comparing distributed autograd with RemoteModule and DDP layers
    # against local autograd.
    @skip_if_lt_x_gpu(NUM_TRAINERS)
    @requires_nccl()
    @dist_init
    @skip_if_rocm_multiprocess
    def test_ddp_dist_autograd_local_vs_remote_gpu(self):
        # [FUNCTION DOC] Test: chains remote CPU linear layers with DDP layers on this rank's GPU,
        # runs local and distributed autograd, and checks gradients of all four layers match
        # (remote ones fetched via RPC). Requires enough GPUs, NCCL and a non-ROCm platform.
        # Process group uses gloo.
        # Each trainer uses a different random seed. Otherwise, they are going
        # to have exactly the same initial model parameters, input, and
        # therefore grads. That means the grads will be the same before and
        # after DDP's all-reduce.
        torch.manual_seed(self.rank)
        dist.init_process_group(
            backend="gloo",
            init_method=INIT_METHOD_TEMPLATE.format(file_name=self.file_name),
            world_size=self.world_size,
            rank=self.rank,
        )

        remote_layer1 = RemoteModule(
            remote_device="worker0/cpu", module_cls=nn.Linear, args=(10, 7, False)
        )
        layer1 = nn.Linear(10, 7, False)
        # Start with the same parameters for remote and local
        layer1.weight = remote_layer1.module_rref.to_here().weight

        layer2 = nn.Linear(7, 5).cuda(self.rank)
        ddp_layer2 = DistributedDataParallel(layer2, device_ids=[self.rank])

        remote_layer3 = RemoteModule(
            remote_device="worker0/cpu", module_cls=nn.Linear, args=(5, 3, False)
        )
        layer3 = nn.Linear(5, 3, False)
        # Start with the same parameters for remote and local
        layer3.weight = remote_layer3.module_rref.to_here().weight

        layer4 = nn.Linear(3, 1).cuda(self.rank)
        ddp_layer4 = DistributedDataParallel(layer4, device_ids=[self.rank])

        # Run local case.
        inputs = torch.rand((10, 10))
        loss = ddp_layer4(
            layer3(ddp_layer2(layer1(inputs).cuda(self.rank)).cpu()).cuda(self.rank)
        ).sum()
        loss.backward()

        # Run remote case.
        with dist_autograd.context() as context_id:
            loss = ddp_layer4(
                remote_layer3(
                    ddp_layer2(remote_layer1(inputs).cuda(self.rank)).cpu()
                ).cuda(self.rank)
            ).sum()
            dist_autograd.backward(context_id, [loss])
            grads_dict = dist_autograd.get_gradients(context_id)
            dist.barrier()
            self.assertEqual(
                layer1.weight.grad,
                rpc.rpc_sync(
                    "worker0",
                    CommonDdpComparisonTest.get_remote_grads,
                    args=(remote_layer1.module_rref, context_id),
                ),
            )
            self.assertEqual(layer2.weight.grad, grads_dict[layer2.weight])
            self.assertEqual(
                layer3.weight.grad,
                rpc.rpc_sync(
                    "worker0",
                    CommonDdpComparisonTest.get_remote_grads,
                    args=(remote_layer3.module_rref, context_id),
                ),
            )
            self.assertEqual(layer4.weight.grad, grads_dict[layer4.weight])
