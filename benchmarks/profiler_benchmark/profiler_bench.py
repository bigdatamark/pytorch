# [MODULE DOC] Command-line benchmark measuring the overhead of the autograd profiler. It times
# a matrix-multiplication workload (sequential loop, or split across forked jit tasks) with the
# profiler disabled and then enabled, optionally on cuda, traced to script, with stacks or
# kineto, and prints the timer result for each. Module globals hold the parallel task count and
# the iteration count, which the main block sets from options. All work happens under the main
# guard; options are described there.
import argparse
import sys
import timeit

import torch
from torch.utils.benchmark import Timer


PARALLEL_TASKS_NUM = 4
INTERNAL_ITER = None


def loop_workload(x):
    # [FUNCTION DOC] HELPER: Sequential workload. Repeatedly squares the input matrix by matrix
    # multiplication for the number of iterations held in the module global, and returns the
    # final matrix. Fails with a type error if the global was never set (it starts as none).
    for i in range(INTERNAL_ITER):
        x = torch.mm(x, x)
    return x


def parallel_workload(x):
    # [FUNCTION DOC] HELPER: Parallel workload. Forks several tasks (the module-wide task count)
    # through the jit fork mechanism, each squaring the input by matrix multiplication for the
    # total iteration count divided by the task count (integer part), then waits for all of
    # them. Returns the original input, not any task result; results of the tasks are discarded.
    def parallel_task(x):
        # [FUNCTION DOC] HELPER: One forked unit of work: repeatedly squares its argument by
        # matrix multiplication for the per-task iteration count and returns the result.
        for i in range(int(INTERNAL_ITER / PARALLEL_TASKS_NUM)):
            x = torch.mm(x, x)
        return x

    futs = []
    for i in range(PARALLEL_TASKS_NUM):
        futs.append(torch.jit._fork(parallel_task, x))
    for i in range(PARALLEL_TASKS_NUM):
        torch.jit._wait(futs[i])
    return x


if __name__ == "__main__":
    torch._C._set_graph_executor_optimize(False)
    parser = argparse.ArgumentParser(description="Profiler benchmark")

    parser.add_argument("--with-cuda", "--with_cuda", action="store_true")
    parser.add_argument("--with-stack", "--with_stack", action="store_true")
    parser.add_argument("--use-script", "--use_script", action="store_true")
    parser.add_argument("--use-kineto", "--use_kineto", action="store_true")
    parser.add_argument(
        "--profiling-tensor-size", "--profiling_tensor_size", default=1, type=int
    )
    parser.add_argument("--workload", "--workload", default="loop", type=str)
    parser.add_argument("--internal-iter", "--internal_iter", default=256, type=int)
    parser.add_argument(
        "--timer-min-run-time", "--timer_min_run_time", default=10, type=int
    )
    parser.add_argument("--cuda-only", "--cuda_only", action="store_true")

    args = parser.parse_args()

    if args.with_cuda and not torch.cuda.is_available():
        print("No CUDA available")
        sys.exit()

    print(
        f"Payload: {args.workload}, {args.internal_iter} iterations; timer min. runtime = {args.timer_min_run_time}\n"
    )
    INTERNAL_ITER = args.internal_iter

    for profiling_enabled in [False, True]:
        print(
            "Profiling {}, tensor size {}x{}, use cuda: {}, use kineto: {}, with stacks: {}, use script: {}".format(
                "enabled" if profiling_enabled else "disabled",
                args.profiling_tensor_size,
                args.profiling_tensor_size,
                args.with_cuda,
                args.use_kineto,
                args.with_stack,
                args.use_script,
            )
        )

        input_x = torch.rand(args.profiling_tensor_size, args.profiling_tensor_size)

        if args.with_cuda:
            input_x = input_x.cuda()

        workload = None
        if args.workload not in ["loop", "parallel"]:
            raise AssertionError(
                f"args.workload must be 'loop' or 'parallel', but got '{args.workload}'"
            )
        if args.workload == "loop":
            workload = loop_workload
        else:
            workload = parallel_workload

        if args.use_script:
            traced_workload = torch.jit.trace(workload, (input_x,))
            workload = traced_workload

        if profiling_enabled:

            def payload():
                # [FUNCTION DOC] HELPER: Timed callable used when profiling is on. Runs the
                # selected workload on the captured input inside an autograd profiler context
                # configured from the command-line flags (device, stack recording, kineto, cpu
                # usage) and returns the workload output. Profiler results are discarded.
                x = None
                with torch.autograd.profiler.profile(
                    use_device="cuda" if args.with_cuda else None,
                    with_stack=args.with_stack,
                    use_kineto=args.use_kineto,
                    use_cpu=not args.cuda_only,
                ):
                    x = workload(input_x)
                return x

        else:

            def payload():
                # [FUNCTION DOC] HELPER: Timed callable used when profiling is off: runs the
                # selected workload on the captured input and returns its output.
                return workload(input_x)

        t = Timer(
            "payload()",
            globals={"payload": payload},
            timer=timeit.default_timer,
        ).blocked_autorange(min_run_time=args.timer_min_run_time)
        print(t)
