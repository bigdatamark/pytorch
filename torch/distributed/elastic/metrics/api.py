#!/usr/bin/env python3
# mypy: allow-untyped-defs
# [MODULE DOC] Metrics API for torchelastic: MetricData tuple, pluggable handlers (console, null),
# MetricStream per group, module-level handler registry (global mapping plus default handler,
# initially null), configure/getStream, put_metric, the prof decorator and deprecated
# profile/publish_metric helpers, and get_elapsed_time_ms.

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import time
from collections import namedtuple
from functools import wraps
from typing_extensions import deprecated


__all__ = [
    "MetricsConfig",
    "MetricHandler",
    "ConsoleMetricHandler",
    "NullMetricHandler",
    "MetricStream",
    "configure",
    "getStream",
    "prof",
    "profile",
    "put_metric",
    "publish_metric",
    "get_elapsed_time_ms",
    "MetricData",
]

MetricData = namedtuple("MetricData", ["timestamp", "group_name", "name", "value"])


class MetricsConfig:
    # [CLASS DOC] Simple holder (slots) of a params mapping of string settings for metrics configuration.
    __slots__ = ["params"]

    def __init__(self, params: dict[str, str] | None = None):
        # [FUNCTION DOC] Stores params, using an empty mapping when none is given.
        self.params = params
        if self.params is None:
            self.params = {}


class MetricHandler(abc.ABC):
    # [CLASS DOC] Abstract base for metric sinks; subclasses implement emit to consume MetricData.
    @abc.abstractmethod
    def emit(self, metric_data: MetricData):
        # [FUNCTION DOC] Abstract: consume one MetricData record. Base body does nothing.
        pass


class ConsoleMetricHandler(MetricHandler):
    # [CLASS DOC] Handler that prints each metric to standard output.
    def emit(self, metric_data: MetricData):
        # [FUNCTION DOC] Prints one line with timestamp, group name, metric name and value.
        print(
            f"[{metric_data.timestamp}][{metric_data.group_name}]: {metric_data.name}={metric_data.value}"
        )


class NullMetricHandler(MetricHandler):
    # [CLASS DOC] Handler that discards all metrics (the default).
    def emit(self, metric_data: MetricData):
        # [FUNCTION DOC] Does nothing.
        pass


class MetricStream:
    # [CLASS DOC] Binds a metric group name to a handler; add_value sends timestamped records to it.
    def __init__(self, group_name: str, handler: MetricHandler):
        # [FUNCTION DOC] Stores the group name and handler.
        self.group_name = group_name
        self.handler = handler

    def add_value(self, metric_name: str, metric_value: int):
        # [FUNCTION DOC] Emits a MetricData record stamped with the current time, this group,
        # the metric name and value through the handler.
        self.handler.emit(
            MetricData(time.time(), self.group_name, metric_name, metric_value)
        )


_metrics_map: dict[str, MetricHandler] = {}
_default_metrics_handler: MetricHandler = NullMetricHandler()


# pyre-fixme[9]: group has type `str`; used as `None`.
def configure(handler: MetricHandler, group: str | None = None):
    # [FUNCTION DOC] Registers the handler for the named group, or replaces the global default
    # handler when group is None. Mutates module-level state; returns nothing.
    if group is None:
        global _default_metrics_handler
        # pyre-fixme[9]: _default_metrics_handler has type `NullMetricHandler`; used
        #  as `MetricHandler`.
        _default_metrics_handler = handler
    else:
        _metrics_map[group] = handler


def getStream(group: str):
    # [FUNCTION DOC] Returns a MetricStream for the group using its configured handler, or the default handler.
    handler = _metrics_map.get(group, _default_metrics_handler)
    return MetricStream(group, handler)


def _get_metric_name(fn):
    # [FUNCTION DOC] HELPER: Returns the qualified name of fn if it has a dotted qualified name
    # (method); otherwise the last module component plus the function name (or the bare name
    # when the module is empty).
    qualname = fn.__qualname__
    split = qualname.split(".")
    if len(split) == 1:
        module = fn.__module__
        if module:
            return module.split(".")[-1] + "." + split[0]
        else:
            return split[0]
    else:
        return qualname


def prof(fn=None, group: str = "torchelastic"):
    # [FUNCTION DOC] Decorator usable bare or with a group argument. The wrapped function publishes
    # a success count on normal return, a failure count on exception (which is re-raised), and
    # always a duration in milliseconds, all named by the function's metric name, to the group.
    r"""
    @profile decorator publishes duration.ms, count, success, failure metrics for the function that it decorates.

    The metric name defaults to the qualified name (``class_name.def_name``) of the function.
    If the function does not belong to a class, it uses the leaf module name instead.

    Usage

    ::

     @metrics.prof
     def x():
         pass


     @metrics.prof(group="agent")
     def y():
         pass
    """

    def wrap(f):
        # [FUNCTION DOC] HELPER: Builds the metric-publishing wrapper around f.
        @wraps(f)
        def wrapper(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Calls f and publishes success/failure/duration metrics; re-raises exceptions.
            key = _get_metric_name(f)
            try:
                start = time.time()
                result = f(*args, **kwargs)
                put_metric(f"{key}.success", 1, group)
            except Exception:
                put_metric(f"{key}.failure", 1, group)
                raise
            finally:
                put_metric(f"{key}.duration.ms", get_elapsed_time_ms(start), group)  # type: ignore[possibly-undefined]
            return result

        return wrapper

    if fn:
        return wrap(fn)
    else:
        return wrap


@deprecated("Deprecated, use `@prof` instead", category=FutureWarning)
def profile(group=None):
    # [FUNCTION DOC] Deprecated (emits a FutureWarning) decorator factory; the wrapped function
    # publishes success/failure counts and a duration in milliseconds to the given group via
    # publish_metric, re-raising exceptions. Use prof instead.
    """
    @profile decorator adds latency and success/failure metrics to any given function.

    Usage

    ::

     @metrics.profile("my_metric_group")
     def some_function(<arguments>):
    """

    def wrap(func):
        # [FUNCTION DOC] HELPER: Builds the metric-publishing wrapper around func.
        @wraps(func)
        def wrapper(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Calls func and publishes success/failure/duration metrics; re-raises exceptions.
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                # pyrefly: ignore [bad-argument-type]
                publish_metric(group, f"{func.__name__}.success", 1)
            except Exception:
                # pyrefly: ignore [bad-argument-type]
                publish_metric(group, f"{func.__name__}.failure", 1)
                raise
            finally:
                publish_metric(
                    # pyrefly: ignore [bad-argument-type]
                    group,
                    f"{func.__name__}.duration.ms",
                    get_elapsed_time_ms(start_time),  # type: ignore[possibly-undefined]
                )
            return result

        return wrapper

    return wrap


def put_metric(metric_name: str, metric_value: int, metric_group: str = "torchelastic"):
    # [FUNCTION DOC] Publishes one data point (name, integer value) to the group's stream.
    """
    Publish a metric data point.

    Usage

    ::

     put_metric("metric_name", 1)
     put_metric("metric_name", 1, "metric_group_name")
    """
    getStream(metric_group).add_value(metric_name, metric_value)


@deprecated(
    "Deprecated, use `put_metric(metric_group)(metric_name, metric_value)` instead",
    category=FutureWarning,
)
def publish_metric(metric_group: str, metric_name: str, metric_value: int):
    # [FUNCTION DOC] Deprecated (FutureWarning): publishes one data point to the group's stream; same as put_metric with different argument order.
    metric_stream = getStream(metric_group)
    metric_stream.add_value(metric_name, metric_value)


def get_elapsed_time_ms(start_time_in_seconds: float):
    # [FUNCTION DOC] Returns the whole milliseconds elapsed since the given wall-clock start time in seconds.
    """Return the elapsed time in millis from the given start time."""
    end_time = time.time()
    return int((end_time - start_time_in_seconds) * 1000)
