# [MODULE DOC] Unit tests for torch.distributed.elastic.rendezvous.utils: the configuration
# string parser, port and endpoint parsing, hostname matching against the local machine, the
# sleep helper and the _PeriodicTimer background timer. Some tests use real sockets, DNS lookups
# of the local host and short real sleeps; two patch socket.getaddrinfo.
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import socket
import threading
import time
from datetime import timedelta
from unittest import TestCase
from unittest.mock import patch

from torch.distributed.elastic.rendezvous.utils import (
    _delay,
    _matches_machine_hostname,
    _parse_rendezvous_config,
    _PeriodicTimer,
    _try_parse_port,
    parse_rendezvous_endpoint,
)


class UtilsTest(TestCase):
    # [CLASS DOC] Tests for the stateless helper functions: _parse_rendezvous_config,
    # _try_parse_port, parse_rendezvous_endpoint, _matches_machine_hostname and _delay.
    def test_parse_rendezvous_config_returns_dict(self) -> None:
        # [FUNCTION DOC] A key=value list with extra whitespace, values containing an equals
        # sign or a slash, and keys in any order must parse into a mapping from stripped key to
        # stripped value.
        expected_config = {
            "a": "dummy1",
            "b": "dummy2",
            "c": "dummy3=dummy4",
            "d": "dummy5/dummy6",
        }

        config = _parse_rendezvous_config(
            " b= dummy2  ,c=dummy3=dummy4,  a =dummy1,d=dummy5/dummy6"
        )

        self.assertEqual(config, expected_config)

    def test_parse_rendezvous_returns_empty_dict_if_str_is_empty(self) -> None:
        # [FUNCTION DOC] An empty or all-whitespace string must parse into an empty mapping.
        config_strs = ["", "   "]

        for config_str in config_strs:
            with self.subTest(config_str=config_str):
                config = _parse_rendezvous_config(config_str)

                self.assertEqual(config, {})

    def test_parse_rendezvous_raises_error_if_str_is_invalid(self) -> None:
        # [FUNCTION DOC] Malformed strings (trailing comma, empty entries, entries with
        # only an equals sign or an empty key) must raise ValueError with the "must be in
        # format" message.
        config_strs = [
            "a=dummy1,",
            "a=dummy1,,c=dummy2",
            "a=dummy1,   ,c=dummy2",
            "a=dummy1,=  ,c=dummy2",
            "a=dummy1, = ,c=dummy2",
            "a=dummy1,  =,c=dummy2",
            " ,  ",
        ]

        for config_str in config_strs:
            with self.subTest(config_str=config_str):
                with self.assertRaisesRegex(
                    ValueError,
                    r"^The rendezvous configuration string must be in format "
                    r"<key1>=<value1>,...,<keyN>=<valueN>.$",
                ):
                    _parse_rendezvous_config(config_str)

    def test_parse_rendezvous_raises_error_if_value_is_empty(self) -> None:
        # [FUNCTION DOC] Entries with a key but no value (no equals sign or an empty value) must
        # raise ValueError saying option 'a' must have a value specified.
        config_strs = [
            "b=dummy1,a,c=dummy2",
            "b=dummy1,c=dummy2,a",
            "b=dummy1,a=,c=dummy2",
            "  a ",
        ]

        for config_str in config_strs:
            with self.subTest(config_str=config_str):
                with self.assertRaisesRegex(
                    ValueError,
                    r"^The rendezvous configuration option 'a' must have a value specified.$",
                ):
                    _parse_rendezvous_config(config_str)

    def test_try_parse_port_returns_port(self) -> None:
        # [FUNCTION DOC] _try_parse_port on a plain digit string returns the integer.
        port = _try_parse_port("123")

        self.assertEqual(port, 123)

    def test_try_parse_port_returns_none_if_str_is_invalid(self) -> None:
        # [FUNCTION DOC] Empty strings, strings with surrounding whitespace and non-numeric text
        # must make _try_parse_port return None.
        port_strs = [
            "",
            "   ",
            "  1",
            "1  ",
            " 1 ",
            "abc",
        ]

        for port_str in port_strs:
            with self.subTest(port_str=port_str):
                port = _try_parse_port(port_str)

                self.assertIsNone(port)

    def test_parse_rendezvous_endpoint_returns_tuple(self) -> None:
        # [FUNCTION DOC] For DNS names, IPv4 addresses and bracketed IPv6 addresses with ports
        # 0, 123 and 65535, parse_rendezvous_endpoint must return the host (brackets removed)
        # and the integer port.
        endpoints = [
            "dummy.com:0",
            "dummy.com:123",
            "dummy.com:65535",
            "dummy-1.com:0",
            "dummy-1.com:123",
            "dummy-1.com:65535",
            "123.123.123.123:0",
            "123.123.123.123:123",
            "123.123.123.123:65535",
            "[2001:db8::1]:0",
            "[2001:db8::1]:123",
            "[2001:db8::1]:65535",
        ]

        for endpoint in endpoints:
            with self.subTest(endpoint=endpoint):
                host, port = parse_rendezvous_endpoint(endpoint, default_port=123)

                expected_host, expected_port = endpoint.rsplit(":", 1)

                if expected_host[0] == "[" and expected_host[-1] == "]":
                    expected_host = expected_host[1:-1]

                self.assertEqual(host, expected_host)
                self.assertEqual(port, int(expected_port))

    def test_parse_rendezvous_endpoint_returns_tuple_if_endpoint_has_no_port(
        self,
    ) -> None:
        # [FUNCTION DOC] Endpoints without a port must return the host (brackets removed for
        # IPv6) and the supplied default port, 123.
        endpoints = ["dummy.com", "dummy-1.com", "123.123.123.123", "[2001:db8::1]"]

        for endpoint in endpoints:
            with self.subTest(endpoint=endpoint):
                host, port = parse_rendezvous_endpoint(endpoint, default_port=123)

                expected_host = endpoint

                if expected_host[0] == "[" and expected_host[-1] == "]":
                    expected_host = expected_host[1:-1]

                self.assertEqual(host, expected_host)
                self.assertEqual(port, 123)

    def test_parse_rendezvous_endpoint_returns_tuple_if_endpoint_is_empty(self) -> None:
        # [FUNCTION DOC] An empty endpoint must give host localhost and the default port.
        # KNOWN ISSUE: the loop variable is never used; both iterations pass an empty string, so
        # the whitespace-only case is not actually exercised.
        endpoints = ["", "  "]

        for endpoint in endpoints:
            with self.subTest(endpoint=endpoint):
                host, port = parse_rendezvous_endpoint("", default_port=123)

                self.assertEqual(host, "localhost")
                self.assertEqual(port, 123)

    def test_parse_rendezvous_endpoint_raises_error_if_hostname_is_invalid(
        self,
    ) -> None:
        # [FUNCTION DOC] Endpoints with invalid hostnames (a tilde, a space before the colon, an
        # empty host) must raise ValueError saying the hostname must be a dot-separated list of
        # labels, an IPv4 address or an IPv6 address.
        endpoints = ["~", "dummy.com :123", "~:123", ":123"]

        for endpoint in endpoints:
            with self.subTest(endpoint=endpoint):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"^The hostname of the rendezvous endpoint '{endpoint}' must be a "
                    r"dot-separated list of labels, an IPv4 address, or an IPv6 address.$",
                ):
                    parse_rendezvous_endpoint(endpoint, default_port=123)

    def test_parse_rendezvous_endpoint_raises_error_if_port_is_invalid(self) -> None:
        # [FUNCTION DOC] Endpoints whose port is empty, non-numeric or negative must raise
        # ValueError saying the port must be an integer between 0 and 65536.
        endpoints = ["dummy.com:", "dummy.com:abc", "dummy.com:-123", "dummy.com:-"]

        for endpoint in endpoints:
            with self.subTest(endpoint=endpoint):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"^The port number of the rendezvous endpoint '{endpoint}' must be an integer "
                    r"between 0 and 65536.$",
                ):
                    parse_rendezvous_endpoint(endpoint, default_port=123)

    def test_parse_rendezvous_endpoint_raises_error_if_port_is_too_big(self) -> None:
        # [FUNCTION DOC] Ports 65536 and 70000 must raise the same port-range ValueError.
        endpoints = ["dummy.com:65536", "dummy.com:70000"]

        for endpoint in endpoints:
            with self.subTest(endpoint=endpoint):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"^The port number of the rendezvous endpoint '{endpoint}' must be an integer "
                    r"between 0 and 65536.$",
                ):
                    parse_rendezvous_endpoint(endpoint, default_port=123)

    def test_matches_machine_hostname_returns_true_if_hostname_is_loopback(
        self,
    ) -> None:
        # [FUNCTION DOC] localhost, 127.0.0.1 and both short and long IPv6 loopback forms must
        # be reported as matching the machine.
        hosts = [
            "localhost",
            "127.0.0.1",
            "::1",
            "0000:0000:0000:0000:0000:0000:0000:0001",
        ]

        for host in hosts:
            with self.subTest(host=host):
                self.assertTrue(_matches_machine_hostname(host))

    def test_matches_machine_hostname_returns_true_if_hostname_is_machine_hostname(
        self,
    ) -> None:
        # [FUNCTION DOC] The machine's own hostname must be reported as matching.
        host = socket.gethostname()

        self.assertTrue(_matches_machine_hostname(host))

    def test_matches_machine_hostname_returns_true_if_hostname_is_machine_fqdn(
        self,
    ) -> None:
        # [FUNCTION DOC] The machine's fully qualified domain name must be reported as matching.
        host = socket.getfqdn()

        self.assertTrue(_matches_machine_hostname(host))

    def test_matches_machine_hostname_returns_true_if_hostname_is_machine_address(
        self,
    ) -> None:
        # [FUNCTION DOC] Every address resolved for the machine hostname must be reported
        # as matching.
        addr_list = socket.getaddrinfo(
            socket.gethostname(), None, proto=socket.IPPROTO_TCP
        )

        for addr in (addr_info[4][0] for addr_info in addr_list):
            with self.subTest(addr=addr):
                self.assertTrue(_matches_machine_hostname(addr))

    def test_matches_machine_hostname_returns_false_if_hostname_does_not_match(
        self,
    ) -> None:
        # [FUNCTION DOC] An unknown name, the unspecified IPv4 address and a different IPv6
        # address must not match the machine.
        hosts = ["dummy", "0.0.0.0", "::2"]

        for host in hosts:
            with self.subTest(host=host):
                self.assertFalse(_matches_machine_hostname(host))

    def test_delay_suspends_thread(self) -> None:
        # [FUNCTION DOC] _delay with a single number of seconds and with a (min, max) tuple must
        # block the calling thread for at least 0.2 seconds, measured with a monotonic clock.
        for seconds in 0.2, (0.2, 0.4):
            with self.subTest(seconds=seconds):
                time1 = time.monotonic()

                _delay(seconds)  # type: ignore[arg-type]

                time2 = time.monotonic()

                self.assertGreaterEqual(time2 - time1, 0.2)

    @patch(
        "socket.getaddrinfo",
        side_effect=[
            [(None, None, 0, "a_host", ("1.2.3.4", 0))],
            [(None, None, 0, "a_different_host", ("1.2.3.4", 0))],
        ],
    )
    def test_matches_machine_hostname_returns_true_if_ip_address_match_between_hosts(
        self,
        _0,
    ) -> None:
        # [FUNCTION DOC] With getaddrinfo patched so that both the given host and the local host
        # resolve to the same IP address, a host with a different name must still match the
        # machine. The patched mock argument is unused.
        self.assertTrue(_matches_machine_hostname("a_host"))

    @patch(
        "socket.getaddrinfo",
        side_effect=[
            [(None, None, 0, "a_host", ("1.2.3.4", 0))],
            [(None, None, 0, "another_host_with_different_ip", ("1.2.3.5", 0))],
        ],
    )
    def test_matches_machine_hostname_returns_false_if_ip_address_not_match_between_hosts(
        self,
        _0,
    ) -> None:
        # [FUNCTION DOC] With getaddrinfo patched so that the given host and the local host
        # resolve to different IP addresses, the host must not match the machine. The patched
        # mock argument is unused.
        self.assertFalse(_matches_machine_hostname("a_host"))


class PeriodicTimerTest(TestCase):
    # [CLASS DOC] Tests for _PeriodicTimer, a helper that runs a function on a background thread
    # at a fixed interval until cancelled or garbage collected. Covers start-once, cancel
    # idempotence, thread cleanup on cancel and delete, naming rules and call spacing.
    def test_start_can_be_called_only_once(self) -> None:
        # [FUNCTION DOC] Starting a timer twice must raise RuntimeError saying it has already
        # started; the timer is cancelled afterward.
        timer = _PeriodicTimer(timedelta(seconds=1), lambda: None)

        timer.start()

        with self.assertRaisesRegex(RuntimeError, r"^The timer has already started.$"):
            timer.start()

        timer.cancel()

    def test_cancel_can_be_called_multiple_times(self) -> None:
        # [FUNCTION DOC] Calling cancel repeatedly on a started timer must not raise.
        timer = _PeriodicTimer(timedelta(seconds=1), lambda: None)

        timer.start()

        timer.cancel()
        timer.cancel()

    def test_cancel_stops_background_thread(self) -> None:
        # [FUNCTION DOC] After set_name and start, a live thread with that name must exist;
        # after cancel, no thread with that name may remain.
        name = "PeriodicTimer_CancelStopsBackgroundThreadTest"

        timer = _PeriodicTimer(timedelta(seconds=1), lambda: None)

        timer.set_name(name)

        timer.start()

        self.assertTrue(any(t.name == name for t in threading.enumerate()))

        timer.cancel()

        self.assertTrue(all(t.name != name for t in threading.enumerate()))

    def test_delete_stops_background_thread(self) -> None:
        # [FUNCTION DOC] Same as the cancel test, but the timer is destroyed with del instead of
        # cancel; the background thread must be gone afterward (the timer's finalizer stops it).
        name = "PeriodicTimer_DeleteStopsBackgroundThreadTest"

        timer = _PeriodicTimer(timedelta(seconds=1), lambda: None)

        timer.set_name(name)

        timer.start()

        self.assertTrue(any(t.name == name for t in threading.enumerate()))

        del timer

        self.assertTrue(all(t.name != name for t in threading.enumerate()))

    def test_set_name_cannot_be_called_after_start(self) -> None:
        # [FUNCTION DOC] Calling set_name after the timer started must raise RuntimeError saying
        # the timer has already started.
        timer = _PeriodicTimer(timedelta(seconds=1), lambda: None)

        timer.start()

        with self.assertRaisesRegex(RuntimeError, r"^The timer has already started.$"):
            timer.set_name("dummy_name")

        timer.cancel()

    def test_timer_calls_background_thread_at_regular_intervals(self) -> None:
        # [FUNCTION DOC] Runs a timer with a 0.2 second interval whose callback records the time
        # since the previous call and stops the wait after 4 calls (waiting up to 60 seconds).
        # After cancel, asserts at least 4 calls happened and that every measured gap is at
        # least the interval. Uses a lower bound on call count to avoid flakiness.
        timer_begin_time: float

        # Call our function every 200ms.
        call_interval = 0.2

        # Keep the log of intervals between each consecutive call.
        actual_call_intervals: list[float] = []

        # Keep the number of times the function was called.
        call_count = 0

        # In order to prevent a flaky test instead of asserting that the
        # function was called an exact number of times we use a lower bound
        # that is guaranteed to be true for a correct implementation.
        min_required_call_count = 4

        timer_stop_event = threading.Event()

        def log_call(self):
            # [FUNCTION DOC] HELPER: Timer callback (takes the test case as an unused argument).
            # Appends the time since the previous call to the interval log, counts the call,
            # sets the stop event when the minimum call count is reached and resets the
            # reference time. Mutates enclosing test state.
            nonlocal timer_begin_time, call_count

            actual_call_intervals.append(time.monotonic() - timer_begin_time)

            call_count += 1
            if call_count == min_required_call_count:
                timer_stop_event.set()

            timer_begin_time = time.monotonic()

        timer = _PeriodicTimer(timedelta(seconds=call_interval), log_call, self)

        timer_begin_time = time.monotonic()

        timer.start()

        # Although this is theoretically non-deterministic, if our timer, which
        # has a 200ms call interval, does not get called 4 times in 60 seconds,
        # there is very likely something else going on.
        timer_stop_event.wait(60)

        timer.cancel()

        self.longMessage = False

        self.assertGreaterEqual(
            call_count,
            min_required_call_count,
            f"The function has been called {call_count} time(s) but expected to be called at least "
            f"{min_required_call_count} time(s).",
        )

        for actual_call_interval in actual_call_intervals:
            self.assertGreaterEqual(
                actual_call_interval,
                call_interval,
                f"The interval between two function calls was {actual_call_interval} second(s) but "
                f"expected to be at least {call_interval} second(s).",
            )
