"""Tests for the host-side serial test runner and firmware builder fixes."""

import asyncio
import sys
from pathlib import Path

import pytest

from mcu_diagnostic_agent.firmware.builder import FirmwareBuilder, FirmwareConfig
from mcu_diagnostic_agent.orchestrator.test_runner import TestRunner, TestStatus

FIRMWARE_OUTPUT = [
    '{"type":"device_info","device_id":"AABB","firmware_version":1}',
    "uart:~$ diag run all",
    "SUITE_START: basic",
    "TEST: cpu_id: PASS - ok",
    "TEST: ram_march_quick: FAIL - stuck bit",
    "TEST: flash_id: SKIP - not available",
    "SUITE: basic: passed=1 failed=1 skipped=1 duration_ms=12",
    "SUITE_START: gpio",
    "TEST: gpio_output: ERROR - driver missing",
    "SUITE: gpio: passed=0 failed=1 skipped=0 duration_ms=3",
    "=== DIAGNOSTIC SUMMARY ===",
]


def _feed(runner):
    runner._running = True
    for line in FIRMWARE_OUTPUT:
        runner._parse_output(line)


def test_text_protocol_counts_each_result_once():
    runner = TestRunner(probe=None, test_suites=["basic", "gpio"])
    _feed(runner)
    suites = asyncio.run(runner.get_results())

    assert [s.name for s in suites] == ["basic", "gpio"]
    basic, gpio = suites
    assert (basic.passed, basic.failed, basic.skipped) == (1, 1, 1)
    assert [t.name for t in basic.tests] == ["cpu_id", "ram_march_quick", "flash_id"]
    assert basic.duration_ms == 12
    # ERROR results count as failures instead of vanishing from the totals.
    assert (gpio.passed, gpio.failed, gpio.skipped) == (0, 1, 0)
    assert gpio.tests[0].status is TestStatus.ERROR
    assert all(t.suite == "basic" for t in basic.tests)


def test_summary_line_stops_the_run():
    runner = TestRunner(probe=None, test_suites=["basic"])
    _feed(runner)
    assert runner._running is False


def test_unknown_json_status_does_not_raise():
    runner = TestRunner(probe=None, test_suites=["basic"])
    runner._parse_output('{"type":"test_result","name":"x","suite":"s","status":"weird"}')
    assert runner.current_suite.tests[0].status is TestStatus.ERROR


def test_run_without_serial_port_fails_loudly(monkeypatch):
    runner = TestRunner(probe=None, test_suites=["basic"])
    monkeypatch.setattr(runner, "_find_serial_port", lambda: None)
    with pytest.raises(RuntimeError, match="serial port"):
        asyncio.run(runner.run())


def test_builder_points_at_the_real_zephyr_app():
    builder = FirmwareBuilder()
    assert (builder.app_dir / "CMakeLists.txt").exists()
    assert builder.get_board("stm32", "STM32F407VG") == "stm32f4_disco"
    assert builder.get_board("stm32", None) == "unknown"


def test_builder_passes_source_dir_before_cmake_args(monkeypatch, tmp_path):
    captured = {}

    class FakeProcess:
        returncode = 1

        async def communicate(self):
            return b"", b"stop here"

    async def fake_exec(*cmd, **kwargs):
        captured["cmd"] = list(cmd)
        return FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    builder = FirmwareBuilder()
    result = asyncio.run(builder.build(FirmwareConfig(
        mcu_family="stm32", mcu_part="STM32F407VG", test_suites=["basic"],
        build_dir=tmp_path / "build")))

    assert not result.success
    cmd = captured["cmd"]
    assert cmd.index(str(builder.app_dir)) < cmd.index("--")
