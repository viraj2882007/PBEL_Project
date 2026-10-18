"""Regression tests for faults found during the full-project review."""

import asyncio
import re
from pathlib import Path

import pytest
import yaml

from mcu_diagnostic_agent.avrdude_support import parse_device_signature
from mcu_diagnostic_agent.firmware.builder import (
    FirmwareBuilder,
    FirmwareConfig,
    expand_suites,
)
from mcu_diagnostic_agent.orchestrator.agent import (
    DiagnosticAgent,
    DiagnosticConfig,
    DiagnosticState,
)
from mcu_diagnostic_agent.probe.base import (
    FlashError,
    ProbeError,
    ProbeInterfaceType,
)
from mcu_diagnostic_agent.probe.factory import ProbeFactory
from mcu_diagnostic_agent.probe.openocd_backend import OpenOCDBackend
from mcu_diagnostic_agent.probe.target import ALL_TARGETS

ROOT = Path(__file__).resolve().parents[1]


# --- AVRDUDE signature parsing -------------------------------------------------------

@pytest.mark.parametrize("text", [
    "avrdude: Device signature = 0x1e950f",
    "avrdude: device signature = 0x1E950F (probably m328p)",
    "Device signature = 1e 95 0f",
])
def test_signature_parser_accepts_real_avrdude_formats(text):
    assert parse_device_signature(text) == "1E 95 0F"


def test_signature_parser_rejects_overlong_packed_value():
    assert parse_device_signature("Device signature = 0x1e950f1") is None


# --- Firmware suite handling ---------------------------------------------------------

def test_aliases_expand_to_firmware_suites():
    assert expand_suites(["basic", "gpio", "analog", "communication"]) == [
        "basic", "gpio", "adc", "dac", "uart", "spi", "i2c",
    ]


def test_expand_suites_dedupes_and_normalizes():
    assert expand_suites([" Analog ", "adc", "BASIC"]) == ["adc", "dac", "basic"]


def test_unknown_suite_is_rejected_instead_of_building_empty_firmware():
    with pytest.raises(ValueError, match="advanced"):
        expand_suites(["basic", "advanced"])


def test_build_reports_unknown_suite_without_running_west(tmp_path):
    builder = FirmwareBuilder()
    config = FirmwareConfig("stm32", "STM32F407VG", test_suites=["nope"])
    result = asyncio.run(builder.build(config))
    assert not result.success
    assert "nope" in result.error


def test_every_firmware_suite_name_exists_in_main_c():
    main_c = (ROOT / "zephyr_app" / "src" / "main.c").read_text()
    from mcu_diagnostic_agent.firmware.builder import FIRMWARE_SUITES
    for suite in FIRMWARE_SUITES:
        assert '{"%s",' % suite in main_c


# --- Board names must agree everywhere ----------------------------------------------

def test_board_mapping_is_consistent_between_builder_yaml_and_target_db():
    families = yaml.safe_load((ROOT / "config" / "mcu_families.yaml").read_text())["families"]
    for part, target in ALL_TARGETS.items():
        from_builder = FirmwareBuilder.BOARD_MAP[target.family][part]
        from_yaml = families[target.family]["zephyr_board_mapping"][part]
        assert from_builder == from_yaml == target.zephyr_board, part


def test_yaml_has_no_duplicate_mapping_keys():
    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader, node, deep=False):
        keys = [loader.construct_object(k, deep=deep) for k, _ in node.value]
        assert len(keys) == len(set(keys)), "duplicate YAML keys: {}".format(
            sorted({k for k in keys if keys.count(k) > 1}))
        return yaml.SafeLoader.construct_mapping(loader, node, deep)

    UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)
    for name in ("mcu_families.yaml", "peripheral_profiles.yaml"):
        yaml.load((ROOT / "config" / name).read_text(), Loader=UniqueKeyLoader)


def test_zephyr_board_conf_files_are_valid_kconfig_fragments():
    """Zephyr auto-loads boards/<board>.conf; C syntax there breaks the build."""
    for conf in (ROOT / "zephyr_app" / "boards").glob("*.conf"):
        for number, line in enumerate(conf.read_text().splitlines(), 1):
            stripped = line.strip()
            assert stripped == "" or stripped.startswith("#") or re.match(
                r"^CONFIG_\w+=", stripped), "{}:{}: {!r}".format(conf.name, number, line)


# --- Probe detection / OpenOCD backend -----------------------------------------------

def test_detect_probes_runs_without_hardware():
    assert isinstance(ProbeFactory().detect_probes(), list)


def _openocd(family="stm32", probe="stlink"):
    backend = OpenOCDBackend(probe_id=probe, target_family=family)
    backend.connected = True
    return backend


def test_openocd_config_uses_valid_source_commands():
    config = OpenOCDBackend(probe_id="stlink", target_family="stm32")._build_config(
        "adapter speed 1000", "STM32F407VG")
    assert "source [find interface/stlink.cfg]" in config
    assert "transport select hla_swd" in config
    assert "target/stm32f4x.cfg" in config
    for line in config.splitlines():
        assert not re.match(r"^interface/", line)


def test_openocd_config_esp_prog_and_picoprobe_use_existing_interface_files():
    esp = OpenOCDBackend(probe_id="esp-prog", interface=ProbeInterfaceType.JTAG,
                         target_family="esp32")._build_config()
    assert "interface/ftdi/esp32_devkitj_v1.cfg" in esp
    pico = OpenOCDBackend(probe_id="picoprobe", target_family="rp2040")._build_config()
    assert "interface/cmsis-dap.cfg" in pico and "picoprobe.cfg" not in pico


def test_openocd_unknown_family_does_not_fall_back_to_stm32():
    with pytest.raises(ProbeError):
        OpenOCDBackend(probe_id="stlink", target_family="arduino")._build_config()


def test_openocd_read_memory_uses_mdb_not_the_write_command(monkeypatch):
    backend = _openocd()
    sent = []

    def fake(cmd, timeout=30.0):
        sent.append(cmd)
        return "0x20000000: 01 02 03 04\n0x20000004: 05 06"

    monkeypatch.setattr(backend, "_send_command", fake)
    assert backend.read_memory(0x20000000, 6) == bytes([1, 2, 3, 4, 5, 6])
    assert sent == ["mdb 0x20000000 6"]


def test_openocd_short_read_raises_instead_of_returning_zeros(monkeypatch):
    backend = _openocd()
    monkeypatch.setattr(backend, "_send_command", lambda cmd, timeout=30.0: "0x0: 01")
    with pytest.raises(ProbeError):
        backend.read_memory(0, 4)


def test_openocd_flash_requires_confirmation_from_openocd(monkeypatch, tmp_path):
    image = tmp_path / "fw.bin"
    image.write_bytes(b"\x00" * 16)
    backend = _openocd()

    monkeypatch.setattr(backend, "_send_command", lambda cmd, timeout=30.0: "")
    result = backend.flash_binary(image)
    assert not result.success  # silent output must not count as a successful write

    calls = []

    def ok(cmd, timeout=30.0):
        calls.append(cmd)
        return "wrote 16 bytes from file" if "write_image" in cmd else "verified 16 bytes"

    monkeypatch.setattr(backend, "_send_command", ok)
    result = backend.flash_binary(image)
    assert result.success and result.verify_passed
    assert "0x8000000" in calls[0]


def test_openocd_failed_verify_is_a_failure(monkeypatch, tmp_path):
    image = tmp_path / "fw.bin"
    image.write_bytes(b"\x00" * 16)
    backend = _openocd()
    monkeypatch.setattr(
        backend, "_send_command",
        lambda cmd, timeout=30.0: "wrote 16 bytes" if "write_image" in cmd else "checksum mismatch")
    result = backend.flash_binary(image)
    assert not result.success and not result.verify_passed


def test_openocd_does_not_guess_flash_base_for_other_families(tmp_path):
    image = tmp_path / "fw.bin"
    image.write_bytes(b"\x00")
    backend = _openocd(family="esp32", probe="esp-prog")
    result = backend.flash_binary(image)
    assert not result.success and "flash base" in result.error
    with pytest.raises(FlashError):
        backend._flash_base(None)
    assert backend._flash_base(0) == 0  # an explicit address of 0 is honoured


def test_openocd_rp2040_default_flash_base_is_xip_region():
    assert OpenOCDBackend(target_family="rp2040")._flash_base(None) == 0x10000000


def test_openocd_register_parse(monkeypatch):
    backend = _openocd()
    monkeypatch.setattr(
        backend, "_send_command", lambda cmd, timeout=30.0: "pc (/32): 0x08000199")
    assert backend.read_core_register("pc") == 0x08000199


# --- Agent ----------------------------------------------------------------------------

def test_agent_error_state_survives_cleanup(tmp_path):
    agent = DiagnosticAgent(DiagnosticConfig(mcu_family="stm32", output_dir=tmp_path))
    agent.session.state = DiagnosticState.ERROR
    asyncio.run(agent._cleanup())
    assert agent.session.state == DiagnosticState.ERROR


def test_agent_passes_explicit_serial_port_to_runner(tmp_path, monkeypatch):
    captured = {}

    class FakeRunner:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def run(self):
            return []

    monkeypatch.setattr(
        "mcu_diagnostic_agent.orchestrator.agent.TestRunner", FakeRunner)
    agent = DiagnosticAgent(DiagnosticConfig(
        mcu_family="stm32", output_dir=tmp_path, serial_port="COM7"))
    asyncio.run(agent._run_tests())
    assert captured["serial_port"] == "COM7"
