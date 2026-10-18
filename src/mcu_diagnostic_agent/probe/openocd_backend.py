"""
OpenOCD backend implementation for probe interface.
Supports targets not well supported by PyOCD (ESP32, RP2040, etc.)
"""

import logging
import re
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional, List

from .base import (
    ProbeInterface, ProbeInterfaceType,
    ProbeError, ConnectionError, FlashError, EraseError, ResetError,
    TargetInfo, FlashResult, MemoryRegion, FlashAlgorithm
)

# Marker returned by the Tcl wrapper in _send_command when OpenOCD raised an error.
_ERROR_MARKER = "__OCD_ERROR__"

logger = logging.getLogger(__name__)


class OpenOCDBackend(ProbeInterface):
    """OpenOCD-based debug probe backend."""

    # OpenOCD config templates per target family.
    # Note: esp32c3/esp32h2 target files ship with Espressif's OpenOCD fork, not upstream.
    CONFIG_TEMPLATES = {
        "stm32": """
source [find interface/{interface}.cfg]
{transport_cmd}
source [find target/stm32{series}x.cfg]
{init_commands}
""",
        "esp32": """
source [find interface/{interface}.cfg]
transport select jtag
source [find target/esp32.cfg]
{init_commands}
""",
        "esp32s2": """
source [find interface/{interface}.cfg]
transport select jtag
source [find target/esp32s2.cfg]
{init_commands}
""",
        "esp32s3": """
source [find interface/{interface}.cfg]
transport select jtag
source [find target/esp32s3.cfg]
{init_commands}
""",
        "esp32c3": """
source [find interface/{interface}.cfg]
transport select jtag
source [find target/esp32c3.cfg]
{init_commands}
""",
        "esp32h2": """
source [find interface/{interface}.cfg]
transport select jtag
source [find target/esp32h2.cfg]
{init_commands}
""",
        "rp2040": """
source [find interface/{interface}.cfg]
transport select swd
source [find target/rp2040.cfg]
{init_commands}
""",
        "samd21": """
source [find interface/{interface}.cfg]
transport select swd
source [find target/at91samdXX.cfg]
{init_commands}
""",
        "samd51": """
source [find interface/{interface}.cfg]
transport select swd
source [find target/atsame5x.cfg]
{init_commands}
""",
    }

    INTERFACE_MAP = {
        "stlink": "stlink",
        "jlink": "jlink",
        "cmsis-dap": "cmsis-dap",
        "ftdi": "ftdi",
        "esp-prog": "ftdi/esp32_devkitj_v1",
        # Picoprobe firmware is CMSIS-DAP based; OpenOCD has no interface/picoprobe.cfg.
        "picoprobe": "cmsis-dap",
    }

    # Flash base addresses are family-specific; guessing the STM32 base for other families
    # would erase/write the wrong region, so unknown families must pass an explicit address.
    FLASH_BASE = {
        "stm32": 0x08000000,
        "rp2040": 0x10000000,
        "samd21": 0x00000000,
        "samd51": 0x00000000,
    }

    _STM32_SERIES_RE = re.compile(r"STM32(WB|WL|[FGHLU]\d)", re.IGNORECASE)

    TRANSPORT_MAP = {
        ProbeInterfaceType.SWD: "swd",
        ProbeInterfaceType.JTAG: "jtag",
    }

    def __init__(self, probe_id: Optional[str] = None,
                 interface: ProbeInterfaceType = ProbeInterfaceType.SWD,
                 target_family: str = "stm32",
                 target_config: Optional[str] = None,
                 freq_khz: int = 1000,
                 openocd_path: str = "openocd"):
        super().__init__(probe_id, interface)
        self.target_family = target_family.lower()
        self.target_config = target_config
        self.freq_khz = freq_khz
        self.openocd_path = openocd_path
        self._process: Optional[subprocess.Popen] = None
        self._config_file: Optional[Path] = None
        self._log_file: Optional[Path] = None
        self._telnet_port = 4444
        self._gdb_port = 3333
        self._tcl_port = 6666

    def _stm32_series(self, target_override: Optional[str]) -> str:
        """Pick the OpenOCD stm32<series>x.cfg series from a part number (default: f1)."""
        for candidate in (target_override, self.target_config):
            match = self._STM32_SERIES_RE.search(candidate or "")
            if match:
                return match.group(1).lower()
        logger.warning("STM32 series unknown; assuming 'f1'. Pass a part number to override.")
        return "f1"

    def _build_config(self, init_commands: str = "",
                      target_override: Optional[str] = None) -> str:
        """Build OpenOCD config file content."""
        interface_name = (
            self.INTERFACE_MAP.get(self.probe_id.lower(), "cmsis-dap")
            if self.probe_id else "cmsis-dap"
        )
        transport = self.TRANSPORT_MAP.get(self.interface, "swd")
        if interface_name == "stlink":
            # ST-Link uses the HLA transports; plain "swd"/"jtag" are rejected by OpenOCD 0.12.
            transport_cmd = f"transport select hla_{transport}"
        else:
            transport_cmd = f"transport select {transport}"

        template = self.CONFIG_TEMPLATES.get(self.target_family)
        if template is None:
            raise ConnectionError(
                f"No OpenOCD configuration for target family '{self.target_family}'. "
                f"Supported: {', '.join(sorted(self.CONFIG_TEMPLATES))}."
            )

        series = self._stm32_series(target_override) if self.target_family == "stm32" else ""

        return template.format(
            interface=interface_name,
            transport=transport,
            transport_cmd=transport_cmd,
            series=series,
            init_commands=init_commands or f"adapter speed {self.freq_khz}"
        )

    def _write_config(self, config_content: str) -> Path:
        """Write config to temporary file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False) as f:
            f.write(config_content)
            return Path(f.name)

    def _read_log_tail(self, limit: int = 2000) -> str:
        """Return the end of OpenOCD's log for error messages."""
        try:
            if self._log_file and self._log_file.exists():
                return self._log_file.read_text(errors="replace")[-limit:]
        except OSError:
            pass
        return ""

    def _start_openocd(self, config: str) -> None:
        """Start OpenOCD and wait until its Tcl RPC port accepts connections."""
        self._config_file = self._write_config(config)
        # OpenOCD logs to stderr continuously. Sending that to an unread PIPE eventually
        # fills the OS buffer and blocks OpenOCD mid-operation, so log to a file instead.
        log_handle = tempfile.NamedTemporaryFile(
            mode='w', suffix='.log', prefix='openocd_', delete=False)
        self._log_file = Path(log_handle.name)

        cmd = [
            self.openocd_path,
            "-f", str(self._config_file),
            "-c", f"telnet_port {self._telnet_port}",
            "-c", f"gdb_port {self._gdb_port}",
            "-c", f"tcl_port {self._tcl_port}",
        ]

        logger.debug(f"Starting OpenOCD: {' '.join(cmd)}")

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=log_handle,
                text=True
            )
        except OSError as exc:
            raise ConnectionError(f"Could not launch OpenOCD ({self.openocd_path}): {exc}")
        finally:
            log_handle.close()

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise ConnectionError(f"OpenOCD failed to start: {self._read_log_tail()}")
            try:
                with socket.create_connection(("127.0.0.1", self._tcl_port), timeout=0.5):
                    return
            except OSError:
                time.sleep(0.2)
        raise ConnectionError(
            f"OpenOCD did not open its command port in time: {self._read_log_tail()}")

    def _send_command(self, cmd: str, timeout: float = 30.0) -> str:
        """Run one OpenOCD command over the Tcl RPC port and return its output.

        The command runs inside ``capture`` so its printed output is returned, and inside
        ``catch`` so an OpenOCD error raises ProbeError instead of being mistaken for success.
        (telnetlib, used previously, was removed in Python 3.13 and never checked errors.)
        """
        script = (
            "if {[catch {capture {%s}} __r]} "
            "{format \"%s %%s\" $__r} else {set __r}" % (cmd, _ERROR_MARKER)
        )
        try:
            with socket.create_connection(("127.0.0.1", self._tcl_port), timeout=timeout) as sock:
                sock.settimeout(timeout)
                sock.sendall(script.encode() + b"\x1a")
                data = b""
                while not data.endswith(b"\x1a"):
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    data += chunk
        except OSError as e:
            raise ProbeError(f"OpenOCD command channel failed: {e}")

        output = data.rstrip(b"\x1a").decode(errors="replace").strip()
        if output.startswith(_ERROR_MARKER):
            raise ProbeError(f"OpenOCD command '{cmd}' failed: {output[len(_ERROR_MARKER):].strip()}")
        return output

    def _flash_base(self, address: Optional[int]) -> int:
        """Return the explicit flash address or the family default; never guess."""
        if address is not None:
            return address
        base = self.FLASH_BASE.get(self.target_family)
        if base is None:
            raise FlashError(
                f"No default flash base address for family '{self.target_family}'; "
                "pass an explicit address."
            )
        return base

    def connect(self, target_override: Optional[str] = None,
                freq_khz: Optional[int] = None) -> TargetInfo:
        """Connect to target using OpenOCD."""
        if self.connected:
            return self.target_info

        if freq_khz:
            self.freq_khz = freq_khz

        try:
            config = self._build_config(f"adapter speed {self.freq_khz}", target_override)
            self._start_openocd(config)

            # Initialize target (raises ProbeError if OpenOCD cannot reach it)
            self._send_command("reset init")
            time.sleep(0.5)

            # Get target info
            self.target_info = self._get_target_info(target_override)
            self.connected = True

            logger.info(f"Connected to {self.target_info.part_number} via OpenOCD")
            return self.target_info

        except Exception as e:
            self.disconnect()
            raise ConnectionError(f"Failed to connect via OpenOCD: {e}")

    def _get_target_info(self, target_override: Optional[str] = None) -> TargetInfo:
        """Get target info via OpenOCD commands."""
        # This is simplified - real implementation would parse OpenOCD output
        return TargetInfo(
            family=self.target_family.upper(),
            part_number=target_override or "Unknown",
            revision="Unknown",
            flash_size_kb=0,
            ram_size_kb=0,
            cpu_type="Cortex-M" if "cortex" in self.target_family else "Unknown",
        )

    def disconnect(self) -> None:
        """Disconnect and stop OpenOCD."""
        if self._process:
            try:
                self._send_command("shutdown")
            except Exception:
                pass
            try:
                self._process.terminate()
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
            finally:
                self._process = None

        for leftover in (self._config_file, self._log_file):
            if leftover and leftover.exists():
                try:
                    leftover.unlink()
                except Exception:
                    pass
        self._config_file = None
        self._log_file = None

        self.connected = False

    def reset(self, halt: bool = True, run: bool = False) -> None:
        """Reset target."""
        if not self.connected:
            raise ResetError("Not connected")
        cmd = "reset halt" if halt else "reset run"
        self._send_command(cmd)

    def halt(self) -> None:
        """Halt target."""
        if not self.connected:
            raise ProbeError("Not connected")
        self._send_command("halt")

    def resume(self) -> None:
        """Resume target."""
        if not self.connected:
            raise ProbeError("Not connected")
        self._send_command("resume")

    def read_memory(self, address: int, length: int) -> bytes:
        """Read memory via OpenOCD ``mdb`` (byte display)."""
        if not self.connected:
            raise ProbeError("Not connected")
        if length <= 0:
            return b""

        # NOTE: ``mwb`` is a memory *write*; reading requires ``mdb``.
        output = self._send_command(f"mdb {address:#x} {length}")
        values = []
        for line in output.splitlines():
            if ":" not in line:
                continue
            values.extend(
                int(token, 16) for token in line.split(":", 1)[1].split()
                if re.fullmatch(r"[0-9a-fA-F]{2}", token)
            )
        if len(values) != length:
            raise ProbeError(
                f"Memory read returned {len(values)} of {length} bytes at {address:#x}")
        return bytes(values)

    def write_memory(self, address: int, data: bytes) -> None:
        """Write memory via OpenOCD ``load_image`` of a temporary binary file."""
        if not self.connected:
            raise ProbeError("Not connected")
        if not data:
            return
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(data)
            temp_path = Path(f.name)
        try:
            self._send_command(f"load_image {{{temp_path.as_posix()}}} {address:#x} bin")
        finally:
            temp_path.unlink(missing_ok=True)

    def read_core_register(self, reg_name: str) -> int:
        """Read core register (target must be halted)."""
        if not self.connected:
            raise ProbeError("Not connected")
        output = self._send_command(f"reg {reg_name}")
        matches = re.findall(r"0x[0-9a-fA-F]+", output)
        if not matches:
            raise ProbeError(f"Could not parse register value for {reg_name}: {output!r}")
        return int(matches[-1], 16)

    def write_core_register(self, reg_name: str, value: int) -> None:
        """Write core register."""
        if not self.connected:
            raise ProbeError("Not connected")
        self._send_command(f"reg {reg_name} {value:#x}")

    def flash_binary(self, file_path: Path, address: Optional[int] = None,
                     verify: bool = True, erase_first: bool = True) -> FlashResult:
        """Flash a firmware image via OpenOCD, reporting only what OpenOCD confirmed."""
        if not self.connected:
            raise FlashError("Not connected")

        start_time = time.time()
        try:
            file_path = Path(file_path)
            size = file_path.stat().st_size
            is_raw = file_path.suffix.lower() in (".bin", "")
            path_arg = "{" + file_path.as_posix() + "}"
            if is_raw:
                flash_addr = self._flash_base(address)
                target_args = f"{path_arg} {flash_addr:#x} bin"
            else:
                # ELF/HEX images carry their own load addresses.
                target_args = path_arg

            erase_word = "erase " if erase_first else ""
            write_output = self._send_command(f"flash write_image {erase_word}{target_args}")
            if "wrote" not in write_output.lower():
                raise FlashError(f"OpenOCD did not confirm the flash write: {write_output!r}")

            verify_passed = False
            if verify:
                verify_output = self._send_command(f"verify_image {target_args}")
                verify_passed = "verified" in verify_output.lower()
                if not verify_passed:
                    return FlashResult(
                        success=False, bytes_written=size, verify_passed=False,
                        error=f"Flash verification failed: {verify_output!r}",
                        duration_ms=(time.time() - start_time) * 1000,
                    )

            return FlashResult(
                success=True,
                bytes_written=size,
                verify_passed=verify_passed,
                duration_ms=(time.time() - start_time) * 1000
            )

        except Exception as e:
            return FlashResult(
                success=False,
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000
            )

    def flash_data(self, address: int, data: bytes, verify: bool = True) -> FlashResult:
        """Flash raw data via temporary file."""
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(data)
            temp_path = Path(f.name)

        try:
            return self.flash_binary(temp_path, address, verify, erase_first=False)
        finally:
            temp_path.unlink(missing_ok=True)

    def erase_flash(self, address: Optional[int] = None, length: Optional[int] = None) -> None:
        """Erase flash (whole bank 0 when no range is given)."""
        if not self.connected:
            raise EraseError("Not connected")
        try:
            if address is not None and length is not None:
                self._send_command(f"flash erase_address {address:#x} {length}")
            else:
                self._send_command("flash erase_sector 0 0 last")
        except ProbeError as e:
            raise EraseError(f"Erase failed: {e}")

    def erase_sector(self, address: int) -> None:
        """Erase the sector containing ``address``."""
        if not self.connected:
            raise EraseError("Not connected")
        try:
            # erase_sector takes sector *indices*; erase_address takes a byte address.
            self._send_command(f"flash erase_address {address:#x} 1")
        except ProbeError as e:
            raise EraseError(f"Sector erase failed: {e}")

    def get_memory_map(self) -> List[MemoryRegion]:
        """Get memory map."""
        return []

    def get_flash_algorithms(self) -> List[FlashAlgorithm]:
        """Get flash algorithms."""
        return []

    def set_breakpoint(self, address: int, hardware: bool = True) -> int:
        """Set breakpoint via GDB."""
        return address

    def remove_breakpoint(self, bp_id: int) -> None:
        """Remove breakpoint."""
        pass

    def step(self) -> None:
        """Single step."""
        if not self.connected:
            raise ProbeError("Not connected")
        self._send_command("step")