"""
PyOCD backend implementation for probe interface.
"""

import logging
import time
from pathlib import Path
from typing import Optional, List

from .base import (
    ProbeInterface, ProbeInterfaceType, ProbeCapabilities,
    ProbeError, ConnectionError, FlashError, EraseError, ResetError,
    TargetInfo, FlashResult, MemoryRegion, FlashAlgorithm
)

logger = logging.getLogger(__name__)


class PyOCDBackend(ProbeInterface):
    """PyOCD-based debug probe backend."""

    def __init__(self, probe_id: Optional[str] = None,
                 interface: ProbeInterfaceType = ProbeInterfaceType.SWD,
                 target_override: Optional[str] = None,
                 freq_khz: int = 1000):
        super().__init__(probe_id, interface)
        self.target_override = target_override
        self.freq_khz = freq_khz
        self._session = None
        self._board = None
        self._target = None
        self._flash = None
        self._memory_map = None
        self._algorithms = None

    # Generic probe *types* used elsewhere in the project (OpenOCD interface selection and
    # the factory defaults). These are not pyOCD unique IDs and must not filter probes.
    PROBE_TYPE_NAMES = {
        "stlink", "jlink", "cmsis-dap", "esp-prog", "picoprobe", "blackmagic",
        "ftdi", "uart",
    }

    def _get_connect_helper(self):
        """Lazy import pyOCD to avoid hard dependency."""
        try:
            from pyocd.core.helpers import ConnectHelper
            return ConnectHelper
        except ImportError as e:
            raise ProbeError(f"PyOCD not installed: {e}. Install with: pip install pyocd")

    def _unique_id(self) -> Optional[str]:
        """Return the probe unique ID filter, or None when probe_id is only a probe type."""
        if self.probe_id and self.probe_id.lower() not in self.PROBE_TYPE_NAMES:
            return self.probe_id
        return None

    def connect(self, target_override: Optional[str] = None,
                freq_khz: Optional[int] = None) -> TargetInfo:
        """Connect to target using PyOCD."""
        if self.connected:
            return self.target_info

        ConnectHelper = self._get_connect_helper()
        effective_khz = freq_khz or self.freq_khz

        try:
            options = {}
            if effective_khz:
                options["frequency"] = int(effective_khz) * 1000
            override = target_override or self.target_override
            if override:
                options["target_override"] = override

            session = ConnectHelper.session_with_chosen_probe(
                unique_id=self._unique_id(),
                blocking=False,
                auto_open=False,
                options=options,
            )
            if session is None:
                raise ProbeError("No debug probe found")
            session.open()

            self._session = session
            self._board = session.board
            self._target = session.target
            self._flash = self._target.memory_map.get_boot_memory()

            self.target_info = self._get_target_info()
            self._memory_map = self._get_memory_map()
            self._algorithms = self._get_flash_algorithms()
            self.connected = True

            logger.info(f"Connected to {self.target_info.part_number} via PyOCD")
            return self.target_info

        except Exception as e:
            self.disconnect()
            raise ConnectionError(f"Failed to connect via PyOCD: {e}")

    def _get_target_info(self) -> TargetInfo:
        """Extract target information from PyOCD session."""
        target = self._target
        flash_bytes = sum(r.length for r in target.memory_map if r.is_flash)
        ram_bytes = sum(r.length for r in target.memory_map if r.is_ram)
        core_type = getattr(target, "core_type", None)
        cpu_name = getattr(core_type, "name", None) or str(core_type or "Cortex-M")
        return TargetInfo(
            family=cpu_name,
            part_number=getattr(target, "part_number", None) or getattr(
                self._session.board, "target_type", "Unknown"),
            revision=str(getattr(target, "revision", "Unknown")),
            flash_size_kb=flash_bytes // 1024,
            ram_size_kb=ram_bytes // 1024,
            cpu_type=cpu_name,
            unique_id=bytes(self._session.probe.unique_id, "ascii")
            if getattr(self._session, "probe", None) else b"",
        )

    def _get_memory_map(self) -> List[MemoryRegion]:
        """Get memory map from target."""
        regions = []
        for region in self._target.memory_map:
            regions.append(MemoryRegion(
                start=region.start,
                length=region.length,
                name=region.name,
                is_flash=region.is_flash,
                is_ram=region.is_ram,
                permissions=getattr(region, 'access', None) or getattr(region, 'permissions', 'rw')
            ))
        return regions

    def _get_flash_algorithms(self) -> List[FlashAlgorithm]:
        """Get flash algorithms from target (pyOCD exposes them per flash region)."""
        algos = []
        for region in self._target.memory_map:
            if not region.is_flash:
                continue
            algo = getattr(region, "algo", None)
            if not algo:
                continue
            algos.append(FlashAlgorithm(
                name=getattr(region, "name", "flash"),
                family=str(self.probe_id or "pyocd"),
                start_address=region.start,
                page_size=getattr(region, "page_size", 0) or 0,
                sector_size=getattr(region, "sector_size", 0) or 0,
                flash_size=region.length,
                ram_start=algo.get("load_address", 0) if isinstance(algo, dict) else 0,
                ram_size=algo.get("analyzer_size", 0) if isinstance(algo, dict) else 0,
            ))
        return algos

    def disconnect(self) -> None:
        """Disconnect from target."""
        if self._session:
            try:
                self._session.close()
            except Exception as e:
                logger.warning(f"Error closing PyOCD session: {e}")
            finally:
                self._session = None
                self._board = None
                self._target = None
                self._flash = None
                self.connected = False

    def reset(self, halt: bool = True, run: bool = False) -> None:
        """Reset target."""
        if not self.connected:
            raise ResetError("Not connected")
        try:
            if halt:
                self._target.reset_and_halt()
            else:
                self._target.reset()
                if run and self._target.is_halted():
                    self._target.resume()
        except Exception as e:
            raise ResetError(f"Reset failed: {e}")

    def halt(self) -> None:
        """Halt target."""
        if not self.connected:
            raise ProbeError("Not connected")
        self._target.halt()

    def resume(self) -> None:
        """Resume target."""
        if not self.connected:
            raise ProbeError("Not connected")
        self._target.resume()

    def read_memory(self, address: int, length: int) -> bytes:
        """Read memory."""
        if not self.connected:
            raise ProbeError("Not connected")
        return self._target.read_memory_block8(address, length)

    def write_memory(self, address: int, data: bytes) -> None:
        """Write memory."""
        if not self.connected:
            raise ProbeError("Not connected")
        self._target.write_memory_block8(address, data)

    def read_core_register(self, reg_name: str) -> int:
        """Read core register."""
        if not self.connected:
            raise ProbeError("Not connected")
        return self._target.read_core_register(reg_name)

    def write_core_register(self, reg_name: str, value: int) -> None:
        """Write core register."""
        if not self.connected:
            raise ProbeError("Not connected")
        self._target.write_core_register(reg_name, value)

    def flash_binary(self, file_path: Path, address: Optional[int] = None,
                     verify: bool = True, erase_first: bool = True) -> FlashResult:
        """Flash binary file."""
        if not self.connected:
            raise FlashError("Not connected")

        start_time = time.time()
        try:
            from pyocd.flash.file_programmer import FileProgrammer

            file_path = Path(file_path)
            size = file_path.stat().st_size
            programmer = FileProgrammer(
                self._session,
                chip_erase="chip" if erase_first else "sector",
                smart_flash=not erase_first,
                no_reset=True,
            )
            flash_addr = address
            if flash_addr is None and file_path.suffix.lower() == ".bin":
                boot = self._target.memory_map.get_boot_memory()
                flash_addr = boot.start if boot else None
            programmer.program(str(file_path), base_address=flash_addr)

            verify_passed = False
            if verify:
                with open(file_path, 'rb') as f:
                    verify_passed = self._verify_flash(flash_addr, f.read()) \
                        if flash_addr is not None else False
                if flash_addr is not None and not verify_passed:
                    return FlashResult(
                        success=False, bytes_written=size, verify_passed=False,
                        error="Flash verification failed",
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
        """Flash raw data."""
        if not self.connected:
            raise FlashError("Not connected")

        start_time = time.time()
        try:
            region = self._target.memory_map.get_region_by_address(address)
            if region is None or not region.is_flash:
                raise FlashError(f"Address 0x{address:08X} is not in a flash region")
            flash_builder = region.flash.get_flash_builder()
            flash_builder.add_data(address, data)
            flash_builder.program()

            verify_passed = self._verify_flash(address, data) if verify else False
            if verify and not verify_passed:
                return FlashResult(
                    success=False, bytes_written=len(data), verify_passed=False,
                    error="Flash verification failed",
                    duration_ms=(time.time() - start_time) * 1000,
                )
            return FlashResult(
                success=True,
                bytes_written=len(data),
                verify_passed=verify_passed,
                duration_ms=(time.time() - start_time) * 1000
            )

        except Exception as e:
            return FlashResult(
                success=False,
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000
            )

    def _verify_flash(self, address: int, data: bytes) -> bool:
        """Verify flashed data."""
        try:
            read_back = self.read_memory(address, len(data))
            return bytes(read_back) == bytes(data)
        except Exception:
            return False

    def erase_flash(self, address: Optional[int] = None, length: Optional[int] = None) -> None:
        """Erase flash (whole chip when no range is given)."""
        if not self.connected:
            raise EraseError("Not connected")
        try:
            from pyocd.flash.eraser import FlashEraser
            eraser = FlashEraser(self._session, FlashEraser.Mode.CHIP
                                 if address is None and length is None
                                 else FlashEraser.Mode.SECTOR)
            if address is None and length is None:
                eraser.erase()
            else:
                eraser.erase([f"0x{address:x}-0x{address + length:x}"])
        except Exception as e:
            raise EraseError(f"Erase failed: {e}")

    def erase_sector(self, address: int) -> None:
        """Erase single sector."""
        if not self.connected:
            raise EraseError("Not connected")
        try:
            from pyocd.flash.eraser import FlashEraser
            FlashEraser(self._session, FlashEraser.Mode.SECTOR).erase([f"0x{address:x}"])
        except Exception as e:
            raise EraseError(f"Sector erase failed: {e}")

    def get_memory_map(self) -> List[MemoryRegion]:
        """Get memory map."""
        return self._memory_map or []

    def get_flash_algorithms(self) -> List[FlashAlgorithm]:
        """Get flash algorithms."""
        return self._algorithms or []

    def set_breakpoint(self, address: int, hardware: bool = True) -> int:
        """Set breakpoint."""
        if not self.connected:
            raise ProbeError("Not connected")
        bp = self._target.set_breakpoint(address, hardware=hardware)
        return bp.id if hasattr(bp, 'id') else address

    def remove_breakpoint(self, bp_id: int) -> None:
        """Remove breakpoint."""
        if not self.connected:
            raise ProbeError("Not connected")
        self._target.remove_breakpoint(bp_id)

    def step(self) -> None:
        """Single step."""
        if not self.connected:
            raise ProbeError("Not connected")
        self._target.step()