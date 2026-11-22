"""Board telemetry (temperatures, power, fan, clock) from `tt-smi -s`, polled in a background thread.

tt-smi reads the firmware telemetry over PCIe and works while the engine holds the device (about 0.45 s per
snapshot). The server calls `BoardMonitor.latest()`, which never waits for tt-smi. Settings:
TSUGUMI_BOARD_POLL (seconds between snapshots, default 10; 0 turns the monitor off) and TT_SMI (the tt-smi
binary; default tt-smi on PATH, else ~/venv-tt-smi/bin/tt-smi, as in run_server.sh).
"""
import json
import os
import shutil
import subprocess
import threading
import time

from tsugumi.env import env
from tsugumi.log import get_logger

log = get_logger("board")


def find_tt_smi():
    p = os.environ.get("TT_SMI") or shutil.which("tt-smi") or os.path.expanduser("~/venv-tt-smi/bin/tt-smi")
    return p if os.path.isfile(p) and os.access(p, os.X_OK) else None


def _num(v):
    """tt-smi gives "0x.." hex strings (smbus_telem) or padded decimal strings (telemetry, limits)."""
    if v is None:
        return None
    try:
        if isinstance(v, str) and v.strip().startswith("0x"):
            return int(v, 16)
        f = float(v)
        return int(f) if f.is_integer() else f
    except (TypeError, ValueError):
        return None


def parse_snapshot(snap, index=0):
    """The fields worth watching from one device of a `tt-smi -s` snapshot, as plain numbers."""
    dev = snap["device_info"][index]
    t, s, lim, b = dev.get("telemetry", {}), dev.get("smbus_telem", {}), dev.get("limits", {}), dev.get("board_info", {})
    corr = sum(_num(s.get(f"GDDR_{i}_{i + 1}_CORR_ERRS")) or 0 for i in (0, 2, 4, 6))
    return dict(
        board=b.get("board_type"),
        asic_temp_c=_num(t.get("asic_temperature")),
        gddr_temp_max_c=_num(s.get("MAX_GDDR_TEMP")),
        throttle_temp_c=_num(s.get("THM_LIMIT_THROTTLE")) or _num(lim.get("therm_trip_l1_limit")),
        # the chip (TDP: core rail) power and its limit; the board's total power (TBP) is not reported,
        # only its limit (BOARD_POWER_LIMIT, 300 W on the P100A)
        chip_power_w=_num(t.get("power")),
        chip_power_limit_w=_num(lim.get("tdp_limit")),
        board_power_limit_w=_num(s.get("BOARD_POWER_LIMIT")),
        chip_current_a=_num(t.get("current")),
        chip_voltage_v=_num(t.get("voltage")),
        aiclk_mhz=_num(t.get("aiclk")),
        aiclk_max_mhz=_num(lim.get("asic_fmax")),
        fan_pct=_num(s.get("FAN_SPEED")),  # telemetry.fan_speed is % in tt-smi 5.x but rpm in 6.x
        fan_rpm=_num(s.get("FAN_RPM")),
        therm_trips=_num(s.get("THERM_TRIP_COUNT")),
        gddr_corr_errs=corr,
        gddr_uncorr_errs=_num(s.get("GDDR_UNCORR_ERRS")),
    )


class BoardMonitor:
    def __init__(self, interval=None, timeout=5.0):
        self.interval = float(env("BOARD_POLL", "10") if interval is None else interval)
        self.timeout = timeout
        self.tt_smi = find_tt_smi()
        self.data, self.error, self.at = None, None, None
        self._stop = threading.Event()
        self._thread = None
        self._last_error = None
        if self.interval <= 0:
            self.error = "off (TSUGUMI_BOARD_POLL=0)"
        elif self.tt_smi is None:
            self.error = "tt-smi not found (set TT_SMI)"
            log.warning("board telemetry off: tt-smi not found (set TT_SMI)")

    def start(self):
        if self.interval > 0 and self.tt_smi and self._thread is None:
            self._thread = threading.Thread(target=self._run, name="board-monitor", daemon=True)
            self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    def poll(self):
        try:
            r = subprocess.run([self.tt_smi, "-s"], capture_output=True, text=True, timeout=self.timeout,
                               stdin=subprocess.DEVNULL)
            if r.returncode != 0:
                raise RuntimeError(f"tt-smi exited with {r.returncode}: {(r.stderr or r.stdout).strip()[-200:]}")
            self.data, self.error = parse_snapshot(json.loads(r.stdout)), None
        except subprocess.TimeoutExpired:
            self.data, self.error = None, f"tt-smi timed out after {self.timeout:.0f}s"
        except Exception as e:  # bad JSON, missing fields, ...
            self.data, self.error = None, f"{type(e).__name__}: {e}"[:300]
        self.at = time.time()
        if self.error != self._last_error:  # log changes only, not the same failure every interval
            if self.error:
                log.warning(f"board telemetry: {self.error}")
            elif self._last_error:
                log.info("board telemetry: back")
            self._last_error = self.error

    def _run(self):
        while not self._stop.is_set():
            self.poll()
            self._stop.wait(self.interval)

    def latest(self):
        """The last snapshot (None until the first one), its age, and the last error."""
        return dict(data=self.data, age_s=None if self.at is None else round(time.time() - self.at, 1),
                    error=self.error)
