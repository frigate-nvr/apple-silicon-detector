# 2.0.0

### Added
- **Native macOS Menu Bar GUI**: Self-contained SwiftUI menu bar app with real-time status, endpoint management, model listing, and an Advanced settings menu (Open at Login, View Logs, Debug Logging, CLI Install/Uninstall).
- **Packaged macOS App**: Self-contained `FrigateDetector.app` bundled via PyInstaller (zero prerequisites), code-signed, and distributed as a DMG installer.
- **Unified CLI**: `detector` command with 11 subcommands: `start`, `stop`, `restart`, `status`, `logs`, `models`, `startup enable/disable`, `debug enable/disable`, `config`, `install-cli`, `uninstall-cli`, and `version`.
- **Service Manager**: Full `launchd` lifecycle management with PID-based process discovery, persistent `config.json`, transitional state flags (Starting/Stopping/Restarting), and SIGHUP-based live config reload.
- **Dual-mode startup**: Boot-time `--startup` flag reads `config.json` to intelligently launch either the GUI app or a headless detector service.
- **Multi-endpoint binding**: The detector now binds to both `tcp://0.0.0.0:5555` and `ipc:///tmp/frigate-detector/zmq_detector` simultaneously by default.
- **Structured exceptions**: Custom exception hierarchy (`DetectorError`, `ModelLoadError`, `TransportError`, `InferenceError`) for clear error propagation.
- **Signal handling**: Graceful shutdown via SIGTERM/SIGINT handlers with `_cleaning_up` guard to prevent recursive cleanup. IPC socket files are removed on exit.
- **uv integration**: Fast dependency and virtual environment management via `pyproject.toml` (Hatchling build backend) with `uv sync` / `uv run`.
- **CI/CD pipeline**: GitHub Actions workflow for ruff lint/format, pytest, and automated DMG build+upload on release.
- **Test suite**: Unit tests for `service_manager` and ZMQ smoke test script.
- **Headless CLI Install**: The menu bar app can symlink the CLI tool to `~/.local/bin/detector` via the Advanced menu.

### Changed
- Replaced the previous `run_mac.sh` terminal wrapper with the unified `detector` CLI and native Swift GUI.
- Default ZMQ endpoints changed from single `ipc:///tmp/cache/zmq_detector` to dual-bind `tcp://0.0.0.0:5555` + `ipc:///tmp/frigate-detector/zmq_detector`.
- Logs now reside in `~/Library/Logs/FrigateDetector/detector.log` with real-time unbuffered output (`PYTHONUNBUFFERED=1`, `write_through=True`, kernel `fsync`).
- Models now persist across app versions in `~/Library/Application Support/FrigateDetector/models/`.
- Moved project packaging from `requirements.txt` + `venv` to modern `pyproject.toml` format via Hatchling.
- Reduced ZMQ heartbeat timeout from 5s to 1s for faster signal responsiveness.
- Added TCP keepalive (`TCP_KEEPALIVE_IDLE=30s`) for faster dead-peer detection.
- Build pipeline rewritten: native Swift compilation + PyInstaller + codesign + DMG packaging (replaces the old shell-script-based `.app` wrapper).

### Removed
- Legacy `run_mac.sh` shell wrapper and `.command` file launcher.
- `requirements.txt` (replaced by `pyproject.toml`).
- `pydantic` and `opencv-contrib-python` dependencies (unused).

### Fixed
- Implemented graceful shutdown with SIGTERM/SIGINT signal handlers to prevent zombie socket bindings and duplicate startup instances.
- Added IPC socket file cleanup on exit to prevent `EADDRINUSE` errors on restart.
- Replaced AppleScript-based login items with silent `launchd` associated bundle identifiers to eliminate intrusive permission pop-ups.
- Resolved an issue where service logs were buffering infinitely and appearing empty at runtime.
- Added explicit ONNX session cleanup (`del self.session`) to release NPU/GPU resources on shutdown.

# 1.1.1
Update dependencies to support newer Python versions

# 1.1.0
Make .app launch more robust and open a terminal directly for more visibility

# 0.1.2
Improve app to run in headless mode so it shows up in ActivityMonitor

# 0.1.1
fix build script

# 0.1.0
Initial Release for the .app runner for apple silicon detector