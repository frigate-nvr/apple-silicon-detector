import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

SERVICE_LABEL = "com.frigate.apple-silicon-detector"
PLIST_FILENAME = f"{SERVICE_LABEL}.plist"
LAUNCH_AGENTS = Path.home() / "Library" / "LaunchAgents"
PLIST_PATH = LAUNCH_AGENTS / PLIST_FILENAME
ASSOCIATED_BUNDLE_ID = "com.frigate.apple-silicon-detector.app"
LOG_DIR = Path.home() / "Library" / "Logs" / "FrigateDetector"
MODELS_DIR = Path.home() / "Library" / "Application Support" / "FrigateDetector" / "models"
# ── Shared constants (keep in sync with macos/swift/Sources/FrigateDetector/Constants.swift) ──
DEFAULT_TCP_ENDPOINT = "tcp://0.0.0.0:5555"
DEFAULT_IPC_ENDPOINT = "ipc:///tmp/frigate-detector/zmq_detector"
DEFAULT_ENDPOINTS = [DEFAULT_TCP_ENDPOINT, DEFAULT_IPC_ENDPOINT]
DEFAULT_PROVIDERS = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
CLI_INSTALL_DIR = Path.home() / ".local" / "bin"
CLI_FALLBACK_DIR = Path("/usr/local/bin")
STATE_DIR = Path("/tmp/frigate-detector")
CONFIG_PATH = MODELS_DIR.parent / "config.json"


@dataclass
class ServiceStatus:
    running: bool
    pid: int | None
    uptime: str | None
    endpoints: list[str]
    startup_enabled: bool
    log_path: str
    err_log_path: str
    models_dir: str
    status_label: str
    debug: bool


def is_bundled() -> bool:
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def get_project_dir() -> Path:
    if is_bundled():
        return Path(sys._MEIPASS).parent.parent.parent
    return Path(__file__).parent.parent.resolve()


def get_detector_service_bin() -> Path:
    if is_bundled():
        # If we are already running the real binary (inside detector-bin)
        if "detector-bin" in str(sys.executable):
            return Path(sys.executable)

        # If we are running the wrapper, find the real binary
        real_bin = Path(sys.executable).parent / "detector-bin" / "detector"
        if real_bin.exists():
            return real_bin
        return Path(sys.executable)

    # Non-bundled fallback: try to find the python executable in the current environment
    # This is usually .venv/bin/python or sys.executable itself
    base_python = Path(sys.executable)
    if base_python.name.startswith("python"):
        return base_python

    return base_python.parent / "python"


def get_cli_bin() -> Path:
    return get_detector_service_bin()


def get_log_paths() -> tuple[Path, Path]:
    return LOG_DIR / "detector.log", LOG_DIR / "detector.err.log"


def get_models_dir() -> Path:
    return MODELS_DIR


def get_app_path() -> Path | None:
    """Return the path to the FrigateDetector.app bundle if bundled."""
    if not is_bundled():
        return None

    # Robustly find the .app bundle by searching upwards.
    # sys.executable might be in Contents/MacOS/ or Contents/MacOS/detector-bin/
    # resolve() handles symlinks to ensure we are looking at the real path.
    curr = Path(sys.executable).resolve()
    for _ in range(5):
        if curr.suffix == ".app":
            return curr
        if curr.parent == curr:
            break
        curr = curr.parent
    return None


def _get_ipc_dir() -> Path:
    """Return the stable IPC directory path."""
    return Path("/tmp/frigate-detector")


def ensure_ipc_dir():
    ipc_dir = _get_ipc_dir()
    ipc_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(str(ipc_dir), 0o755)  # World-readable/executable for cross-UID access
    except Exception:
        pass


def get_service_args(endpoints: list[str] | None = None, providers: list[str] | None = None) -> list[str]:
    """Return the base argument list for the detector service."""
    if endpoints is None:
        endpoints = DEFAULT_ENDPOINTS
    if providers is None:
        providers = DEFAULT_PROVIDERS

    args = ["--model", "AUTO"]
    if endpoints:
        args.append("--endpoint")
        args.extend(endpoints)
    if providers:
        args.append("--providers")
        args.extend(providers)
    return args


def load_config() -> dict:
    """Load the persistent configuration from config.json."""
    if not CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text())
    except Exception:
        return {}


def save_config(
    headless: bool | None = None,
    endpoints: list[str] | None = None,
    providers: list[str] | None = None,
    model: str | None = None,
    debug: bool | None = None,
):
    """Save the persistent configuration to config.json, merging with existing settings."""
    config = load_config()

    if headless is not None:
        config["headless"] = headless
    if endpoints is not None:
        config["endpoints"] = endpoints
    if providers is not None:
        config["providers"] = providers
    if model is not None:
        config["model"] = model
    if debug is not None:
        config["debug"] = debug

    # Ensure defaults for fresh config
    if "endpoints" not in config:
        config["endpoints"] = DEFAULT_ENDPOINTS
    if "providers" not in config:
        config["providers"] = DEFAULT_PROVIDERS
    if "model" not in config:
        config["model"] = "AUTO"
    if "headless" not in config:
        config["headless"] = False
    if "debug" not in config:
        config["debug"] = False

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=4))

    # Signal running detector to reload config
    try:
        for pid in get_pids():
            try:
                os.kill(pid, signal.SIGHUP)
            except OSError:
                pass
    except Exception:
        pass


def generate_plist(run_at_load: bool = False) -> str:
    """Generate a static launchd plist content that always calls the startup launcher."""
    service_bin = get_detector_service_bin()
    run_at_load_xml = "<true/>" if run_at_load else "<false/>"

    # Always invoke the binary with --service --startup flags.
    # The entry_point.py will then decide based on config.json whether to launch the GUI or Service.
    if is_bundled():
        exec_args = [
            f"<string>{service_bin}</string>",
            "<string>--service</string>",
            "<string>--startup</string>",
        ]
    else:
        # For non-bundled, we use the module execution
        exec_args = [
            f"<string>{service_bin}</string>",
            "<string>-m</string>",
            "<string>detector</string>",
            "<string>--service</string>",
            "<string>--startup</string>",
        ]

    args_xml = "\n        ".join(exec_args)

    # Static log paths for the launcher/daemon
    out_path, err_path = get_log_paths()

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{SERVICE_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        {args_xml}
    </array>
    <key>RunAtLoad</key>
    {run_at_load_xml}
    <key>StandardOutPath</key>
    <string>{out_path}</string>
    <key>StandardErrorPath</key>
    <string>{err_path}</string>
    <key>AssociatedBundleIdentifiers</key>
    <array>
        <string>{ASSOCIATED_BUNDLE_ID}</string>
    </array>
</dict>
</plist>"""


def is_installed() -> bool:
    return PLIST_PATH.exists()


def install_service(run_at_load: bool = False):
    """Install the static launchd plist agent."""
    LAUNCH_AGENTS.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    get_models_dir().mkdir(parents=True, exist_ok=True)

    # Always generate the same static plist
    content = generate_plist(run_at_load=run_at_load)
    PLIST_PATH.write_text(content)


def spawn_detector_process(endpoints: list[str] | None = None, providers: list[str] | None = None) -> subprocess.Popen:
    """Launch the detector service as a child process of the current app."""
    service_bin = get_detector_service_bin()
    base_args = get_service_args(endpoints, providers)

    # CRITICAL: Always include --service when spawning the daemon
    cmd = [str(service_bin)]
    if is_bundled():
        cmd.append("--service")
    else:
        cmd.extend(["-m", "detector.zmq_onnx_client"])

    cmd.extend(base_args)

    # Log and config/models redirection
    out_path, _ = get_log_paths()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    get_models_dir().parent.mkdir(parents=True, exist_ok=True)

    log_fd = os.open(str(out_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND)

    # Environment: Force unbuffered output so logs appear in real-time
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd,
        stdout=log_fd,
        stderr=log_fd,
        cwd=get_models_dir().parent,
        env=env,
        start_new_session=True,
    )
    os.close(log_fd)  # Safe: Popen has dup'd the fd
    return proc


def uninstall_service():
    """Remove the unified service launch agent."""
    if PLIST_PATH.exists():
        subprocess.run(["launchctl", "unload", str(PLIST_PATH)], capture_output=True, timeout=5)
        PLIST_PATH.unlink()


def _set_state_flag(state: str, active: bool):
    """Manage temporary state flags in /tmp."""
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        flag_path = STATE_DIR / f"detector.{state}.flag"
        if active:
            flag_path.touch()
        elif flag_path.exists():
            flag_path.unlink()
    except Exception:
        pass


def _get_active_state() -> str | None:
    """Return the currently active transitional state based on flags."""
    try:
        if (STATE_DIR / "detector.restarting.flag").exists():
            return "Restarting..."
        if (STATE_DIR / "detector.stopping.flag").exists():
            return "Stopping..."
        if (STATE_DIR / "detector.starting.flag").exists():
            return "Starting..."
    except Exception:
        pass
    return None


def start_service(endpoints: list[str] | None = None, providers: list[str] | None = None):
    """Start the service via launchd (headless) or manual spawn (GUI fallback)."""
    if is_running():
        return

    # Use manual background spawn (always).
    # We avoid 'launchctl start' because the current plist includes --startup,
    # which triggers a GUI redirect in entry_point.py, causing a loop.
    # spawn_detector_process gives us direct engine execution.
    _set_state_flag("starting", True)
    try:
        spawn_detector_process(endpoints, providers)

        # Brief poll
        for _ in range(10):
            if is_running():
                break
            time.sleep(0.5)
    finally:
        _set_state_flag("starting", False)


def stop_service():
    """Unconditionally stop all detector processes regardless of mode."""
    _set_state_flag("stopping", True)
    try:
        # 1. Cleanup launchd registration (very important to stop KeepAlive if it was set)
        if PLIST_PATH.exists():
            subprocess.run(["launchctl", "stop", SERVICE_LABEL], capture_output=True, timeout=5)
            subprocess.run(["launchctl", "unload", str(PLIST_PATH)], capture_output=True, timeout=5)

        # 2. Kill the process by PID if it exists (safer than pkill -f)
        pids = get_pids()
        if pids:
            # Broadcast SIGTERM to prompt graceful python shutdown flushes
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGTERM)
                except OSError:
                    pass

            # Give it 2s to clean up natively
            for _ in range(20):
                time.sleep(0.1)
                if not get_pids():
                    break
            else:
                # Force kill any lingering processes after timeout
                for pid in get_pids():
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except OSError:
                        pass
    finally:
        _set_state_flag("stopping", False)


def restart_service():
    _set_state_flag("restarting", True)
    try:
        stop_service()
        time.sleep(0.5)
        start_service()
    finally:
        _set_state_flag("restarting", False)


def is_running() -> bool:
    try:
        # Use get_pids() to ensure consistent, hardened matching logic
        return len(get_pids()) > 0
    except Exception:
        return False


def get_pids() -> list[int]:
    pids = []
    try:
        # 1. Try launchctl list
        res = subprocess.run(["launchctl", "list", SERVICE_LABEL], capture_output=True, text=True, timeout=5)
        if res.returncode == 0:
            m = re.search(r'"PID" = (\d+);', res.stdout)
            if m:
                pids.append(int(m.group(1)))

        # 2. Check all "detector" processes and verify they have --service in their cmdline
        # On macOS pgrep -f can be unreliable with complex regexes.
        res = subprocess.run(["pgrep", "-f", "detector"], capture_output=True, text=True, timeout=5)
        if res.returncode == 0:
            for line in res.stdout.splitlines():
                if not line.strip():
                    continue
                pid_str = line.strip()

                # Use ps to get the full command line for this PID
                ps_res = subprocess.run(
                    ["ps", "-ww", "-p", pid_str, "-o", "command="], capture_output=True, text=True, timeout=5
                )
                if ps_res.returncode == 0:
                    cmdline = ps_res.stdout.strip()
                    # We match both --service (daemon) and detector.zmq_onnx_client (the actual engine)
                    is_bundled_match = "detector-bin" in cmdline

                    if ("--service" in cmdline or "detector.zmq_onnx_client" in cmdline) and (
                        is_bundled_match or not is_bundled()
                    ):
                        pids.append(int(pid_str))
    except Exception:
        pass

    # Return unique PIDs, excluding current process
    my_pid = os.getpid()
    return list({p for p in pids if p != my_pid})


def get_pid(pids: list[int] | None = None) -> int | None:
    pids = pids if pids is not None else get_pids()
    return pids[0] if pids else None


def _get_uptime_for_pid(pid: int | None) -> str | None:
    if not pid:
        return None
    try:
        res = subprocess.run(["ps", "-o", "etime=", "-p", str(pid)], capture_output=True, text=True, timeout=5)
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return None


def is_run_at_load() -> bool:
    """Check if the unified plist has RunAtLoad set to true."""
    if not PLIST_PATH.exists():
        return False
    try:
        content = PLIST_PATH.read_text()
        # Look for <key>RunAtLoad</key> followed by <true/>
        m = re.search(r"<key>RunAtLoad</key>\s*<(true|false)/>", content)
        return m is not None and m.group(1) == "true"
    except Exception:
        return False


def set_run_at_load(
    enabled: bool,
    headless: bool = False,
    endpoints: list[str] | None = None,
    providers: list[str] | None = None,
    model: str = "AUTO",
):
    """Toggle RunAtLoad and persist startup mode settings."""
    if enabled:
        # 1. Update/Write the persistent config
        save_config(headless=headless, endpoints=endpoints, providers=providers, model=model)

        # 2. Ensure the static plist is installed
        install_service(run_at_load=True)

        # 3. Notify launchd
        if PLIST_PATH.exists():
            subprocess.run(["launchctl", "unload", str(PLIST_PATH)], capture_output=True, timeout=5)
            subprocess.run(["launchctl", "load", "-w", str(PLIST_PATH)], capture_output=True, timeout=5)
    else:
        # Uninstall entirely
        uninstall_service()


def get_status() -> ServiceStatus:
    out_path, err_path = get_log_paths()
    pids = get_pids()
    running = len(pids) > 0
    pid = pids[0] if pids else None

    # Determine the current status label
    transitional = _get_active_state()
    if transitional:
        status_label = transitional
    else:
        status_label = "Running" if running else "Stopped"

    config = load_config()
    endpoints = config.get("endpoints", DEFAULT_ENDPOINTS)

    return ServiceStatus(
        running=running,
        pid=pid,
        uptime=_get_uptime_for_pid(pid),
        endpoints=endpoints,
        startup_enabled=is_run_at_load(),
        log_path=homify(out_path),
        err_log_path=homify(err_path),
        models_dir=homify(get_models_dir()),
        status_label=status_label,
        debug=config.get("debug", False),
    )


def _add_to_path(install_dir: Path) -> str | None:
    """Add install_dir to ~/.zshrc or ~/.bash_profile if not already there."""
    home = Path.home()
    profiles = [home / ".zshrc", home / ".bash_profile"]
    export_cmd = f'\nexport PATH="{install_dir}:$PATH"\n'

    updated = []
    for profile in profiles:
        try:
            # Create .zshrc if neither exist, since it's macOS default
            if profile.exists() or (profile.name == ".zshrc" and not any(p.exists() for p in profiles)):
                content = profile.read_text() if profile.exists() else ""
                if str(install_dir) not in content:
                    with profile.open("a") as f:
                        f.write(export_cmd)
                    updated.append(profile.name)
        except Exception:
            pass

    if updated:
        return f"\nAdded to PATH in {', '.join(updated)}. Please restart your terminal."
    return None


def install_cli(force_system: bool = False) -> tuple[bool, str]:
    cli_bin = get_cli_bin()
    if not cli_bin.exists():
        return False, f"CLI binary not found at {cli_bin}"

    if force_system:
        target_dir = CLI_FALLBACK_DIR
        target_link = target_dir / "detector"
        script = f"do shell script \\\"ln -sf '{cli_bin}' '{target_link}'\\\" with administrator privileges"
        res = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=5)
        if res.returncode == 0:
            return True, f"Installed completely to {target_link}"
        return False, f"Failed to install to {target_link}: {res.stderr}"

    # Try local install
    CLI_INSTALL_DIR.mkdir(parents=True, exist_ok=True)
    target_link = CLI_INSTALL_DIR / "detector"
    try:
        if target_link.exists() or target_link.is_symlink():
            target_link.unlink()
        target_link.symlink_to(cli_bin)

        # Check PATH
        path_env = os.environ.get("PATH", "")
        if str(CLI_INSTALL_DIR) not in path_env:
            added_msg = _add_to_path(CLI_INSTALL_DIR)
            if added_msg:
                return True, f"Installed successfully to {target_link}{added_msg}"
            return (
                True,
                f"Installed to {target_link}.\nMake sure {CLI_INSTALL_DIR} is in your PATH.",
            )
        return True, f"Installed successfully to {target_link}"
    except Exception as e:
        return False, f"Failed to install to {target_link}: {e}"


def is_cli_installed() -> bool:
    """Return True if a CLI symlink exists in either install location."""
    for path in [CLI_INSTALL_DIR / "detector", CLI_FALLBACK_DIR / "detector"]:
        if path.exists() or path.is_symlink():
            return True
    return False


def uninstall_cli() -> tuple[bool, str]:
    """Remove CLI symlinks from standard locations. Returns (success, message)."""
    local_link = CLI_INSTALL_DIR / "detector"
    system_link = CLI_FALLBACK_DIR / "detector"
    removed = []

    if local_link.is_symlink():
        try:
            local_link.unlink()
            removed.append(str(local_link))
        except Exception as e:
            return False, f"Failed to remove {local_link}: {e}"

    if system_link.is_symlink():
        try:
            res = subprocess.run(
                [
                    "osascript",
                    "-e",
                    f"do shell script \\\"rm '{system_link}'\\\" with administrator privileges",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode == 0:
                removed.append(str(system_link))
        except Exception as e:
            return False, f"Failed to remove {system_link}: {e}"

    if removed:
        return True, f"CLI removed from: {', '.join(removed)}"
    return False, "No CLI installation found"


def homify(path: Path | str) -> str:
    """Replace the user's home directory with ~."""
    try:
        if not path:
            return ""
        path = Path(path)
        home = Path.home()
        if path.is_absolute() and path.is_relative_to(home):
            return f"~/{path.relative_to(home)}"
    except Exception:
        pass
    return str(path)
