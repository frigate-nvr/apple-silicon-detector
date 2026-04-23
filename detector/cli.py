import argparse
import sys
import time
from datetime import datetime

from detector import __version__, service_manager
from detector.service_manager import homify


def main():
    parser = argparse.ArgumentParser(description="Frigate Detector CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # start
    parser_start = subparsers.add_parser("start", help="Start detector (foreground by default)")
    parser_start.add_argument(
        "-d",
        "--daemon",
        action="store_true",
        help="Start as background service (via launchd)",
    )
    parser_start.add_argument(
        "--endpoint",
        nargs="+",
        default=service_manager.DEFAULT_ENDPOINTS,
        help="ZMQ endpoint(s) to bind to (default: tcp + ipc)",
    )
    parser_start.add_argument(
        "--providers",
        nargs="+",
        default=service_manager.DEFAULT_PROVIDERS,
        help="ONNX execution providers",
    )
    parser_start.add_argument("--model", default="AUTO", help="Model path or AUTO")
    parser_start.add_argument("-v", "--verbose", action="store_true", help="Debug logging")

    # stop
    subparsers.add_parser("stop", help="Stop the background service")

    # restart
    subparsers.add_parser("restart", help="Restart the background service")

    # status
    subparsers.add_parser("status", help="Show running state, PID, endpoint, model")

    # logs
    parser_logs = subparsers.add_parser("logs", help="Tail the live log")
    parser_logs.add_argument("--err", action="store_true", help="Tail the error log")
    parser_logs.add_argument(
        "-n",
        "--lines",
        type=int,
        default=10,
        help="Number of lines to show before following",
    )
    parser_logs.add_argument("-f", "--follow", action="store_true", help="Follow log output")
    parser_logs.add_argument("--debug", action="store_true", help="Show verbose GUI execution logs")

    # models
    subparsers.add_parser("models", help="List installed models with sizes")

    # startup
    parser_startup = subparsers.add_parser("startup", help="Enable or disable auto-start on login")
    parser_startup.add_argument("state", choices=["enable", "disable"], help="Set auto-start state (enable or disable)")

    parser_startup.add_argument(
        "--endpoint",
        nargs="+",
        help="ZMQ endpoint(s) to bind to (headless only)",
    )
    parser_startup.add_argument(
        "--providers",
        nargs="+",
        help="ONNX execution providers (headless only)",
    )
    parser_startup.add_argument("--model", help="Model path or AUTO (headless only)")

    # debug
    parser_debug = subparsers.add_parser("debug", help="Enable or disable verbose debug logging")
    parser_debug.add_argument(
        "state", choices=["enable", "disable"], help="Set debug logging state (enable or disable)"
    )

    # config
    subparsers.add_parser("config", help="Show current paths and settings")

    # install-cli
    parser_install = subparsers.add_parser("install-cli", help="Install detector CLI to your PATH")
    parser_install.add_argument("--system", action="store_true", help="Install system-wide (requires sudo)")

    # uninstall-cli
    subparsers.add_parser("uninstall-cli", help="Remove detector CLI from your PATH")

    # version
    subparsers.add_parser("version", help="Show version")

    args = parser.parse_args()

    if args.command == "version":
        print(f"Frigate Detector v{__version__} (Apple Silicon)")

    elif args.command == "start":
        if any(ep.startswith("ipc://") for ep in args.endpoint):
            service_manager.ensure_ipc_dir()

        if args.daemon:
            service_manager.start_service(args.endpoint, args.providers)

            print("Starting background service...", end="", flush=True)
            for _ in range(12):
                if service_manager.is_running():
                    break
                time.sleep(0.5)
                print(".", end="", flush=True)
            print()

            status = service_manager.get_status()
            if status.running:
                print(f"✓ Running (PID {status.pid})")
                for ep in status.endpoints:
                    print(f"  Endpoint: {ep}")
            else:
                print("Failed to start daemon. Check logs using 'detector logs --err'")
                sys.exit(1)
        else:
            # Foreground mode
            print("Starting detector in foreground...")
            print("Press Ctrl+C to stop.")
            try:
                # If bundled, we need to run the client main directly
                from detector import zmq_onnx_client

                zmq_onnx_client.main(args.endpoint, args.model, args.providers, args.verbose)
            except KeyboardInterrupt:
                print("\nStopping...")
                sys.exit(0)
            except Exception as e:
                print(f"\nError: {e}")
                sys.exit(1)

    elif args.command == "stop":
        print("Stopping...", end="", flush=True)
        service_manager.stop_service()
        print("\r✓ Stopped    ")

    elif args.command == "restart":
        print("Restarting...", end="", flush=True)
        service_manager.restart_service()
        print("\r✓ Restarted  ")

    elif args.command == "status":
        status = service_manager.get_status()
        if status.running:
            print(f"● {status.status_label}")
            print(f"  PID:       {status.pid}")
            for ep in status.endpoints:
                print(f"  Endpoint:  {ep}")
            print(f"  Uptime:    {status.uptime or 'Unknown'}")
        else:
            if status.status_label != "Stopped":
                print(f"○ {status.status_label}")
            else:
                print("○ Frigate Detector is not running")

        print(f"  Startup:   {status.startup_enabled}")
        print(f"  Debug:     {'enabled' if status.debug else 'disabled'}")
        print(f"  Log:       {status.log_path}")

    elif args.command == "models":
        models_dir = service_manager.get_models_dir()
        print(f"Models directory: {homify(models_dir)}\n")
        print(f"{'NAME':<30} {'SIZE':<10} {'MODIFIED'}")
        count = 0
        if models_dir.exists():
            for f in models_dir.glob("*.onnx"):
                size_mb = f.stat().st_size / (1024 * 1024)
                mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                print(f"{f.name:<30} {size_mb:.1f} MB   {mtime}")
                count += 1
        print(f"\n{count} model(s) installed")

    elif args.command == "startup":
        if args.state == "enable":
            service_manager.set_run_at_load(
                True,
                endpoints=args.endpoint,
                providers=args.providers,
                model=args.model or "AUTO",
            )
            print("✓ Startup auto-launch enabled (headless)")
        else:
            service_manager.set_run_at_load(False)
            print("✓ Startup auto-launch disabled")

    elif args.command == "debug":
        enabled = args.state == "enable"
        service_manager.save_config(debug=enabled)
        print(f"✓ Debug logging {'enabled' if enabled else 'disabled'}")
        if enabled:
            print("  Note: This affects the background service and CLI. Check logs with 'detector logs'")

    elif args.command == "config":
        status = service_manager.get_status()
        print(f"Frigate Detector v{__version__} configuration:\n")
        print("PATHS")
        print(f"  App bundle:    {sys.executable if getattr(sys, 'frozen', False) else 'running from source'}")
        print(f"  Models:        {status.models_dir}")
        print(f"  Logs:          {homify(service_manager.LOG_DIR)}")
        print(f"  Service plist: {homify(service_manager.PLIST_PATH)}\n")

        print("DEFAULTS")
        print(f"  Endpoints:     {', '.join(service_manager.DEFAULT_ENDPOINTS)}")
        print(f"  Providers:     {' '.join(service_manager.DEFAULT_PROVIDERS)}")
        print("  Model:         AUTO")
        print(f"  Debug Mode:    {'ON' if status.debug else 'OFF'}\n")

        print("CLI")
        print(f"  detector:      {homify(service_manager.CLI_INSTALL_DIR / 'detector')}")

    elif args.command == "install-cli":
        success, msg = service_manager.install_cli(force_system=args.system)
        print(msg)
        if not success:
            sys.exit(1)

    elif args.command == "uninstall-cli":
        success, msg = service_manager.uninstall_cli()
        print(msg)
        if not success:
            sys.exit(1)

    elif args.command == "logs":
        out_path, err_path = service_manager.get_log_paths()
        target = err_path if args.err else out_path

        if not target.exists():
            print(f"Log file does not exist: {homify(target)}")
            return

        with open(target) as f:
            lines = f.readlines()
            for line in lines[-args.lines :]:
                print(line, end="")

            if args.follow:
                try:
                    while True:
                        line = f.readline()
                        if not line:
                            time.sleep(0.1)
                            continue
                        print(line, end="", flush=True)
                except KeyboardInterrupt:
                    pass


if __name__ == "__main__":
    main()
