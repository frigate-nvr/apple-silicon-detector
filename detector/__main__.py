import os
import sys

# Add the project root to sys.path if needed
# This is usually handled by PyInstaller, but good for local debugging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import subprocess

from detector import cli, service_manager, zmq_onnx_client


def main():
    """
    Unified entry point for the Apple Silicon Detector.
    Acts as the service if --service is passed, otherwise acts as the CLI.
    """
    # Robust handling of the --service flag
    is_service = False
    if "--service" in sys.argv:
        sys.argv.remove("--service")
        is_service = True

    is_startup = False
    if "--startup" in sys.argv:
        sys.argv.remove("--startup")
        is_startup = True

    if is_startup:
        # Load persistent config to decide between GUI and Headless Service
        config = service_manager.load_config()
        headless = config.get("headless", False)

        if not headless:
            app_path = service_manager.get_app_path()
            if app_path:
                # Launch the native macOS app bundle.
                # The GUI app will then start its own detector process.
                print(f"Startup: Launching GUI app at {app_path}")
                subprocess.run(["open", str(app_path)])
                return
            else:
                print("Startup: GUI requested but app bundle not found. Falling back to headless.")

        # Headless mode: Start the detector loop directly with saved settings
        print("Startup: Starting detector service in headless mode")
        zmq_onnx_client.main(
            endpoints=config.get("endpoints"), model_path=config.get("model", "AUTO"), providers=config.get("providers")
        )
    elif is_service:
        # Run the service logic normally
        zmq_onnx_client.main()
    else:
        # Run the CLI logic
        cli.main()


if __name__ == "__main__":
    main()
