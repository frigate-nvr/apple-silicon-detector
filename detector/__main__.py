import os
import sys

# Add the project root to sys.path if needed
# This is usually handled by PyInstaller, but good for local debugging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detector import cli, zmq_onnx_client


def main():
    """
    Unified entry point for the Apple Silicon Detector.
    Acts as the service if --service is passed, otherwise acts as the CLI.
    """
    if "--service" in sys.argv:
        sys.argv.remove("--service")
        zmq_onnx_client.main()
    else:
        # Run the CLI logic
        cli.main()


if __name__ == "__main__":
    main()
