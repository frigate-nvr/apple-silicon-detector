#!/usr/bin/env python3
"""
ZMQ TCP ONNX Runtime Client

This client connects to the ZMQ TCP proxy, accepts tensor inputs,
runs inference via ONNX Runtime, and returns detection results.

Protocol:
- Receives multipart messages: [header_json_bytes, tensor_bytes]
- Header contains shape and dtype information
- Runs ONNX inference on the tensor
- Returns results in the expected format: [20, 6] float32 array

Note: Timeouts are normal when Frigate has no motion to detect.
The server will continue running and waiting for requests.
"""

import json
import logging
import logging.config
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import zmq

from detector.exceptions import (
    InferenceError,
    ModelLoadError,
    TransportError,
)
from detector.model_util import post_process_dfine, post_process_rfdetr, post_process_yolo


def setup_logging(verbose=False, log_to_file=False):
    """Configure logging for the detector."""

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "console": {
                "format": "%(asctime)s [client] %(levelname)s - %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "console",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "level": "DEBUG" if verbose else "INFO",
            "handlers": ["console"],
        },
    }

    logging.config.dictConfig(config)


# Initial minimal logging before main setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ZmqOnnxClient:
    """
    ZMQ TCP client that runs ONNX inference on received tensors.
    """

    def __init__(
        self,
        endpoints: list[str] | None = None,
        model_path: str | None = "AUTO",
        providers: list[str] | None = None,
    ):
        """
        Initialize the ZMQ ONNX client.

        Args:
            endpoints: ZMQ endpoints to bind to (TCP and/or IPC)
            model_path: Path to ONNX model file or "AUTO" for automatic model management
            providers: ONNX Runtime execution providers
        """
        if endpoints is None:
            endpoints = ["tcp://0.0.0.0:5555", "ipc:///tmp/frigate-detector/zmq_detector"]
        self.endpoints = endpoints
        self.model_path = model_path
        self.current_model = None
        self.model_ready = False

        # Ensure logging is setup if it hasn't been already
        # (This is a safety measure for when the client is instantiated directly)
        if not logging.getLogger().handlers:
            setup_logging(log_to_file=False)

        from detector.service_manager import ensure_ipc_dir, get_models_dir

        self.models_dir = str(get_models_dir())

        # Initialize ZMQ context and socket
        self.context = None
        self.socket = None

        # Ensure IPC directory exists before ZMQ initialization
        if any(ep.startswith("ipc://") for ep in self.endpoints):
            ensure_ipc_dir()

        self._initialize_zmq()

        # Register signal handlers for graceful shutdown
        import signal

        try:
            signal.signal(signal.SIGTERM, self._handle_signal)
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGHUP, self._handle_sighup)
        except Exception as e:
            logger.debug(f"Could not register signal handlers: {e}")

        # Initialize ONNX Runtime session
        self.session = None
        if self.model_path != "AUTO":
            self.session = self._initialize_onnx_session(providers)

        self.zero_result = np.zeros((20, 6), dtype=np.float32)
        self._cleaning_up = False

        logger.info(f"ZMQ ONNX client initialized with endpoints: {endpoints}")
        if self.model_path != "AUTO":
            logger.info(f"ONNX model loaded from: {self.model_path}")
        else:
            logger.info("ZMQ ONNX client started in AUTO mode - waiting for model requests")

    def _initialize_zmq(self):
        """Initialize ZMQ context and socket with proper error handling."""
        try:
            # Clean up any existing resources
            self.cleanup()

            # Create new context
            self.context = zmq.Context()
            logger.debug("ZMQ context created successfully")

            # Create new socket
            self.socket = self.context.socket(zmq.REP)
            logger.debug("ZMQ REP socket created successfully")

            # Set socket options
            # Reduce timeout to 1s to make the process more responsive to signals
            self.socket.setsockopt(zmq.RCVTIMEO, 1000)
            self.socket.setsockopt(zmq.SNDTIMEO, 1000)
            self.socket.setsockopt(zmq.LINGER, 0)  # Don't wait for unsent messages on close
            # TCP keepalive: detect dead peers and free the port faster
            self.socket.setsockopt(zmq.TCP_KEEPALIVE, 1)
            self.socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 30)
            logger.debug("ZMQ socket options set successfully")

            logger.debug("ZMQ context and socket initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize ZMQ: {e}")
            self.cleanup()
            raise TransportError(f"Failed to initialize ZMQ: {e}") from e

    def _reset_socket(self):
        """Reset the socket when encountering state issues."""
        try:
            logger.info("Resetting ZMQ socket due to state issues")

            # Close existing socket
            if self.socket:
                self.socket.close()
                self.socket = None

            self.socket = self.context.socket(zmq.REP)
            self.socket.setsockopt(zmq.RCVTIMEO, 1000)
            self.socket.setsockopt(zmq.SNDTIMEO, 1000)
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.setsockopt(zmq.TCP_KEEPALIVE, 1)
            self.socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 30)

            # Rebind to all endpoints
            for ep in self.endpoints:
                self.socket.bind(ep)
            logger.info("Socket reset and rebound successfully")

        except Exception as e:
            logger.error(f"Failed to reset socket: {e}")
            raise TransportError(f"Failed to reset socket: {e}") from e

    def _create_onnx_session(
        self,
        model_path: str,
        providers: list[str] | None = None,
    ) -> ort.InferenceSession | None:
        """
        Create an ONNX Runtime session with CoreML optimizations.

        Args:
            model_path: Path to the ONNX model file
            providers: Execution providers (e.g., ['CoreMLExecutionProvider', 'CPUExecutionProvider'])
            session_options: Session options

        Returns:
            ONNX Runtime inference session or None if creation fails
        """
        try:
            cache_dir = os.path.join(self.models_dir, "cache")
            os.makedirs(cache_dir, exist_ok=True)

            if providers is None:
                providers = ["CoreMLExecutionProvider"]

            # Configure CoreML EP with optimizations
            provider_options = []
            if "CoreMLExecutionProvider" in providers:
                coreml_options = {
                    "ModelFormat": "MLProgram",  # Use MLProgram format for better performance
                    "MLComputeUnits": "ALL",  # Use all available compute units
                    "ModelCacheDirectory": cache_dir,
                }
                provider_options.append(("CoreMLExecutionProvider", coreml_options))

            # Add other providers without options
            for provider in providers:
                if provider != "CoreMLExecutionProvider":
                    provider_options.append((provider, {}))

            logger.info(f"Loading ONNX model with providers: {[p[0] for p in provider_options]}")
            session = ort.InferenceSession(model_path, providers=provider_options)

            # Log model input/output info
            input_info = session.get_inputs()[0]
            output_info = session.get_outputs()[0]
            logger.info(f"Model input: {input_info.name}, shape: {input_info.shape}, type: {input_info.type}")
            logger.info(f"Model output: {output_info.name}, shape: {output_info.shape}, type: {output_info.type}")

            return session

        except Exception as e:
            logger.error(f"Failed to create ONNX session: {e}")
            raise ModelLoadError(f"Failed to create ONNX session: {e}") from e

    def _initialize_onnx_session(
        self,
        providers: list[str] | None = None,
    ) -> ort.InferenceSession | None:
        """
        Initialize ONNX Runtime session with CoreML optimizations.

        Args:
            providers: Execution providers (e.g., ['CoreMLExecutionProvider', 'CPUExecutionProvider'])
            session_options: Session options

        Returns:
            ONNX Runtime inference session or None if no model path
        """
        if not self.model_path:
            logger.warning("No model path provided, ONNX inference will be skipped")
            return None

        return self._create_onnx_session(self.model_path, providers)

    def _check_model_exists(self, model_name: str) -> bool:
        """
        Check if a model exists in the models directory.

        Args:
            model_name: Name of the model file to check

        Returns:
            True if model exists, False otherwise
        """
        model_path = os.path.join(self.models_dir, model_name)
        return os.path.exists(model_path)

    def _load_model(
        self,
        model_name: str,
        providers: list[str] | None = None,
    ) -> bool:
        """
        Load a model from the models directory with CoreML optimizations.

        Args:
            model_name: Name of the model file to load
            providers: ONNX Runtime execution providers
            session_options: ONNX Runtime session options

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            model_path = os.path.join(self.models_dir, model_name)
            logger.info(f"Loading model from: {model_path}")

            self.session = self._create_onnx_session(model_path, providers)
            if self.session is None:
                return False

            self.current_model = model_name
            self.model_ready = True

            # Small delay to ensure model is fully ready
            time.sleep(0.1)
            logger.info("Model ready for inference")

            return True

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise ModelLoadError(f"Failed to load model {model_name}: {e}") from e

    def _save_model(self, model_name: str, model_data: bytes) -> bool:
        """
        Save model data to the models directory.

        Args:
            model_name: Name of the model file to save
            model_data: Binary model data

        Returns:
            True if model saved successfully, False otherwise
        """
        try:
            # Ensure models directory exists
            os.makedirs(self.models_dir, exist_ok=True)

            model_path = os.path.join(self.models_dir, model_name)
            logger.info(f"Saving model to: {model_path}")

            with open(model_path, "wb") as f:
                f.write(model_data)

            logger.info(f"Model saved successfully: {model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to save model {model_name}: {e}")
            return False

    def _decode_request(self, frames: list[bytes]) -> tuple[np.ndarray | None, dict]:
        """
        Decode the incoming request frames.

        Args:
            frames: List of message frames

        Returns:
            Tuple of (tensor, header_dict)
        """
        try:
            if len(frames) < 1:
                raise ValueError(f"Expected at least 1 frame, got {len(frames)}")

            # Parse header
            header_bytes = frames[0]
            header = json.loads(header_bytes.decode("utf-8"))

            if "model_request" in header:
                return None, header

            if "model_data" in header:
                if len(frames) < 2:
                    raise ValueError(f"Model data request expected 2 frames, got {len(frames)}")
                return None, header

            if len(frames) < 2:
                raise ValueError(f"Tensor request expected 2 frames, got {len(frames)}")

            tensor_bytes = frames[1]
            shape = tuple(header.get("shape", []))
            dtype_str = header.get("dtype", "uint8")

            dtype = np.dtype(dtype_str)
            tensor = np.frombuffer(tensor_bytes, dtype=dtype).reshape(shape)
            return tensor, header

        except Exception as e:
            logger.error(f"Failed to decode request: {e}")
            raise

    def _run_inference(self, tensor: np.ndarray, header: dict) -> np.ndarray:
        """
        Run ONNX inference on the input tensor.

        Args:
            tensor: Input tensor
            header: Request header containing metadata (e.g., shape, layout)

        Returns:
            Detection results as numpy array

        Raises:
            RuntimeError: If no ONNX session is available or inference fails
        """
        if self.session is None:
            logger.warning("No ONNX session available, returning zero results")
            return self.zero_result

        try:
            # Prepare input for ONNX Runtime
            # Determine input spatial size (W, H) from header/shape/layout
            model_type = header.get("model_type")
            width, height = self._extract_input_hw(header)

            if model_type == "dfine":
                # DFine model requires both images and orig_target_sizes inputs
                input_data = {
                    "images": tensor.astype(np.float32),
                    "orig_target_sizes": np.array([[height, width]], dtype=np.int64),
                }
            else:
                # Other models use single input
                input_name = self.session.get_inputs()[0].name
                input_data = {input_name: tensor}

            # Run inference
            t_start = 0.0
            if logger.isEnabledFor(logging.DEBUG):
                t_start = time.perf_counter()

            outputs = self.session.run(None, input_data)

            t_after_onnx = 0.0
            if logger.isEnabledFor(logging.DEBUG):
                t_after_onnx = time.perf_counter()

            # Post-process based on model type
            if model_type == "yolo-generic" or model_type == "yologeneric":
                result = post_process_yolo(outputs, width, height)
            elif model_type == "dfine":
                result = post_process_dfine(outputs, width, height)
            elif model_type == "rfdetr":
                result = post_process_rfdetr(outputs)
            else:
                logger.error(f"Unknown model_type '{model_type}' — returning zero detections")
                return self.zero_result

            if logger.isEnabledFor(logging.DEBUG):
                t_after_post = time.perf_counter()
                onnx_ms = (t_after_onnx - t_start) * 1000.0
                post_ms = (t_after_post - t_after_onnx) * 1000.0
                total_ms = (t_after_post - t_start) * 1000.0
                logger.debug(f"Inference timing: onnx={onnx_ms:.2f}ms, post={post_ms:.2f}ms, total={total_ms:.2f}ms")

            # Ensure float32 dtype
            result = result.astype(np.float32)

            return result

        except Exception as e:
            logger.error(f"ONNX inference failed: {e}")
            raise InferenceError(f"ONNX inference failed: {e}") from e

    def _extract_input_hw(self, header: dict) -> tuple[int, int]:
        """
        Extract (width, height) from the header and/or tensor shape, supporting
        NHWC/NCHW as well as 3D/4D inputs. Falls back to 320x320 if unknown.

        Preference order:
        1) Explicit header keys: width/height
        2) Use provided layout to interpret shape
        3) Heuristics on shape
        """
        try:
            if "width" in header and "height" in header:
                return int(header["width"]), int(header["height"])

            shape = tuple(header.get("shape", []))
            layout = header.get("layout") or header.get("order")

            if layout and shape:
                layout = str(layout).upper()
                if len(shape) == 4:
                    if layout == "NCHW":
                        return int(shape[3]), int(shape[2])
                    if layout == "NHWC":
                        return int(shape[2]), int(shape[1])
                if len(shape) == 3:
                    if layout == "CHW":
                        return int(shape[2]), int(shape[1])
                    if layout == "HWC":
                        return int(shape[1]), int(shape[0])

            if shape:
                if len(shape) == 4:
                    _, d1, d2, d3 = shape
                    if d1 in (1, 3):
                        return int(d3), int(d2)
                    if d3 in (1, 3):
                        return int(d2), int(d1)
                    return int(d2), int(d1)
                if len(shape) == 3:
                    d0, d1, d2 = shape
                    if d0 in (1, 3):
                        return int(d2), int(d1)
                    if d2 in (1, 3):
                        return int(d1), int(d0)
                    return int(d1), int(d0)
                if len(shape) == 2:
                    h, w = shape
                    return int(w), int(h)
        except Exception as e:
            logger.debug(f"Failed to extract input size from header: {e}")

        logger.debug("Falling back to default input size (320x320)")
        return 320, 320

    def _build_response(self, result: np.ndarray) -> list[bytes]:
        """
        Build the response message.

        Args:
            result: Detection results

        Returns:
            List of response frames
        """
        try:
            # Build header
            header = {
                "shape": list(result.shape),
                "dtype": str(result.dtype.name),
                "timestamp": time.time(),
            }
            header_bytes = json.dumps(header).encode("utf-8")

            # Convert result to bytes
            result_bytes = result.tobytes(order="C")

            return [header_bytes, result_bytes]

        except Exception as e:
            logger.error(f"Failed to build response: {e}")
            # Return zero result as fallback
            header = {
                "shape": [20, 6],
                "dtype": "float32",
                "error": "Failed to build response",
            }
            header_bytes = json.dumps(header).encode("utf-8")
            result_bytes = self.zero_result.tobytes(order="C")
            return [header_bytes, result_bytes]

    def _handle_model_request(self, header: dict) -> list[bytes]:
        """
        Handle model availability request.

        Args:
            header: Request header containing model information

        Returns:
            Response message indicating model availability
        """
        model_name = header.get("model_name")

        if not model_name:
            logger.error("Model request missing model_name")
            return self._build_error_response("Model request missing model_name")

        logger.info(f"Model availability request for: {model_name}")

        if self._check_model_exists(model_name):
            logger.info(f"Model {model_name} exists locally")

            if (
                self.current_model == model_name
                and self.session is not None
                and self.model_ready
            ):
                logger.info(f"Model {model_name} already loaded, reusing session")
                response_header = {
                    "model_available": True,
                    "model_loaded": True,
                    "model_name": model_name,
                    "message": f"Model {model_name} already loaded",
                }
                return [json.dumps(response_header).encode("utf-8")]

            if self._load_model(model_name):
                response_header = {
                    "model_available": True,
                    "model_loaded": True,
                    "model_name": model_name,
                    "message": f"Model {model_name} loaded successfully",
                }
            else:
                response_header = {
                    "model_available": True,
                    "model_loaded": False,
                    "model_name": model_name,
                    "message": f"Model {model_name} exists but failed to load",
                }
        else:
            logger.info(f"Model {model_name} not found, requesting transfer")
            response_header = {
                "model_available": False,
                "model_name": model_name,
                "message": f"Model {model_name} not found, please send model data",
            }

        return [json.dumps(response_header).encode("utf-8")]

    def _handle_model_data(self, header: dict, model_data: bytes) -> list[bytes]:
        """
        Handle model data transfer.

        Args:
            header: Request header containing model information
            model_data: Binary model data

        Returns:
            Response message indicating save success/failure
        """
        model_name = header.get("model_name")

        if not model_name:
            logger.error("Model data missing model_name")
            return self._build_error_response("Model data missing model_name")

        logger.info(f"Received model data for: {model_name}")

        if self._save_model(model_name, model_data):
            # Try to load the model
            if self._load_model(model_name):
                response_header = {
                    "model_saved": True,
                    "model_loaded": True,
                    "model_name": model_name,
                    "message": f"Model {model_name} saved and loaded successfully",
                }
            else:
                response_header = {
                    "model_saved": True,
                    "model_loaded": False,
                    "model_name": model_name,
                    "message": f"Model {model_name} saved but failed to load",
                }
        else:
            response_header = {
                "model_saved": False,
                "model_loaded": False,
                "model_name": model_name,
                "message": f"Failed to save model {model_name}",
            }

        return [json.dumps(response_header).encode("utf-8")]

    def _build_error_response(self, error_msg: str) -> list[bytes]:
        """Build an error response message."""
        error_header = {"error": error_msg}
        return [json.dumps(error_header).encode("utf-8")]

    def _handle_sighup(self, sig, frame):
        """Handle SIGHUP to reload configuration dynamically."""
        logger.info("Received SIGHUP, reloading configuration...")
        try:
            from detector import service_manager

            config = service_manager.load_config()
            debug_enabled = bool(config.get("debug"))

            new_level = logging.DEBUG if debug_enabled else logging.INFO
            logging.getLogger().setLevel(new_level)
        except Exception as e:
            logger.error(f"Failed to reload config on SIGHUP: {e}")

    def _handle_signal(self, sig, frame):
        """Handle termination signals."""
        sig_name = "SIGTERM" if sig == signal.SIGTERM else "SIGINT"
        msg = f"{sig_name} received: shutting down gracefully"
        logger.info(msg)
        try:
            self.cleanup()
        except Exception as e:
            logger.error(f"Cleanup failed during signal handling: {e}")
        finally:
            logger.info("Process exiting now")
            try:
                sys.stdout.flush()
                sys.stderr.flush()
                import os

                os.fsync(sys.stdout.fileno())
            except (OSError, ValueError, AttributeError):
                pass

            # Hard exit to ensure we don't hang
            os._exit(0)

    def start_server(self):
        """
        Start the ZMQ server and listen for requests.
        """
        try:
            # Bind socket to all endpoints
            for ep in self.endpoints:
                logger.info(f"Attempting to bind to endpoint: {ep}")
                self.socket.bind(ep)
                logger.info(f"ZMQ server successfully bound to {ep}")
            logger.info("Detector is ready to accept model requests and inference requests")

            # Note: Signal handlers are now registered in __init__

            while True:
                try:
                    frames = self.socket.recv_multipart()
                    tensor, header = self._decode_request(frames)

                    if "model_request" in header:
                        # Model availability check (1 frame) - only during initialization
                        response = self._handle_model_request(header)
                        self.socket.send_multipart(response)
                    elif "model_data" in header and len(frames) >= 2:
                        # Model data transfer (2 frames) - only during initialization
                        model_data = frames[1]
                        response = self._handle_model_data(header, model_data)
                        self.socket.send_multipart(response)
                    elif tensor is not None:
                        # Regular inference request (2 frames) - always handle this
                        if self.model_ready and self.session is not None:
                            result = self._run_inference(tensor, header)
                        else:
                            result = self.zero_result
                            if not self.model_ready:
                                logger.debug("Model not ready, returning zero detections")

                        response = self._build_response(result)
                        self.socket.send_multipart(response)
                    else:
                        # Unknown request type - send zero detections instead of error
                        logger.warning("Unknown request type, sending zero detections")
                        result = self.zero_result
                        response = self._build_response(result)
                        self.socket.send_multipart(response)

                except zmq.ZMQError as e:
                    # EAGAIN (Resource temporarily unavailable) is expected during idle periods (1s timeout)
                    if e.errno == zmq.EAGAIN:
                        logger.debug("ZMQ heartbeat: Waiting for Frigate request...")
                        continue

                    # Handle other specific ZMQ errors
                    error_msg = str(e)
                    if "Operation cannot be accomplished in current state" in error_msg:
                        logger.info("Socket state issue, resetting socket...")
                        try:
                            self._reset_socket()
                            continue
                        except Exception as reset_error:
                            logger.error(f"Failed to reset socket: {reset_error}")
                            break
                    else:
                        # Send error response for other ZMQ errors
                        logger.error(f"ZMQ error: {e}")
                        break

                except Exception as e:
                    logger.error(f"Unexpected error: {e}")
                    self._send_error_response(str(e))

        except KeyboardInterrupt:
            logger.info("Shutting down server...")
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            self.cleanup()

    def _send_error_response(self, error_msg: str):
        """Send an error response to the client."""
        try:
            error_header = {"shape": [20, 6], "dtype": "float32", "error": error_msg}
            error_response = [
                json.dumps(error_header).encode("utf-8"),
                self.zero_result.tobytes(order="C"),
            ]
            self.socket.send_multipart(error_response)
        except Exception as send_error:
            logger.error(f"Failed to send error response: {send_error}")

    def cleanup(self):
        """Perform comprehensive cleanup of all resources."""
        if getattr(self, "_cleaning_up", False):
            return
        self._cleaning_up = True

        logger.info("Starting cleanup sequence...")

        try:
            # Remove IPC sockets from filesystem
            for ep in self.endpoints:
                if ep.startswith("ipc://"):
                    socket_path = ep.replace("ipc://", "")
                    try:
                        Path(socket_path).unlink(missing_ok=True)
                        logger.debug(f"Removed IPC socket: {socket_path}")
                    except Exception as e:
                        logger.debug(f"Failed to remove IPC socket {socket_path}: {e}")

            if getattr(self, "socket", None):
                try:
                    self.socket.close(linger=0)
                except Exception:
                    pass
                self.socket = None
            if getattr(self, "context", None):
                try:
                    self.context.term()
                except Exception:
                    pass
                self.context = None

            if getattr(self, "session", None):
                try:
                    # Deleting the session object triggers the release of
                    # hardware resources (NPU/GPU) in ONNX Runtime.
                    del self.session
                except Exception:
                    pass
                self.session = None

            # Force hardware-level flush of all outputs to ensure logs reach the disk
            try:
                for handler in logging.getLogger().handlers:
                    handler.flush()
                import os
                import sys

                sys.stdout.flush()
                sys.stderr.flush()

                # Command the macOS kernel to commit data to physical storage
                try:
                    os.fsync(sys.stdout.fileno())
                except (OSError, ValueError):
                    pass
                try:
                    os.fsync(sys.stderr.fileno())
                except (OSError, ValueError):
                    pass

                # Brief sleep to allow kernel to finish the operation
                import time

                time.sleep(0.1)
            except Exception:
                pass

            logger.info("Cleanup completed")
            sys.stdout.flush()
            sys.stderr.flush()

        except Exception as e:
            try:
                logger.error(f"Cleanup error: {e}")
            except Exception:
                pass
        finally:
            self._cleaning_up = False


def main(endpoints=None, model_path=None, providers=None, verbose=False):
    """Main function to run the ZMQ ONNX client."""
    import sys

    # Ensure streams are unbuffered as early as possible
    try:
        sys.stdout.reconfigure(write_through=True)
        sys.stderr.reconfigure(write_through=True)
    except (AttributeError, ValueError):
        pass

    from detector import service_manager

    config = service_manager.load_config()

    if endpoints is None:
        import argparse

        parser = argparse.ArgumentParser(description="ZMQ TCP ONNX Runtime Client")
        parser.add_argument(
            "--endpoint",
            nargs="+",
            default=["tcp://0.0.0.0:5555", "ipc:///tmp/frigate-detector/zmq_detector"],
            help="ZMQ endpoint(s) to bind to (default: tcp + ipc)",
        )
        parser.add_argument(
            "--model",
            default="AUTO",
            help="Path to ONNX model file or AUTO for automatic model management",
        )
        parser.add_argument(
            "--providers",
            nargs="+",
            default=["CoreMLExecutionProvider"],
            help="ONNX Runtime execution providers",
        )
        parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

        args = parser.parse_args()
        endpoints = args.endpoint
        model_path = args.model
        providers = args.providers
        verbose = args.verbose

    # Override verbose with persistent config if not explicitly set via CLI
    # Hard-cast to bool to avoid "truthy" strings (e.g. "off") from enabling debug
    if not verbose and bool(config.get("debug")):
        verbose = True

    # Configure robust logging to stdout, parent process captures it to file
    setup_logging(verbose=verbose, log_to_file=False)

    # Refresh logger after setup_logging
    global logger
    logger = logging.getLogger(__name__)

    # Create and start client
    client = ZmqOnnxClient(endpoints=endpoints, model_path=model_path, providers=providers)

    try:
        client.start_server()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        client.cleanup()


if __name__ == "__main__":
    main()
