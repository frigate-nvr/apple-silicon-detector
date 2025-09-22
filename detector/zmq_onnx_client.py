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
import os
import time
from typing import List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import zmq

from model_util import post_process_yolo, post_process_rfdetr, post_process_dfine

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ZmqOnnxClient:
    """
    ZMQ TCP client that runs ONNX inference on received tensors.
    """

    def __init__(
        self,
        endpoint: str = "ipc:///tmp/cache/zmq_detector",
        model_path: Optional[str] = "AUTO",
        providers: Optional[List[str]] = None,
        session_options: Optional[ort.SessionOptions] = None,
    ):
        """
        Initialize the ZMQ ONNX client.

        Args:
            endpoint: ZMQ IPC endpoint to bind to
            model_path: Path to ONNX model file or "AUTO" for automatic model management
            providers: ONNX Runtime execution providers
            session_options: ONNX Runtime session options
        """
        self.endpoint = endpoint
        self.model_path = model_path
        self.current_model = None
        self.model_ready = False
        self.models_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "models"
        )

        # Initialize ZMQ context and socket
        self.context = None
        self.socket = None
        self._initialize_zmq()

        # Initialize ONNX Runtime session
        self.session = None
        if self.model_path != "AUTO":
            self.session = self._initialize_onnx_session(providers, session_options)

        # Preallocate zero result for error cases
        self.zero_result = np.zeros((20, 6), dtype=np.float32)

        logger.info(f"ZMQ ONNX client initialized with endpoint: {endpoint}")
        if self.model_path != "AUTO":
            logger.info(f"ONNX model loaded from: {self.model_path}")
        else:
            logger.info(
                "ZMQ ONNX client started in AUTO mode - waiting for model requests"
            )

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
            self.socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second receive timeout
            self.socket.setsockopt(zmq.SNDTIMEO, 5000)  # 5 second send timeout
            self.socket.setsockopt(
                zmq.LINGER, 0
            )  # Don't wait for unsent messages on close
            logger.debug("ZMQ socket options set successfully")

            logger.debug("ZMQ context and socket initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize ZMQ: {e}")
            self.cleanup()
            raise

    def _reset_socket(self):
        """Reset the socket when encountering state issues."""
        try:
            logger.info("Resetting ZMQ socket due to state issues")

            # Close existing socket
            if self.socket:
                self.socket.close()
                self.socket = None

            # Create new socket
            self.socket = self.context.socket(zmq.REP)
            self.socket.setsockopt(zmq.RCVTIMEO, 5000)
            self.socket.setsockopt(zmq.SNDTIMEO, 5000)
            self.socket.setsockopt(zmq.LINGER, 0)

            # Rebind to endpoint
            self.socket.bind(self.endpoint)
            logger.info("Socket reset and rebound successfully")

        except Exception as e:
            logger.error(f"Failed to reset socket: {e}")
            raise

    def _initialize_onnx_session(
        self,
        providers: Optional[List[str]] = None,
        session_options: Optional[ort.SessionOptions] = None,
    ) -> Optional[ort.InferenceSession]:
        """
        Initialize ONNX Runtime session.

        Args:
            providers: Execution providers (e.g., ['CoreMLExecutionProvider', 'CPUExecutionProvider'])
            session_options: Session options

        Returns:
            ONNX Runtime inference session or None if no model path
        """
        if not self.model_path:
            logger.warning("No model path provided, ONNX inference will be skipped")
            return None

        try:
            # Set default providers for Apple Silicon if none specified
            if providers is None:
                providers = ["CoreMLExecutionProvider"]

            logger.info(f"Loading ONNX model with providers: {providers}")
            session = ort.InferenceSession(self.model_path, providers=providers)

            # Log model input/output info
            input_info = session.get_inputs()[0]
            output_info = session.get_outputs()[0]
            logger.info(
                f"Model input: {input_info.name}, shape: {input_info.shape}, type: {input_info.type}"
            )
            logger.info(
                f"Model output: {output_info.name}, shape: {output_info.shape}, type: {output_info.type}"
            )

            return session

        except Exception as e:
            logger.error(f"Failed to initialize ONNX session: {e}")
            return None

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
        providers: Optional[List[str]] = None,
        session_options: Optional[ort.SessionOptions] = None,
    ) -> bool:
        """
        Load a model from the models directory.

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

            if providers is None:
                providers = ["CoreMLExecutionProvider"]

            self.session = ort.InferenceSession(model_path, providers=providers)
            self.current_model = model_name
            self.model_ready = True

            input_info = self.session.get_inputs()[0]
            output_info = self.session.get_outputs()[0]
            logger.info(
                f"Model loaded: {input_info.name}, shape: {input_info.shape}, type: {input_info.type}"
            )
            logger.info(
                f"Model output: {output_info.name}, shape: {output_info.shape}, type: {output_info.type}"
            )

            # Small delay to ensure model is fully ready
            time.sleep(0.1)
            logger.info("Model ready for inference")

            return True

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False

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

    def _decode_request(self, frames: List[bytes]) -> Tuple[np.ndarray, dict]:
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
                    raise ValueError(
                        f"Model data request expected 2 frames, got {len(frames)}"
                    )
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
            if logger.isEnabledFor(logging.DEBUG):
                t_start = time.perf_counter()

            outputs = self.session.run(None, input_data)

            if logger.isEnabledFor(logging.DEBUG):
                t_after_onnx = time.perf_counter()

            if model_type == "yolo-generic" or model_type == "yologeneric":
                result = post_process_yolo(outputs, width, height)
            elif model_type == "dfine":
                result = post_process_dfine(outputs, width, height)
            elif model_type == "rfdetr":
                result = post_process_rfdetr(outputs)

            if logger.isEnabledFor(logging.DEBUG):
                t_after_post = time.perf_counter()
                onnx_ms = (t_after_onnx - t_start) * 1000.0
                post_ms = (t_after_post - t_after_onnx) * 1000.0
                total_ms = (t_after_post - t_start) * 1000.0
                logger.debug(
                    f"Inference timing: onnx={onnx_ms:.2f}ms, post={post_ms:.2f}ms, total={total_ms:.2f}ms"
                )

            # Ensure float32 dtype
            result = result.astype(np.float32)

            return result

        except Exception as e:
            logger.error(f"ONNX inference failed: {e}")
            return self.zero_result

    def _extract_input_hw(self, header: dict) -> Tuple[int, int]:
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

    def _build_response(self, result: np.ndarray) -> List[bytes]:
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

    def _handle_model_request(self, header: dict) -> List[bytes]:
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
            # Try to load the model
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

    def _handle_model_data(self, header: dict, model_data: bytes) -> List[bytes]:
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

    def _build_error_response(self, error_msg: str) -> List[bytes]:
        """Build an error response message."""
        error_header = {"error": error_msg}
        return [json.dumps(error_header).encode("utf-8")]

    def start_server(self):
        """
        Start the ZMQ server and listen for requests.
        """
        try:
            # Log the exact endpoint being used
            logger.info(f"Attempting to bind to endpoint: {self.endpoint}")

            # Bind socket to endpoint
            self.socket.bind(self.endpoint)
            logger.info(f"ZMQ server successfully bound to {self.endpoint}")
            logger.info(
                "Detector is ready to accept model requests and inference requests"
            )

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
                                logger.debug(
                                    "Model not ready, returning zero detections"
                                )

                        response = self._build_response(result)
                        self.socket.send_multipart(response)
                    else:
                        # Unknown request type - send zero detections instead of error
                        logger.warning("Unknown request type, sending zero detections")
                        result = self.zero_result
                        response = self._build_response(result)
                        self.socket.send_multipart(response)

                except zmq.ZMQError as e:
                    error_msg = str(e)

                    # Handle specific ZMQ errors
                    if "Resource temporarily unavailable" in error_msg:
                        logger.debug(
                            "ZMQ heartbeat: Unable to communicate with Frigate"
                        )
                        continue
                    elif (
                        "Operation cannot be accomplished in current state" in error_msg
                    ):
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
                        self._send_error_response(str(e))

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
        """Clean up resources."""
        try:
            if self.socket:
                self.socket.close()
                self.socket = None
            if self.context:
                self.context.term()
                self.context = None
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


def main():
    """Main function to run the ZMQ ONNX client."""
    import argparse

    parser = argparse.ArgumentParser(description="ZMQ TCP ONNX Runtime Client")
    parser.add_argument(
        "--endpoint",
        default="tcp://*:5555",
        help="ZMQ TCP endpoint (default: tcp://*:5555)",
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
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create and start client
    client = ZmqOnnxClient(
        endpoint=args.endpoint, model_path=args.model, providers=args.providers
    )

    try:
        client.start_server()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        client.cleanup()


if __name__ == "__main__":
    main()
