#!/usr/bin/env python3
"""
ZMQ IPC ONNX Runtime Client

This client connects to the ZMQ IPC proxy, accepts tensor inputs,
runs inference via ONNX Runtime, and returns detection results.

Protocol:
- Receives multipart messages: [header_json_bytes, tensor_bytes]
- Header contains shape and dtype information
- Runs ONNX inference on the tensor
- Returns results in the expected format: [20, 6] float32 array
"""

import json
import logging
import sys
import time
from typing import Any, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import zmq

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ZmqOnnxClient:
    """
    ZMQ IPC client that runs ONNX inference on received tensors.
    """
    
    def __init__(
        self,
        endpoint: str = "ipc:///tmp/cache/zmq_detector",
        model_path: Optional[str] = None,
        providers: Optional[List[str]] = None,
        session_options: Optional[ort.SessionOptions] = None
    ):
        """
        Initialize the ZMQ ONNX client.
        
        Args:
            endpoint: ZMQ IPC endpoint to bind to
            model_path: Path to ONNX model file
            providers: ONNX Runtime execution providers
            session_options: ONNX Runtime session options
        """
        self.endpoint = endpoint
        self.model_path = model_path
        
        # Initialize ZMQ context and socket
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REP)
        
        # Initialize ONNX Runtime session
        self.session = self._initialize_onnx_session(providers, session_options)
        
        # Preallocate zero result for error cases
        self.zero_result = np.zeros((20, 6), dtype=np.float32)
        
        logger.info(f"ZMQ ONNX client initialized with endpoint: {endpoint}")
        if self.model_path:
            logger.info(f"ONNX model loaded from: {self.model_path}")
    
    def _initialize_onnx_session(
        self,
        providers: Optional[List[str]] = None,
        session_options: Optional[ort.SessionOptions] = None
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
                providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
            
            # Set default session options if none specified
            if session_options is None:
                session_options = ort.SessionOptions()
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                session_options.intra_op_num_threads = 1
            
            logger.info(f"Loading ONNX model with providers: {providers}")
            session = ort.InferenceSession(
                self.model_path,
                sess_options=session_options,
                providers=providers
            )
            
            # Log model input/output info
            input_info = session.get_inputs()[0]
            output_info = session.get_outputs()[0]
            logger.info(f"Model input: {input_info.name}, shape: {input_info.shape}, type: {input_info.type}")
            logger.info(f"Model output: {output_info.name}, shape: {output_info.shape}, type: {output_info.type}")
            
            return session
            
        except Exception as e:
            logger.error(f"Failed to initialize ONNX session: {e}")
            return None
    
    def _decode_request(self, frames: List[bytes]) -> Tuple[np.ndarray, dict]:
        """
        Decode the incoming request frames.
        
        Args:
            frames: List of message frames
            
        Returns:
            Tuple of (tensor, header_dict)
        """
        try:
            if len(frames) < 2:
                raise ValueError(f"Expected 2 frames, got {len(frames)}")
            
            # Parse header
            header_bytes = frames[0]
            tensor_bytes = frames[1]
            
            header = json.loads(header_bytes.decode("utf-8"))
            shape = tuple(header.get("shape", []))
            dtype_str = header.get("dtype", "uint8")
            
            # Convert numpy dtype string to dtype object
            dtype = np.dtype(dtype_str)
            
            # Reconstruct tensor
            tensor = np.frombuffer(tensor_bytes, dtype=dtype).reshape(shape)
            
            logger.debug(f"Received tensor: shape={shape}, dtype={dtype}, size={tensor.nbytes} bytes")
            
            return tensor, header
            
        except Exception as e:
            logger.error(f"Failed to decode request: {e}")
            raise
    
    def _run_inference(self, tensor: np.ndarray) -> np.ndarray:
        """
        Run ONNX inference on the input tensor.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Detection results as numpy array
        """
        if self.session is None:
            logger.warning("No ONNX session available, returning zero results")
            return self.zero_result
        
        try:
            # Prepare input for ONNX Runtime
            input_name = self.session.get_inputs()[0].name
            input_data = {input_name: tensor.astype(np.float32)}
            
            # Run inference
            start_time = time.time()
            outputs = self.session.run(None, input_data)
            inference_time = time.time() - start_time
            
            logger.debug(f"Inference completed in {inference_time:.3f}s")
            
            # Get the first output (assuming single output model)
            result = outputs[0]
            
            # Ensure result has the expected shape [20, 6]
            if result.shape != (20, 6):
                logger.warning(f"Unexpected output shape: {result.shape}, reshaping to (20, 6)")
                if result.size >= 120:  # 20 * 6
                    result = result.flatten()[:120].reshape(20, 6)
                else:
                    # Pad with zeros if too small
                    padded = np.zeros((20, 6), dtype=np.float32)
                    if result.size > 0:
                        padded.flat[:result.size] = result.flat
                    result = padded
            
            # Ensure float32 dtype
            result = result.astype(np.float32)
            
            return result
            
        except Exception as e:
            logger.error(f"ONNX inference failed: {e}")
            return self.zero_result
    
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
                "timestamp": time.time()
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
                "error": "Failed to build response"
            }
            header_bytes = json.dumps(header).encode("utf-8")
            result_bytes = self.zero_result.tobytes(order="C")
            return [header_bytes, result_bytes]
    
    def start_server(self):
        """
        Start the ZMQ server and listen for requests.
        """
        try:
            self.socket.bind(self.endpoint)
            logger.info(f"ZMQ server bound to {self.endpoint}")
            
            while True:
                try:
                    # Receive request
                    frames = self.socket.recv_multipart()
                    logger.debug(f"Received request with {len(frames)} frames")
                    
                    # Process request
                    tensor, header = self._decode_request(frames)
                    
                    # Run inference
                    result = self._run_inference(tensor)
                    
                    # Build and send response
                    response = self._build_response(result)
                    self.socket.send_multipart(response)
                    
                    logger.debug("Response sent successfully")
                    
                except zmq.ZMQError as e:
                    logger.error(f"ZMQ error: {e}")
                    # Send error response
                    error_header = {
                        "shape": [20, 6],
                        "dtype": "float32",
                        "error": str(e)
                    }
                    error_response = [
                        json.dumps(error_header).encode("utf-8"),
                        self.zero_result.tobytes(order="C")
                    ]
                    try:
                        self.socket.send_multipart(error_response)
                    except:
                        pass
                        
                except Exception as e:
                    logger.error(f"Unexpected error: {e}")
                    # Send error response
                    error_header = {
                        "shape": [20, 6],
                        "dtype": "float32",
                        "error": str(e)
                    }
                    error_response = [
                        json.dumps(error_header).encode("utf-8"),
                        self.zero_result.tobytes(order="C")
                    ]
                    try:
                        self.socket.send_multipart(error_response)
                    except:
                        pass
                        
        except KeyboardInterrupt:
            logger.info("Shutting down server...")
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if self.socket:
                self.socket.close()
            if self.context:
                self.context.term()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


def main():
    """Main function to run the ZMQ ONNX client."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ZMQ IPC ONNX Runtime Client")
    parser.add_argument(
        "--endpoint",
        default="ipc:///tmp/cache/zmq_detector",
        help="ZMQ IPC endpoint (default: ipc:///tmp/cache/zmq_detector)"
    )
    parser.add_argument(
        "--model",
        help="Path to ONNX model file"
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["CoreMLExecutionProvider", "CPUExecutionProvider"],
        help="ONNX Runtime execution providers"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and start client
    client = ZmqOnnxClient(
        endpoint=args.endpoint,
        model_path=args.model,
        providers=args.providers
    )
    
    try:
        client.start_server()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        client.cleanup()


if __name__ == "__main__":
    main()
