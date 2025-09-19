"""Apple Silicon Frigate Detector

A high-performance ONNX-based object detection client optimized for Apple Silicon,
designed to work seamlessly with Frigate NVR via ZMQ communication.
"""

__version__ = "1.0.0"
__author__ = "Apple Silicon Frigate Detector Team"

from .zmq_onnx_client import ZmqOnnxClient
from .model_util import post_process_yolo, post_process_rfdetr, post_process_dfine

__all__ = [
    "ZmqOnnxClient",
    "post_process_yolo", 
    "post_process_rfdetr", 
    "post_process_dfine"
]
