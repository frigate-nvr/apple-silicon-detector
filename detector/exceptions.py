"""Custom exception hierarchy for the Apple Silicon Detector."""


class DetectorError(Exception):
    """Base exception for all detector errors."""


class ModelLoadError(DetectorError):
    """Raised when a model fails to load or initialize."""


class TransportError(DetectorError):
    """Raised when ZMQ transport encounters an error."""


class InferenceError(DetectorError):
    """Raised when ONNX inference fails."""
