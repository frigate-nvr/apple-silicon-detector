# Apple Silicon Detector Implementation

This directory contains the core detector implementation that provides ZMQ-based object detection for Frigate using ONNX Runtime optimized for Apple Silicon.

## Architecture

### ZMQ Communication
- **Protocol**: REQ/REP over IPC endpoints
- **Transport**: IPC sockets for local communication with Frigate
- **Message Format**: Multipart messages with header and tensor data

### ONNX Runtime Integration
- **Execution Providers**: CoreML (Apple Silicon optimized) with CPU fallback
- **Model Support**: YOLOv9, RF-DETR, D-FINE, and custom ONNX models
- **Auto Model Detection**: Automatically selects appropriate model based on available files

## Input/Output Protocol

### Request Format
The detector expects multipart ZMQ messages:
- **Part 1**: JSON header containing:
  - `shape`: Tensor dimensions (e.g., `[1, 3, 320, 320]`)
  - `dtype`: Data type (typically `"float32"`)
- **Part 2**: Raw tensor bytes in C-order numpy array format

### Response Format
Returns multipart ZMQ messages:
- **Part 1**: JSON header containing:
  - `shape`: Result dimensions (typically `[20, 6]`)
  - `dtype`: Result data type (`"float32"`)
  - `timestamp`: Processing timestamp
- **Part 2**: Detection results as float32 array with shape `[20, 6]`
  - Each detection: `[x, y, width, height, confidence, class_id]`
  - Maximum 20 detections per frame

## Model Processing

### Supported Models
- **YOLOv9**: Object detection with standard YOLO output format
- **RF-DETR**: Transformer-based detection with custom post-processing
- **D-FINE**: Efficient detection model with specialized processing
- **Custom Models**: Any ONNX model compatible with Frigate's expected input/output

### Post-Processing
Each model type has specialized post-processing functions in `model_util.py`:
- `post_process_yolo()`: Standard YOLO non-maximum suppression
- `post_process_rfdetr()`: Transformer-specific output processing
- `post_process_dfine()`: D-FINE model output formatting

### Model Loading via Frigate
When `model_path="AUTO"`, the detector works with Frigate's automatic model loading:
1. Frigate automatically loads and configures the model via ZMQ communication
2. The detector receives model information from Frigate's model management system
3. No local model selection required - leverages Frigate's existing model handling

## Configuration

### Default Settings
- **Endpoint**: `ipc:///tmp/cache/zmq_detector`
- **Model**: Loaded via Frigate's automatic model loading system
- **Providers**: `["CoreMLExecutionProvider", "CPUExecutionProvider"]`
- **Input Shape**: Auto-detected from model
- **Max Detections**: 20 per frame

### Execution Providers
- **CoreMLExecutionProvider**: Primary provider for Apple Silicon optimization
  - Uses MLProgram format for optimal performance (requires iOS 15+ or macOS 12+)
  - Leverages all available compute units (CPU, GPU, Neural Engine)
  - Model caching enabled for faster subsequent loads
- **CPUExecutionProvider**: Fallback for compatibility
  - Ensures functionality on all systems
  - Used when CoreML is unavailable

## Error Handling

### Robust Error Recovery
- **ZMQ Errors**: Automatic socket reset and error response
- **ONNX Errors**: Fallback to zero results with detailed logging
- **Decoding Errors**: Graceful handling of malformed requests
- **Resource Cleanup**: Proper cleanup on shutdown

### Error Response Format
When errors occur, the detector returns:
- **Header**: Error information in JSON format
- **Result**: Zero-filled array `[20, 6]` to maintain protocol compatibility

## Performance Characteristics

### Apple Silicon Optimization
- **M1/M2**: Basic CoreML support with model caching
- **M3**: Significant performance improvements
  - YOLOv9 320-t: ~8ms inference time
  - RF-DETR 320-Nano: ~80ms inference time
  - D-FINE 640-s: ~120ms inference time
- **M4**: Enhanced Neural Engine utilization

### Memory Management
- **Efficient Tensor Handling**: Minimal copying with numpy views
- **Batch Processing Ready**: Architecture supports future batch inference
- **Memory Cleanup**: Automatic resource management

## Integration Points

### Frigate Integration
- **Plugin Compatibility**: Works with Frigate's ZMQ detector plugin
- **Protocol Compliance**: Maintains exact protocol compatibility
- **Configuration**: Uses standard Frigate detector configuration

### ONNX Runtime
- **Version Compatibility**: Works with ONNX Runtime 1.22+
- **Provider Flexibility**: Supports any ONNX Runtime execution provider
- **Model Format**: Standard ONNX model files (.onnx)

## Development Notes

### Logging
Comprehensive logging includes:
- Connection status and endpoint information
- Request/response details and timing
- Inference performance metrics
- Error conditions and recovery
- Resource cleanup operations

### Testing
- **Unit Tests**: Located in `test/` directory
- **Integration Tests**: ZMQ connection and protocol testing
- **Performance Tests**: Inference timing and memory usage

### Extensibility
The architecture supports:
- **Custom Models**: Easy integration of new model types
- **Custom Post-Processing**: Model-specific output handling
- **Custom Providers**: Additional ONNX Runtime execution providers
- **Protocol Extensions**: Future protocol enhancements
