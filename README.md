# ZMQ IPC ONNX Runtime Client

This Python client connects to the ZMQ IPC proxy, accepts tensor inputs, runs inference via ONNX Runtime, and returns detection results in the format expected by the Frigate detector.

## Features

- **ZMQ IPC Communication**: Implements the REQ/REP protocol over IPC endpoints
- **ONNX Runtime Integration**: Runs inference using ONNX models with optimized execution providers
- **Apple Silicon Optimized**: Defaults to CoreML execution provider for optimal performance on Apple Silicon
- **Error Handling**: Robust error handling with fallback to zero results
- **Flexible Configuration**: Configurable endpoints, model paths, and execution providers

## Installation

### 1. Create and Activate Virtual Environment

First, create a virtual environment to isolate dependencies:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

### 2. Install Dependencies

With the virtual environment activated, install the required dependencies:

```bash
pip install -r requirements.txt
```

### 3. Verify Installation

Ensure you have an ONNX model file ready for inference and verify the installation:

```bash
# Check if virtual environment is active (should show venv path)
which python

# Verify ONNX Runtime is installed
python -c "import onnxruntime; print('ONNX Runtime version:', onnxruntime.__version__)"
```

**Note**: Always activate the virtual environment before running the client or installing additional packages.

## Virtual Environment Management

### Activating the Environment

Every time you open a new terminal or want to work with this project:

```bash
# Navigate to project directory
cd /path/to/apple-silicon-frigate-detector

# Activate virtual environment
source venv/bin/activate

# Your prompt should now show (venv) indicating the environment is active
```

### Deactivating the Environment

When you're done working with the project:

```bash
deactivate
```

### Updating Dependencies

To update packages in your virtual environment:

```bash
# Ensure virtual environment is active
source venv/bin/activate

# Update pip first
pip install --upgrade pip

# Update specific packages
pip install --upgrade package_name

# Or update all packages (use with caution)
pip list --outdated | cut -d ' ' -f1 | xargs -n1 pip install -U
```

### Recreating the Environment

If you need to recreate the virtual environment:

```bash
# Remove old environment
rm -rf venv

# Create new environment
python3 -m venv venv

# Activate and install dependencies
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Command Line Interface

**Important**: Make sure your virtual environment is activated before running any commands:

```bash
# Activate virtual environment (if not already active)
source venv/bin/activate
```

Run the client with basic settings:
```bash
python zmq_onnx_client.py --model /path/to/your/model.onnx
```

Run with custom endpoint:
```bash
python zmq_onnx_client.py --model /path/to/your/model.onnx --endpoint ipc:///tmp/custom_endpoint
```

Run with specific execution providers:
```bash
python zmq_onnx_client.py --model /path/to/your/model.onnx --providers CoreMLExecutionProvider CPUExecutionProvider
```

Enable verbose logging:
```bash
python zmq_onnx_client.py --model /path/to/your/model.onnx --verbose
```

### Programmatic Usage

```python
from zmq_onnx_client import ZmqOnnxClient

# Create client instance
client = ZmqOnnxClient(
    endpoint="ipc:///tmp/cache/zmq_detector",
    model_path="/path/to/your/model.onnx",
    providers=["CoreMLExecutionProvider", "CPUExecutionProvider"]
)

# Start the server
client.start_server()
```

## Protocol

The client implements the same protocol as the Frigate ZMQ detector:

### Request Format
- **Multipart message**: `[header_json_bytes, tensor_bytes]`
- **Header**: JSON containing `shape` and `dtype` information
- **Tensor**: Raw bytes of the numpy array in C-order

### Response Format
- **Multipart message**: `[header_json_bytes, tensor_bytes]`
- **Header**: JSON containing result shape, dtype, and timestamp
- **Result**: Detection results as float32 array with shape `[20, 6]`

## Configuration

### Endpoints
- **Default**: `ipc:///tmp/cache/zmq_detector`
- **Custom**: Any valid ZMQ IPC endpoint

### Execution Providers
- **CoreMLExecutionProvider**: Optimized for Apple Silicon (default)
- **CPUExecutionProvider**: Fallback CPU execution
- **Custom**: Any ONNX Runtime execution provider

### Model Requirements
- Input: Should accept the tensor format sent by Frigate
- Output: Should produce results that can be reshaped to `[20, 6]` float32
- Format: Standard ONNX model file

## Error Handling

The client includes comprehensive error handling:
- **ZMQ Errors**: Automatic socket reset and error response
- **ONNX Errors**: Fallback to zero results with error logging
- **Decoding Errors**: Graceful handling of malformed requests
- **Resource Cleanup**: Proper cleanup on shutdown

## Performance

- **CoreML Optimization**: Leverages Apple's Neural Engine when available
- **Memory Management**: Efficient tensor handling with minimal copying
- **Async Processing**: Non-blocking ZMQ communication
- **Batch Processing**: Ready for future batch inference support

## Troubleshooting

### Common Issues

1. **Virtual Environment Not Active**: Ensure you've activated the virtual environment with `source venv/bin/activate`
2. **Permission Denied**: Ensure the IPC endpoint directory has proper permissions
3. **Model Loading Failed**: Verify the ONNX model path and format
4. **Provider Not Available**: Check ONNX Runtime installation and available providers
5. **ZMQ Bind Failed**: Ensure the endpoint is not already in use
6. **Package Not Found**: If you get import errors, make sure you're in the virtual environment and dependencies are installed

### Debug Mode

Enable verbose logging to see detailed operation information:
```bash
python zmq_onnx_client.py --model /path/to/model.onnx --verbose
```

### Logs

The client provides comprehensive logging:
- Connection status
- Request/response details
- Inference timing
- Error conditions
- Resource cleanup

## Integration with Frigate

This client is designed to work seamlessly with Frigate's ZMQ detector plugin:

1. Start the ONNX client with your model
2. Configure Frigate to use the ZMQ detector with the same endpoint
3. The client will automatically handle all inference requests

## License

This project is provided as-is for integration with Frigate and ONNX Runtime inference.
