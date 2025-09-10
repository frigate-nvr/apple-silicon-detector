# Apple Silicon Detector

This Python client connects to the ZMQ IPC proxy, accepts tensor inputs, runs inference via ONNX Runtime, and returns detection results in the format expected by the Frigate detector.

## Features

- **ZMQ IPC Communication**: Implements the REQ/REP protocol over IPC endpoints
- **ONNX Runtime Integration**: Runs inference using ONNX models with optimized execution providers
- **Apple Silicon Optimized**: Defaults to CoreML execution provider for optimal performance on Apple Silicon
- **Error Handling**: Robust error handling with fallback to zero results
- **Flexible Configuration**: Configurable endpoints, model paths, and execution providers

## Setup (via Makefile)

```bash
# Create local venv and install dependencies
make install

# Optional: verify ONNX Runtime is available
venv/bin/python3 -c "import onnxruntime; print('ONNX Runtime version:', onnxruntime.__version__)"
```

## Virtual Environment

- The Makefile manages `venv/` and uses `venv/bin/python3` and `venv/bin/pip3` directly.
- If you prefer to activate manually (optional): `source venv/bin/activate`
- Recreate the environment: `make reinstall` (removes `venv/` and reinstalls)

## Usage

### Make targets

Run the client with a model:
```bash
make run MODEL=/path/to/your/model.onnx
```

Custom endpoint (examples include TCP):
```bash
make run MODEL=/path/to/your/model.onnx ENDPOINT="tcp://*:5555"
```

Specific execution providers:
```bash
make run MODEL=/path/to/your/model.onnx PROVIDERS="CoreMLExecutionProvider CPUExecutionProvider"
```

Enable verbose logging:
```bash
make run MODEL=/path/to/your/model.onnx VERBOSE=1
```

### Programmatic Usage

```python
from zmq_onnx_client import ZmqOnnxClient

# Create client instance
client = ZmqOnnxClient(
    endpoint="tcp://*:5555",
    model_path="/path/to/your/model.onnx",
    providers=["CoreMLExecutionProvider", "CPUExecutionProvider"]
)

# Start the server
client.start_server()
```

## Supported Models

The following models are supported by this detector:

| Apple Silicon Chip | YOLOv9      | RF-DETR         | D-FINE        |
| -------------------| ----------- | --------------- | ------------- |
| M1                 |             |                 |               |
| M2                 |             |                 |               |
| M3                 | 320-t: 8 ms | 320-Nano: 80 ms | 640-s: 120 ms |
| M4                 |             |                 |               |

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
- **Examples**: `tcp://*:5555`, 
- **Custom**: Any valid ZMQ endpoint (TCP)

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
make run MODEL=/path/to/model.onnx VERBOSE=1
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
