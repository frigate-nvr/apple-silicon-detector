# Apple Silicon Detector for Frigate

An optimized object detection client for Frigate that leverages Apple Silicon's Neural Engine for high-performance inference using ONNX Runtime. Provides seamless integration with Frigate's ZMQ detector plugin.

## Features

- **ZMQ IPC Communication**: Implements the REQ/REP protocol over IPC endpoints
- **ONNX Runtime Integration**: Runs inference using ONNX models with optimized execution providers
- **Apple Silicon Optimized**: Defaults to CoreML execution provider for optimal performance on Apple Silicon
- **Error Handling**: Robust error handling with fallback to zero results
- **Flexible Configuration**: Configurable endpoints, model paths, and execution providers

## Quick Start

### 1. Install Dependencies
```bash
make install
```

### 2. Run the Detector
```bash
make run
```

That's it! The detector will automatically use the configured model and start communicating with Frigate.

## What's Included

- **Model Loading**: Uses whatever model Frigate configures via its automatic model loading
- **Apple Silicon Optimization**: Uses CoreML execution provider for maximum performance
- **Frigate Integration**: Drop-in replacement for Frigate's built-in detectors
- **Multiple Model Support**: YOLOv9, RF-DETR, D-FINE, and custom ONNX models

## Supported Models

The following models are supported by this detector:

| Apple Silicon Chip | YOLOv9      | RF-DETR         | D-FINE        |
| -------------------| ----------- | --------------- | ------------- |
| M1                 |             |                 |               |
| M2                 |             |                 |               |
| M3                 | 320-t: 8 ms | 320-Nano: 80 ms | 640-s: 120 ms |
| M4                 |             |                 |               |

### Model Configuration
The detector uses the model that Frigate configures:
1. Frigate automatically loads and configures the model via ZMQ
2. The detector receives model information from Frigate's automatic model loading
3. No manual model selection required - works with Frigate's existing model management

For implementation details, see the [detector README](detector/README.md).

## Virtual Environment Management

- The Makefile automatically manages `venv/` and uses `venv/bin/python3` and `venv/bin/pip3` directly
- If you prefer to activate manually (optional): `source venv/bin/activate`
- Recreate the environment: `make reinstall` (removes `venv/` and reinstalls)
- Verify installation: `venv/bin/python3 -c "import onnxruntime; print('ONNX Runtime version:', onnxruntime.__version__)"`

## Advanced Configuration

### Custom Model Selection
```bash
make run MODEL=/path/to/your/model.onnx
```

### Custom Endpoints
```bash
make run MODEL=/path/to/your/model.onnx ENDPOINT="tcp://*:5555"
```

### Custom Execution Providers
```bash
make run MODEL=/path/to/your/model.onnx PROVIDERS="CoreMLExecutionProvider CPUExecutionProvider"
```

### Verbose Logging
```bash
make run MODEL=/path/to/your/model.onnx VERBOSE=1
```

### Programmatic Usage

```python
from detector.zmq_onnx_client import ZmqOnnxClient

# Create client instance
client = ZmqOnnxClient(
    endpoint="tcp://*:5555",
    model_path="/path/to/your/model.onnx",
    providers=["CoreMLExecutionProvider", "CPUExecutionProvider"]
)

# Start the server
client.start_server()
```

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
- **Permission Denied**: Ensure the IPC endpoint directory has proper permissions (`/tmp/cache/`)
- **Model Loading Failed**: Verify ONNX model files are in the `models/` directory
- **ZMQ Bind Failed**: Ensure the endpoint is not already in use by another process
- **Package Not Found**: Run `make reinstall` to recreate the virtual environment

### Debug Mode
Enable verbose logging for detailed operation information:
```bash
make run VERBOSE=1
```

## Integration with Frigate

This detector works seamlessly with Frigate's ZMQ detector plugin:

1. **Start the detector**: `make run`
2. **Configure Frigate**: Add the ZMQ detector configuration (see Quick Start above)
3. **Done**: Frigate automatically loads the model and the detector handles all inference requests

For detailed implementation information, see the [detector documentation](detector/README.md).

## License

This project is provided as-is for integration with Frigate and ONNX Runtime inference.
