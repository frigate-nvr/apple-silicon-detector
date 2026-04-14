# Apple Silicon Detector for Frigate

An optimized object detection client for Frigate that leverages Apple Silicon's Neural Engine for high-performance inference using ONNX Runtime. Ships as a native macOS menu bar app with a full CLI — no terminal expertise required.

## Features

- **Native Menu Bar GUI**: SwiftUI-based macOS menu bar app with real-time status, controls, and settings
- **Unified CLI**: Full `detector` command with 11 subcommands for terminal-based management
- **Self-Contained App**: Zero-prerequisite `FrigateDetector.app` distributed as a DMG installer
- **ZMQ Multi-Endpoint**: Binds simultaneously to TCP and IPC endpoints for maximum flexibility
- **ONNX Runtime + CoreML**: Optimized inference on Apple Silicon's Neural Engine
- **Service Management**: `launchd` integration with auto-start, persistent config, and live reload
- **Multiple Model Support**: YOLOv9, RF-DETR, D-FINE, and custom ONNX models

## Quick Start

### Option A: Desktop App (recommended — no terminal required)
1. Download `FrigateDetector.dmg` from the [Releases](https://github.com/frigate-nvr/apple-silicon-detector/releases) page.
2. Drag `FrigateDetector.app` to `/Applications/`.
3. Open it — the Frigate Detector icon will appear in your macOS menu bar.
4. Click the icon → toggle **Open at Login** or use the **Start Detector** button.
5. Optionally install the `detector` CLI via **Advanced → Install CLI Tool...** referenced in [CLI Reference](#cli-reference).

### Option B: Headless (SSH / Terminal)
No display required. After copying the app to your server, use the embedded CLI binary:
```bash
# Enable the background daemon
/Applications/FrigateDetector.app/Contents/MacOS/detector startup enable --headless

# Check status
/Applications/FrigateDetector.app/Contents/MacOS/detector status

# View logs
/Applications/FrigateDetector.app/Contents/MacOS/detector logs -f
```

### Option C: Source (developer install)
Runs directly from the repository using [uv](https://docs.astral.sh/uv/):
```bash
git clone https://github.com/frigate-nvr/apple-silicon-detector
cd apple-silicon-detector

# Install dependencies
uv sync

# Run the detector in the foreground
uv run detector start

# Or run as a background daemon
uv run detector start --daemon
```

The detector will automatically use the model in the Frigate communication and start communicating with Frigate. See [the Frigate documentation](https://deploy-preview-19787--frigate-docs.netlify.app/configuration/object_detectors#apple-silicon-detector) for instructions on setting up the detector.

## CLI Reference

The `detector` CLI provides complete control over the detector service:

| Command | Description |
|---|---|
| `detector start` | Start the detector in the foreground |
| `detector start --daemon` | Start as a background service |
| `detector stop` | Stop the running detector |
| `detector restart` | Restart the detector service |
| `detector status` | Show running state, PID, endpoints, and settings |
| `detector logs` | Show recent log output |
| `detector logs -f` | Tail live log output |
| `detector logs --err` | Show error log |
| `detector models` | List installed models with sizes |
| `detector startup enable` | Enable auto-start on login (GUI mode) |
| `detector startup enable --headless` | Enable auto-start on login (headless mode) |
| `detector startup disable` | Disable auto-start on login |
| `detector debug enable` | Enable verbose debug logging |
| `detector debug disable` | Disable verbose debug logging |
| `detector config` | Show current paths and settings |
| `detector install-cli` | Install `detector` CLI symlink to `~/.local/bin/` |
| `detector uninstall-cli` | Remove `detector` CLI symlink |
| `detector version` | Show version |

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

## Advanced Configuration

### Custom Model Path
```bash
detector start --model /path/to/your/model.onnx
```

### Custom & Multiple Endpoints
By default, the detector binds simultaneously to both `tcp://0.0.0.0:5555` and `ipc:///tmp/frigate-detector/zmq_detector`. You can specify your own:
```bash
# Bind to a specific TCP port
detector start --endpoint tcp://0.0.0.0:9999

# Bind to multiple endpoints at once
detector start --endpoint tcp://0.0.0.0:9999 ipc:///tmp/custom_socket
```

### Custom Execution Providers
```bash
detector start --providers CoreMLExecutionProvider CPUExecutionProvider
```

### Verbose Logging
```bash
# One-time verbose session
detector start --verbose

# Persistent debug mode (survives restarts)
detector debug enable
```

### Programmatic Usage

```python
from detector.zmq_onnx_client import ZmqOnnxClient

# Create client instance
client = ZmqOnnxClient(
    endpoints=["tcp://0.0.0.0:5555", "ipc:///tmp/frigate-detector/zmq_detector"],
    model_path="/path/to/your/model.onnx",
    providers=["CoreMLExecutionProvider", "CPUExecutionProvider"]
)

# Start the server
client.start_server()
```

## File Locations

| Resource | Path |
|---|---|
| Models | `~/Library/Application Support/FrigateDetector/models/` |
| Logs | `~/Library/Logs/FrigateDetector/detector.log` |
| Error log | `~/Library/Logs/FrigateDetector/detector.err.log` |
| Config | `~/Library/Application Support/FrigateDetector/config.json` |
| Service plist | `~/Library/LaunchAgents/com.frigate.apple-silicon-detector.plist` |
| CLI symlink | `~/.local/bin/detector` |

## Error Handling

The client includes comprehensive error handling with a structured exception hierarchy:
- **`ModelLoadError`**: ONNX model fails to load or initialize
- **`TransportError`**: ZMQ transport encounters an error, with automatic socket reset
- **`InferenceError`**: ONNX inference fails, with fallback to zero results
- **Graceful shutdown**: SIGTERM/SIGINT handlers ensure clean resource release and IPC socket cleanup

## Performance

- **CoreML Optimization**: Leverages Apple's Neural Engine when available
- **Memory Management**: Efficient tensor handling with minimal copying
- **TCP Keepalive**: Detects dead peers and frees ports faster (30s idle timeout)
- **Fast Signal Response**: 1s ZMQ heartbeat timeout for responsive shutdown
- **Resource Cleanup**: Explicit ONNX session release frees NPU/GPU on shutdown

## Troubleshooting

### Common Issues
- **Port Already in Use**: Stop the existing process with `detector stop` or use a different endpoint: `--endpoint tcp://0.0.0.0:5556`
- **Model Loading Failed**: Verify ONNX model files exist in `~/Library/Application Support/FrigateDetector/models/`
- **ZMQ Bind Failed**: Ensure the endpoint is not already in use by another process
- **App Won't Start**: Check logs with `detector logs --err` for detailed error output
- **Stale Process**: Use `detector stop` which handles SIGTERM → SIGKILL escalation automatically

### Debug Mode
Enable persistent verbose logging:
```bash
detector debug enable
detector logs -f
# ... reproduce the issue ...
detector debug disable
```

## Development

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for dependency management
- Xcode Command Line Tools (for Swift GUI build)

### Build from Source
```bash
make install     # Install all dependencies
make test        # Run pytest suite
make lint        # Run ruff linter
make format      # Auto-format with ruff
make typecheck   # Run pyright type checker
make check       # Run lint + typecheck + test
make smoke-test  # Run ZMQ connection smoke test
make build       # Build FrigateDetector.app + DMG
```

## Integration with Frigate

This detector works seamlessly with Frigate's ZMQ detector plugin:

1. **Start the detector**: `detector start` (or launch `FrigateDetector.app`)
2. **Configure Frigate**: Add the ZMQ detector configuration (see Quick Start above)
3. **Done**: Frigate automatically loads the model and the detector handles all inference requests

For detailed implementation information, see the [detector documentation](detector/README.md).

## License

This project is provided as-is for integration with Frigate and ONNX Runtime inference.
