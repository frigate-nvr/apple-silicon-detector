#!/usr/bin/env python3
"""
Command Line Interface for Apple Silicon Frigate Detector
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from .zmq_onnx_client import ZmqOnnxClient


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def validate_model_path(model_path: str) -> Path:
    """Validate that the model path exists and is an ONNX file."""
    path = Path(model_path).expanduser().resolve()
    
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Model file does not exist: {path}")
    
    if not path.suffix.lower() == '.onnx':
        raise argparse.ArgumentTypeError(f"Model file must be an .onnx file: {path}")
    
    return path


def parse_providers(providers_str: Optional[str]) -> Optional[List[str]]:
    """Parse execution providers from string."""
    if not providers_str:
        return None
    
    return [p.strip() for p in providers_str.split() if p.strip()]


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Apple Silicon Frigate Detector - ONNX inference server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with model
  frigate-detector --model ~/models/yolov9.onnx
  
  # Custom endpoint and providers
  frigate-detector --model ~/models/yolov9.onnx --endpoint tcp://*:5555 --providers "CoreMLExecutionProvider CPUExecutionProvider"
  
  # Verbose logging
  frigate-detector --model ~/models/yolov9.onnx --verbose
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        type=validate_model_path,
        required=True,
        help="Path to ONNX model file"
    )
    
    parser.add_argument(
        "--endpoint", "-e",
        default="tcp://*:5555",
        help="ZMQ endpoint (default: tcp://*:5555)"
    )
    
    parser.add_argument(
        "--providers", "-p",
        type=parse_providers,
        help="Space-separated list of ONNX Runtime execution providers"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Create and start client
        logger.info(f"Starting Frigate detector with model: {args.model}")
        logger.info(f"Endpoint: {args.endpoint}")
        if args.providers:
            logger.info(f"Execution providers: {args.providers}")
        
        client = ZmqOnnxClient(
            endpoint=args.endpoint,
            model_path=str(args.model),
            providers=args.providers
        )
        
        logger.info("Starting server...")
        client.start_server()
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
