#!/usr/bin/env python3
"""
Simple ZMQ connection test script to verify connectivity.
"""

import zmq
import time
import json
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_zmq_server(endpoint="ipc:///tmp/cache/zmq_detector"):
    """Test ZMQ server connectivity."""
    try:
        # Create context and socket
        context = zmq.Context.instance()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, 5000)
        socket.setsockopt(zmq.SNDTIMEO, 5000)

        logger.info(f"Connecting to {endpoint}")
        socket.connect(endpoint)

        # Create test data
        test_tensor = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        header = {
            "shape": list(test_tensor.shape),
            "dtype": str(test_tensor.dtype.name),
        }

        # Send test message
        message = [json.dumps(header).encode("utf-8"), test_tensor.tobytes(order="C")]

        logger.info("Sending test message...")
        socket.send_multipart(message)

        # Wait for response
        logger.info("Waiting for response...")
        response = socket.recv_multipart()

        if len(response) >= 2:
            response_header = json.loads(response[0].decode("utf-8"))
            logger.info(f"Received response: {response_header}")

            if "error" in response_header:
                logger.error(f"Server returned error: {response_header['error']}")
                return False
            else:
                logger.info("Connection test successful!")
                return True
        else:
            logger.error(f"Unexpected response format: {len(response)} frames")
            return False

    except zmq.ZMQError as e:
        logger.error(f"ZMQ error: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False
    finally:
        try:
            socket.close()
            context.term()
        except:
            pass


def test_zmq_bind(endpoint="ipc:///tmp/cache/zmq_detector"):
    """Test if we can bind to the endpoint (useful for debugging)."""
    try:
        context = zmq.Context.instance()
        socket = context.socket(zmq.REP)

        logger.info(f"Testing bind to {endpoint}")
        socket.bind(endpoint)
        logger.info("Bind successful!")

        # Clean up
        socket.close()
        context.term()
        return True

    except zmq.ZMQError as e:
        logger.error(f"Bind failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False


def main():
    """Main test function."""
    import argparse

    parser = argparse.ArgumentParser(description="ZMQ Connection Test")
    parser.add_argument(
        "--endpoint",
        default="ipc:///tmp/cache/zmq_detector",
        help="ZMQ endpoint to test",
    )
    parser.add_argument(
        "--bind-test",
        action="store_true",
        help="Test binding to endpoint instead of connecting",
    )

    args = parser.parse_args()

    if args.bind_test:
        success = test_zmq_bind(args.endpoint)
    else:
        success = test_zmq_server(args.endpoint)

    if success:
        logger.info("Test completed successfully")
        exit(0)
    else:
        logger.error("Test failed")
        exit(1)


if __name__ == "__main__":
    main()
