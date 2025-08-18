VENV := venv
PYTHON := python3
PIP := $(VENV)/bin/pip3
PY := $(VENV)/bin/python3

.PHONY: help venv install reinstall clean run

help:
	@echo "Targets:"
	@echo "  venv       - Create local virtual environment in $(VENV)/"
	@echo "  install    - Create venv (if needed) and install dependencies"
	@echo "  run        - Run the ZMQ ONNX client"
	@echo ""
	@echo "Examples:"
	@echo "  make install"
	@echo "  make run MODEL=/path/to/model.onnx"
	@echo "  make run MODEL=/path/to/model.onnx ENDPOINT=ipc:///tmp/cache/zmq_detector"
	@echo "  make run MODEL=/path/to/model.onnx PROVIDERS=\"CoreMLExecutionProvider CPUExecutionProvider\" VERBOSE=1"

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run: venv
	$(PY) zmq_onnx_client.py $(if $(MODEL),--model $(MODEL),) $(if $(ENDPOINT),--endpoint $(ENDPOINT),) $(if $(PROVIDERS),--providers $(PROVIDERS),) $(if $(VERBOSE),-v,)

reinstall: clean install

clean:
	rm -rf $(VENV)


 