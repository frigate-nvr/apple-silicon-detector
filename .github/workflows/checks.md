name: On pull request

on:
  push:
    branches:
      - main
  pull_request:
    paths-ignore:
      - "docs/**"
      - ".github/**"

env:
  DEFAULT_PYTHON: 3.11

jobs:
  python_checks:
    runs-on: ubuntu-latest
    name: Python Checks
    steps:
      - name: Check out the repository
        uses: actions/checkout@v4
        with:
          persist-credentials: false
      - name: Set up Python ${{ env.DEFAULT_PYTHON }}
        uses: actions/setup-python@v5.4.0
        with:
          python-version: ${{ env.DEFAULT_PYTHON }}
      - name: Install requirements
        run: |
          python3 -m pip install -U pip
          python3 -m pip install ruff
      - name: Check formatting
        run: |
          ruff format --check --diff detector test *.py
      - name: Check lint
        run: |
          ruff check frigate migrations docker *.py
