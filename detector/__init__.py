from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("frigate-apple-silicon-detector")
except PackageNotFoundError:
    __version__ = "dev"
__app_name__ = "FrigateDetector"
