#!/usr/bin/env python3
"""
Build script for Apple Silicon Frigate Detector

This script handles building, testing, and packaging the application
for distribution.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and handle errors."""
    print(f"\n{'='*50}")
    print(f"🔧 {description or cmd}")
    print(f"{'='*50}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print("STDOUT:")
        print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    if result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}")
        sys.exit(1)
    else:
        print("✅ Success!")
    
    return result


def clean_build():
    """Clean previous build artifacts."""
    print("🧹 Cleaning build artifacts...")
    
    dirs_to_clean = [
        "build",
        "dist", 
        "*.egg-info",
        "__pycache__",
        "apple_silicon_frigate_detector/__pycache__"
    ]
    
    for pattern in dirs_to_clean:
        for path in Path(".").glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
                print(f"  Removed directory: {path}")
            else:
                path.unlink()
                print(f"  Removed file: {path}")


def check_dependencies():
    """Check that build dependencies are installed."""
    print("📦 Checking build dependencies...")
    
    required_packages = ["build", "twine", "pytest"]
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"  ❌ {package}")
    
    if missing:
        print(f"\n❌ Missing required packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        sys.exit(1)


def run_tests():
    """Run the test suite."""
    if Path("test").exists():
        run_command("python -m pytest test/ -v", "Running tests")
    else:
        print("⚠️  No tests found, skipping test phase")


def lint_code():
    """Run code linting."""
    try:
        run_command("python -m ruff check apple_silicon_frigate_detector/", "Linting code with ruff")
    except:
        print("⚠️  Ruff not available, skipping linting")
    
    try:
        run_command("python -m black --check apple_silicon_frigate_detector/", "Checking code formatting")
    except:
        print("⚠️  Black not available, skipping format check")


def build_package():
    """Build the Python package."""
    run_command("python -m build", "Building Python package")
    
    # List built files
    dist_files = list(Path("dist").glob("*"))
    print(f"\n📦 Built files:")
    for file in dist_files:
        size = file.stat().st_size / 1024 / 1024  # MB
        print(f"  {file.name} ({size:.1f} MB)")


def validate_package():
    """Validate the built package."""
    run_command("python -m twine check dist/*", "Validating package")


def test_install():
    """Test installation in a clean virtual environment."""
    print("🧪 Testing installation in clean environment...")
    
    # Create test venv
    test_venv = Path("test_venv")
    if test_venv.exists():
        shutil.rmtree(test_venv)
    
    run_command(f"python -m venv {test_venv}", "Creating test virtual environment")
    
    # Get the built wheel
    wheel_file = list(Path("dist").glob("*.whl"))[0]
    
    # Install and test
    pip_cmd = f"{test_venv}/bin/pip"
    python_cmd = f"{test_venv}/bin/python"
    
    run_command(f"{pip_cmd} install {wheel_file}", "Installing package in test environment")
    run_command(f"{python_cmd} -c 'import apple_silicon_frigate_detector; print(\"Import successful!\")'", "Testing import")
    run_command(f"{test_venv}/bin/frigate-detector --version", "Testing CLI entry point")
    
    # Cleanup
    shutil.rmtree(test_venv)
    print("✅ Installation test passed!")


def main():
    """Main build process."""
    print("🚀 Apple Silicon Frigate Detector Build Process")
    print("=" * 60)
    
    # Parse arguments
    skip_tests = "--skip-tests" in sys.argv
    skip_lint = "--skip-lint" in sys.argv
    skip_install_test = "--skip-install-test" in sys.argv
    
    try:
        # Build steps
        clean_build()
        check_dependencies()
        
        if not skip_lint:
            lint_code()
        
        if not skip_tests:
            run_tests()
        
        build_package()
        validate_package()
        
        if not skip_install_test:
            test_install()
        
        print("\n" + "=" * 60)
        print("🎉 BUILD SUCCESSFUL!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Test the package: pip install dist/*.whl")
        print("2. Upload to PyPI: python -m twine upload dist/*")
        print("3. Create GitHub release with installer scripts")
        
    except KeyboardInterrupt:
        print("\n❌ Build interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Build failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
