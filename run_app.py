#!/usr/bin/env python3
"""
Main application runner for the Social Support System
Handles both API and Streamlit frontend startup
"""

import subprocess
import sys
import time
import signal
import os
from pathlib import Path
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import streamlit
        import fastapi
        import uvicorn
        logger.info("✅ All required dependencies found")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.info("Please install dependencies: pip install -r requirements.txt")
        return False

def start_api_server():
    """Start the FastAPI backend server"""
    logger.info("🚀 Starting FastAPI backend server...")

    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).parent)

    cmd = [
        sys.executable, "-m", "uvicorn",
        "src.api.main:app",
        "--host", "localhost",
        "--port", "8000",
        "--reload"
    ]

    return subprocess.Popen(cmd, env=env)

def start_streamlit_app():
    """Start the Streamlit frontend application"""
    logger.info("🚀 Starting Streamlit frontend...")

    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).parent)

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        "src/frontend/streamlit_app.py",
        "--server.port", "8501",
        "--server.address", "localhost"
    ]

    return subprocess.Popen(cmd, env=env)

def generate_test_data():
    """Generate synthetic test data"""
    logger.info("📊 Generating synthetic test data...")

    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).parent)

    cmd = [sys.executable, "data/synthetic/generate_test_data.py"]

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    if result.returncode == 0:
        logger.info("✅ Test data generated successfully")
        print(result.stdout)
    else:
        logger.error("❌ Failed to generate test data")
        print(result.stderr)

def setup_environment():
    """Setup environment and create necessary directories"""
    logger.info("🔧 Setting up environment...")

    # Create necessary directories
    directories = [
        "data/uploads",
        "data/temp",
        "data/synthetic",
        "logs"
    ]

    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Created directory: {dir_path}")

    # Copy .env.example to .env if it doesn't exist
    env_example = Path(".env.example")
    env_file = Path(".env")

    if env_example.exists() and not env_file.exists():
        import shutil
        shutil.copy(env_example, env_file)
        logger.info("📄 Created .env file from .env.example")
        logger.info("⚠️  Please update .env file with your configuration")

def main():
    """Main function to run the application"""
    parser = argparse.ArgumentParser(description="Social Support System Runner")
    parser.add_argument("--mode", choices=["full", "api", "frontend", "setup", "data"],
                       default="full", help="Running mode")
    parser.add_argument("--no-data", action="store_true",
                       help="Skip test data generation")

    args = parser.parse_args()

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Setup environment
    if args.mode in ["full", "setup"]:
        setup_environment()

    if args.mode == "setup":
        logger.info("✅ Environment setup complete")
        return

    # Generate test data
    if args.mode in ["full", "data"] and not args.no_data:
        generate_test_data()

    if args.mode == "data":
        return

    processes = []

    try:
        # Start services based on mode
        if args.mode in ["full", "api"]:
            api_process = start_api_server()
            processes.append(("API Server", api_process))
            time.sleep(3)  # Give API time to start

        if args.mode in ["full", "frontend"]:
            streamlit_process = start_streamlit_app()
            processes.append(("Streamlit App", streamlit_process))
            time.sleep(2)  # Give Streamlit time to start

        if processes:
            logger.info("🎉 Application started successfully!")
            logger.info("📱 Frontend: http://localhost:8501")
            if args.mode in ["full", "api"]:
                logger.info("🔧 API Docs: http://localhost:8000/docs")

            logger.info("Press Ctrl+C to stop all services")

            # Wait for processes
            for name, process in processes:
                process.wait()

    except KeyboardInterrupt:
        logger.info("🛑 Shutting down services...")

        # Terminate all processes
        for name, process in processes:
            logger.info(f"Stopping {name}...")
            process.terminate()

            # Wait for graceful shutdown
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning(f"Force killing {name}...")
                process.kill()

        logger.info("✅ All services stopped")

    except Exception as e:
        logger.error(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()