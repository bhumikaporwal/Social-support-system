#!/usr/bin/env python3
"""
Setup script for the UAE Social Support System
Automates the complete system setup including dependencies and configuration
"""

import subprocess
import sys
import os
import shutil
import time
import json
from pathlib import Path
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class SocialSupportSetup:
    """Complete setup automation for the Social Support System"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.requirements_installed = False
        self.databases_started = False
        self.ollama_configured = False

    def check_system_requirements(self):
        """Check system requirements"""
        logger.info("🔍 Checking system requirements...")

        # Check Python version
        if sys.version_info < (3, 8):
            logger.error("❌ Python 3.8+ is required")
            return False

        # Check if Docker is available
        try:
            subprocess.run(["docker", "--version"], check=True, capture_output=True)
            logger.info("✅ Docker is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("❌ Docker is not installed or not accessible")
            return False

        # Check if Docker Compose is available
        try:
            subprocess.run(["docker-compose", "--version"], check=True, capture_output=True)
            logger.info("✅ Docker Compose is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("❌ Docker Compose is not installed")
            return False

        # Check available disk space (need at least 10GB)
        import shutil
        free_space = shutil.disk_usage(self.project_root).free / (1024**3)
        if free_space < 10:
            logger.warning(f"⚠️  Low disk space: {free_space:.1f}GB available (10GB+ recommended)")
        else:
            logger.info(f"✅ Sufficient disk space: {free_space:.1f}GB available")

        logger.info("✅ System requirements check completed")
        return True

    def create_directory_structure(self):
        """Create necessary directories"""
        logger.info("📁 Creating directory structure...")

        directories = [
            "data/uploads",
            "data/temp",
            "data/synthetic",
            "logs",
            "docs",
            "tests",
            "config"
        ]

        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created: {directory}")

        logger.info("✅ Directory structure created")

    def setup_environment_file(self):
        """Setup environment configuration"""
        logger.info("⚙️  Setting up environment configuration...")

        env_example = self.project_root / ".env.example"
        env_file = self.project_root / ".env"

        if not env_file.exists() and env_example.exists():
            shutil.copy(env_example, env_file)
            logger.info("📄 Created .env file from .env.example")

            # Update with Docker service names
            with open(env_file, 'r') as f:
                content = f.read()

            # Replace localhost with Docker service names
            content = content.replace("localhost", "postgres").replace("postgres:27017", "mongodb:27017")
            content = content.replace("QDRANT_HOST=postgres", "QDRANT_HOST=qdrant")
            content = content.replace("NEO4J_URI=bolt://postgres:7687", "NEO4J_URI=bolt://neo4j:7687")
            content = content.replace("OLLAMA_BASE_URL=http://postgres:11434", "OLLAMA_BASE_URL=http://ollama:11434")

            with open(env_file, 'w') as f:
                f.write(content)

            logger.info("🔧 Updated .env with Docker configurations")
        else:
            logger.info("📄 .env file already exists")

    def install_python_dependencies(self):
        """Install Python dependencies"""
        logger.info("📦 Installing Python dependencies...")

        try:
            # Upgrade pip first
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)

            # Install requirements
            requirements_file = self.project_root / "requirements.txt"
            if requirements_file.exists():
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
                ], check=True)
                logger.info("✅ Python dependencies installed")
                self.requirements_installed = True
            else:
                logger.error("❌ requirements.txt not found")
                return False

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install dependencies: {e}")
            return False

        return True

    def setup_databases(self):
        """Setup and start database services"""
        logger.info("🗄️  Setting up databases...")

        try:
            # Pull required Docker images
            logger.info("📥 Pulling Docker images...")
            images = [
                "postgres:13",
                "mongo:5.0",
                "qdrant/qdrant:latest",
                "neo4j:5.0",
                "ollama/ollama:latest",
                "redis:7-alpine"
            ]

            for image in images:
                logger.info(f"Pulling {image}...")
                subprocess.run(["docker", "pull", image], check=True, capture_output=True)

            # Start databases using Docker Compose
            logger.info("🚀 Starting database services...")
            subprocess.run([
                "docker-compose", "up", "-d",
                "postgres", "mongodb", "qdrant", "neo4j", "redis"
            ], check=True, cwd=self.project_root)

            # Wait for services to be ready
            logger.info("⏳ Waiting for services to be ready...")
            time.sleep(30)

            # Check service health
            self.check_database_health()

            logger.info("✅ Database services started")
            self.databases_started = True

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to start databases: {e}")
            return False

        return True

    def check_database_health(self):
        """Check if databases are healthy"""
        logger.info("🏥 Checking database health...")

        services = {
            "PostgreSQL": ("localhost", 5432),
            "MongoDB": ("localhost", 27017),
            "Qdrant": ("localhost", 6333),
            "Neo4j": ("localhost", 7474),
            "Redis": ("localhost", 6379)
        }

        for service, (host, port) in services.items():
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((host, port))
                sock.close()

                if result == 0:
                    logger.info(f"✅ {service} is healthy on {host}:{port}")
                else:
                    logger.warning(f"⚠️  {service} may not be ready yet on {host}:{port}")
            except Exception as e:
                logger.warning(f"⚠️  Could not check {service}: {e}")

    def setup_ollama(self):
        """Setup Ollama and download models"""
        logger.info("🤖 Setting up Ollama LLM...")

        try:
            # Start Ollama service
            subprocess.run(["docker-compose", "up", "-d", "ollama"], check=True, cwd=self.project_root)

            # Wait for Ollama to be ready
            logger.info("⏳ Waiting for Ollama to start...")
            time.sleep(10)

            # Download required models
            models = ["llama2:7b-chat"]

            for model in models:
                logger.info(f"📥 Downloading model: {model}")
                try:
                    # Use docker exec to pull model inside container
                    subprocess.run([
                        "docker", "exec", "social-support-ollama",
                        "ollama", "pull", model
                    ], check=True, timeout=1800)  # 30 minutes timeout
                    logger.info(f"✅ Model {model} downloaded successfully")
                except subprocess.TimeoutExpired:
                    logger.warning(f"⚠️  Model {model} download timed out, continuing...")
                except subprocess.CalledProcessError as e:
                    logger.warning(f"⚠️  Failed to download {model}: {e}")

            self.ollama_configured = True
            logger.info("✅ Ollama setup completed")

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to setup Ollama: {e}")
            return False

        return True

    def initialize_databases(self):
        """Initialize database schemas"""
        logger.info("🏗️  Initializing database schemas...")

        try:
            # Run database initialization
            init_script = self.project_root / "src" / "database" / "init_db.py"
            if init_script.exists():
                env = os.environ.copy()
                env['PYTHONPATH'] = str(self.project_root)
                subprocess.run([sys.executable, str(init_script)], check=True, env=env)
                logger.info("✅ Database schemas initialized")
            else:
                logger.warning("⚠️  Database initialization script not found")

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to initialize databases: {e}")
            return False

        return True

    def generate_test_data(self):
        """Generate synthetic test data"""
        logger.info("📊 Generating synthetic test data...")

        try:
            test_data_script = self.project_root / "data" / "synthetic" / "generate_test_data.py"
            if test_data_script.exists():
                env = os.environ.copy()
                env['PYTHONPATH'] = str(self.project_root)
                subprocess.run([sys.executable, str(test_data_script)], check=True, env=env)
                logger.info("✅ Test data generated")
            else:
                logger.warning("⚠️  Test data generation script not found")

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to generate test data: {e}")
            return False

        return True

    def create_startup_scripts(self):
        """Create convenient startup scripts"""
        logger.info("📜 Creating startup scripts...")

        # Create start script
        start_script_content = """#!/bin/bash
echo "🚀 Starting UAE Social Support System..."

# Start all services
docker-compose up -d

# Wait for services
echo "⏳ Waiting for services to start..."
sleep 10

# Start API
echo "🔧 Starting API server..."
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload &

# Start Frontend
echo "📱 Starting frontend..."
streamlit run src/frontend/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 &

echo "✅ System started!"
echo "📱 Frontend: http://localhost:8501"
echo "🔧 API Docs: http://localhost:8000/docs"
echo "📊 Langfuse: http://localhost:3000"
"""

        start_script = self.project_root / "start.sh"
        with open(start_script, 'w') as f:
            f.write(start_script_content)

        # Make executable on Unix systems
        if os.name != 'nt':
            os.chmod(start_script, 0o755)

        # Create stop script
        stop_script_content = """#!/bin/bash
echo "🛑 Stopping UAE Social Support System..."

# Kill Python processes
pkill -f "uvicorn"
pkill -f "streamlit"

# Stop Docker services
docker-compose down

echo "✅ System stopped!"
"""

        stop_script = self.project_root / "stop.sh"
        with open(stop_script, 'w') as f:
            f.write(stop_script_content)

        if os.name != 'nt':
            os.chmod(stop_script, 0o755)

        logger.info("✅ Startup scripts created")

    def run_tests(self):
        """Run basic system tests"""
        logger.info("🧪 Running system tests...")

        try:
            # Test API endpoints
            test_script = self.project_root / "tests" / "test_basic.py"
            if test_script.exists():
                env = os.environ.copy()
                env['PYTHONPATH'] = str(self.project_root)
                subprocess.run([sys.executable, "-m", "pytest", str(test_script), "-v"],
                             check=True, env=env)
                logger.info("✅ Basic tests passed")
            else:
                logger.warning("⚠️  Test files not found")

        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠️  Some tests failed: {e}")

    def print_setup_summary(self):
        """Print setup summary and next steps"""
        logger.info("📋 Setup Summary:")

        print("\n" + "="*60)
        print("🇦🇪 UAE Social Support System - Setup Complete!")
        print("="*60)

        print(f"✅ Requirements installed: {self.requirements_installed}")
        print(f"✅ Databases started: {self.databases_started}")
        print(f"✅ Ollama configured: {self.ollama_configured}")

        print("\n📍 Access Points:")
        print("  📱 Frontend: http://localhost:8501")
        print("  🔧 API Docs: http://localhost:8000/docs")
        print("  📊 Langfuse: http://localhost:3000")
        print("  🗄️  Databases:")
        print("     - PostgreSQL: localhost:5432")
        print("     - MongoDB: localhost:27017")
        print("     - Qdrant: localhost:6333")
        print("     - Neo4j: localhost:7474")

        print("\n🚀 Quick Start:")
        print("  python run_app.py --mode full")
        print("  # Or use the generated scripts:")
        print("  ./start.sh  # Start system")
        print("  ./stop.sh   # Stop system")

        print("\n📚 Documentation:")
        print("  - README.md: Complete documentation")
        print("  - docs/solution_summary.md: Technical overview")
        print("  - API docs: http://localhost:8000/docs")

        print("\n🔧 Configuration:")
        print("  - Edit .env for custom settings")
        print("  - Check docker-compose.yml for service config")

        print("="*60)

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="UAE Social Support System Setup")
    parser.add_argument("--skip-models", action="store_true",
                       help="Skip Ollama model downloads")
    parser.add_argument("--skip-tests", action="store_true",
                       help="Skip running tests")
    parser.add_argument("--minimal", action="store_true",
                       help="Minimal setup (no test data, no models)")

    args = parser.parse_args()

    setup = SocialSupportSetup()

    try:
        # System requirements check
        if not setup.check_system_requirements():
            sys.exit(1)

        # Setup steps
        setup.create_directory_structure()
        setup.setup_environment_file()

        if not setup.install_python_dependencies():
            sys.exit(1)

        if not setup.setup_databases():
            sys.exit(1)

        if not args.skip_models and not args.minimal:
            setup.setup_ollama()

        setup.initialize_databases()

        if not args.minimal:
            setup.generate_test_data()

        setup.create_startup_scripts()

        if not args.skip_tests and not args.minimal:
            setup.run_tests()

        setup.print_setup_summary()

        logger.info("🎉 Setup completed successfully!")

    except KeyboardInterrupt:
        logger.info("🛑 Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()