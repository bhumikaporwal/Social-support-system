# Social Support System

A comprehensive AI-powered social support application that helps process applications for social services, perform eligibility assessments, and provide intelligent document analysis using multiple AI agents and machine learning models.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation Guide for Windows](#installation-guide-for-windows)
- [Environment Setup](#environment-setup)
- [Database Setup](#database-setup)
- [Running the Application](#running-the-application)
- [Docker Deployment](#docker-deployment)
- [ML Model Training](#ml-model-training)
- [API Documentation](#api-documentation)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)

## Prerequisites

Before installing the Social Support System on Windows, ensure you have the following installed:

### Required Software

1. **Python 3.8 or higher**
   - Download from [python.org](https://www.python.org/downloads/windows/)
   - **Important**: During installation, check "Add Python to PATH"
   - Verify installation: Open Command Prompt and run `python --version`

2. **Git for Windows**
   - Download from [git-scm.com](https://git-scm.com/download/win)
   - Use default installation settings
   - Verify installation: `git --version`

3. **Docker Desktop for Windows** (Optional - for containerized deployment)
   - Download from [docker.com](https://docs.docker.com/desktop/install/windows-install/)
   - Requires Windows 10/11 Pro, Enterprise, or Education
   - Enable WSL 2 integration if prompted

### Database Prerequisites (if not using Docker)

4. **PostgreSQL 13+**
   - Download from [postgresql.org](https://www.postgresql.org/download/windows/)
   - During installation, remember the password you set for the postgres user
   - Add PostgreSQL bin directory to PATH

5. **MongoDB Community Server**
   - Download from [mongodb.com](https://www.mongodb.com/try/download/community)
   - Install as a Windows Service (recommended)

6. **Redis** (Optional - for caching)
   - Download from [github.com/microsoftarchive/redis/releases](https://github.com/microsoftarchive/redis/releases)
   - Or use Docker: `docker run -d -p 6379:6379 redis:alpine`

### Optional but Recommended

7. **Ollama for Windows** (for local LLM processing)
   - Download from [ollama.ai](https://ollama.ai/download/windows)
   - Required for AI chatbot functionality

8. **Tesseract OCR** (for document OCR)
   - Download from [github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
   - Add to PATH: `C:\Program Files\Tesseract-OCR`

## Installation Guide for Windows

### Step 1: Clone the Repository

Open Command Prompt or PowerShell as Administrator and run:

```cmd
git clone https://github.com/your-username/Social-support-system.git
cd Social-support-system
```

### Step 2: Create Python Virtual Environment

```cmd
python -m venv venv
venv\Scripts\activate
```

**Note**: You'll need to activate the virtual environment every time you work with the project:
```cmd
venv\Scripts\activate
```

### Step 3: Install Python Dependencies

```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you encounter any installation errors, try installing packages individually or use:
```cmd
pip install --no-cache-dir -r requirements.txt
```

## Environment Setup

### Step 1: Create Environment File

Copy the example environment file:
```cmd
copy .env.example .env
```

### Step 2: Configure Environment Variables

Edit the `.env` file with your preferred text editor (Notepad, VS Code, etc.):

```env
# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=social_support_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_postgres_password

MONGODB_URL=mongodb://localhost:27017
MONGODB_DB=social_support_docs

QDRANT_HOST=localhost
QDRANT_PORT=6333

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

# Ollama Configuration (for local LLM)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2:7b-chat

# OpenAI Configuration (alternative to Ollama)
OPENAI_API_KEY=your_openai_api_key_here

# Application Configuration
API_HOST=localhost
API_PORT=8000
STREAMLIT_PORT=8501
SECRET_KEY=your-secret-key-here-change-this-in-production
ENVIRONMENT=development

# File Upload Configuration
MAX_FILE_SIZE=50MB
UPLOAD_DIR=./data/uploads
TEMP_DIR=./data/temp
```

## Database Setup

### Option 1: Docker Database Setup (Recommended)

If you have Docker Desktop installed, this is the easiest option:

```cmd
# Start all databases at once
docker-compose up postgres mongodb qdrant neo4j redis -d

# Or start them individually
docker run -d --name postgres -e POSTGRES_DB=social_support_db -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=your_password -p 5432:5432 postgres:13

docker run -d --name mongodb -p 27017:27017 mongo:5.0

docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/your_password neo4j:5.0

docker run -d --name redis -p 6379:6379 redis:alpine
```

### Option 2: Manual Database Setup on Windows

#### PostgreSQL Setup
1. Download and install PostgreSQL from [postgresql.org](https://www.postgresql.org/download/windows/)
2. During installation, remember the password for the postgres user
3. Open pgAdmin or psql command line
4. Create database:
   ```sql
   CREATE DATABASE social_support_db;
   ```

#### MongoDB Setup
1. Install MongoDB Community Server
2. MongoDB will start automatically as a Windows service
3. No additional setup required - databases will be created automatically

#### Neo4j Setup (Optional)
1. Download Neo4j Desktop from [neo4j.com](https://neo4j.com/download/)
2. Create a new project and database
3. Set password and update `.env` file
4. Access via browser: http://localhost:7474

#### Qdrant Setup (Optional - for vector search)
1. Use Docker: `docker run -d -p 6333:6333 qdrant/qdrant`
2. Or download from [qdrant.tech](https://qdrant.tech/documentation/quick_start/)

### Ollama Setup (for AI Chatbot)

1. **Download and Install Ollama**
   - Visit [ollama.ai](https://ollama.ai/download/windows)
   - Download the Windows installer
   - Run the installer

2. **Download AI Models**
   ```cmd
   ollama pull llama2:7b-chat
   ollama pull mistral:7b
   ```

3. **Verify Installation**
   ```cmd
   ollama list
   curl http://localhost:11434/api/tags
   ```

## Running the Application

### Method 1: Using the Application Runner (Recommended)

The application includes a built-in runner script that handles both backend and frontend:

```cmd
# Activate virtual environment (do this every time you start working)
venv\Scripts\activate

# Run full application (API + Frontend)
python run_app.py

# Alternative modes:
python run_app.py --mode api          # API only
python run_app.py --mode frontend     # Frontend only
python run_app.py --mode setup        # Setup environment only
python run_app.py --mode data         # Generate test data only
```

The application will be available at:
- **Frontend (Streamlit)**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### Method 2: Manual Startup

If you prefer to start services separately:

#### Start Backend API
```cmd
venv\Scripts\activate
uvicorn src.api.main:app --host localhost --port 8000 --reload
```

#### Start Frontend (in a new Command Prompt window)
```cmd
cd C:\Users\devan\source\repos\socialAI\Social-support-system
venv\Scripts\activate
streamlit run src/frontend/streamlit_app.py --server.port 8501
```

### Using the Web Interface

1. **Access the Application**
   - **Frontend**: http://localhost:8501
   - **API Documentation**: http://localhost:8000/docs
   - **API Redoc**: http://localhost:8000/redoc

2. **Submit New Application**
   - Navigate to "New Application" page
   - Fill in personal information
   - Upload required documents (PDF, images)
   - Review and submit

3. **Track Application Status**
   - Use "Application Status" page
   - Enter Application ID and identifying information
   - View real-time processing status

4. **AI Assistant** (if Ollama is installed)
   - Access via "AI Assistant" page
   - Ask questions about eligibility, documents, process
   - Get personalized guidance

## Docker Deployment

### Full Stack Deployment with Docker

If you have Docker Desktop installed, you can run the entire application stack:

1. **Build and start all services**:
   ```cmd
   docker-compose up --build
   ```

2. **Run in detached mode**:
   ```cmd
   docker-compose up -d
   ```

3. **View logs**:
   ```cmd
   docker-compose logs -f
   ```

4. **Stop all services**:
   ```cmd
   docker-compose down
   ```

### Accessing Services

When using Docker deployment:
- **Frontend**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **PostgreSQL**: localhost:5432
- **MongoDB**: localhost:27017
- **Neo4j Browser**: http://localhost:7474
- **Qdrant Dashboard**: http://localhost:6333/dashboard

## ML Model Training

### Generate Training Data and Train Models

```cmd
# Activate virtual environment
venv\Scripts\activate

# Generate synthetic training data and train ML models
python train_model.py
```

This will:
1. Generate 800 synthetic application records
2. Train multiple ML models for eligibility prediction
3. Save the best performing model to the `models/` directory
4. Display model performance metrics

### Alternative Training Script
```cmd
python train_and_test_model.py
```

## API Documentation

Once the application is running, you can access:

- **Interactive API Docs (Swagger)**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc

### Key API Endpoints

- `POST /applications/` - Submit new application
- `GET /applications/{id}` - Get application by ID
- `POST /applications/{id}/process` - Process application through AI agents
- `POST /documents/upload` - Upload supporting documents
- `GET /health` - Health check endpoint

## Troubleshooting

### Common Windows Issues

#### Python/Pip Issues
```cmd
# If pip is not recognized
python -m pip install --upgrade pip

# If Python modules not found
set PYTHONPATH=%CD%
python run_app.py
```

#### Port Already in Use
```cmd
# Find process using port 8000
netstat -ano | findstr :8000

# Kill process (replace PID with actual process ID)
taskkill /PID <PID> /F
```

#### Virtual Environment Issues
```cmd
# If activation script not working
venv\Scripts\activate.bat

# Or use PowerShell
venv\Scripts\Activate.ps1
```

#### Database Connection Issues
1. **PostgreSQL**: Ensure service is running
   ```cmd
   # Open Services management console
   services.msc
   # Look for "postgresql-x64-13" service
   ```

2. **MongoDB**: Ensure service is running
   ```cmd
   net start MongoDB
   ```

#### Docker Issues
1. **Docker not starting**: Ensure Docker Desktop is running
2. **Port conflicts**: Stop conflicting services or change ports in `docker-compose.yml`
3. **Permission denied**: Run Command Prompt as Administrator

#### Streamlit Issues
```cmd
# If Streamlit won't start
streamlit --version
pip install --upgrade streamlit

# Clear Streamlit cache
streamlit cache clear
```

#### Ollama Issues
```cmd
# If Ollama models aren't working
ollama serve
ollama list
ollama pull llama2:7b-chat
```

### Logs and Debugging

- Application logs: Check `logs/` directory
- Docker logs: `docker-compose logs [service_name]`
- Streamlit logs: Visible in terminal where Streamlit is running
- FastAPI logs: Visible in uvicorn terminal

### Performance Optimization

#### For Windows Systems
1. **Increase virtual memory** if running multiple databases
2. **Disable Windows Defender** real-time scanning for project folder (temporary)
3. **Close unnecessary applications** to free up memory
4. **Use SSD storage** for better database performance

### Getting Help

If you encounter issues:
1. Check the [Troubleshooting](#troubleshooting) section above
2. Review logs in the `logs/` directory
3. Ensure all prerequisites are properly installed
4. Verify database connections and services are running
5. Try running `python run_app.py --mode setup` to reset the environment

## Project Structure

```
Social-support-system/
├── src/                          # Source code
│   ├── agents/                   # AI agents
│   │   ├── eligibility_agent.py  # Eligibility assessment
│   │   ├── document_extraction_agent.py
│   │   ├── economic_enablement_agent.py
│   │   └── orchestrator.py       # Agent coordination
│   ├── api/                      # FastAPI backend
│   │   └── main.py               # API endpoints
│   ├── frontend/                 # Streamlit frontend
│   │   └── streamlit_app.py      # Web interface
│   ├── ml/                       # Machine learning
│   │   └── model_trainer.py      # ML model training
│   ├── models/                   # Data models
│   ├── services/                 # Business logic
│   ├── database/                 # Database connections
│   └── config/                   # Configuration
├── data/                         # Data storage
│   ├── synthetic/                # Generated test data
│   ├── uploads/                  # Uploaded files
│   └── temp/                     # Temporary files
├── models/                       # Trained ML models
├── tests/                        # Unit tests
├── docs/                         # Documentation
├── config/                       # Configuration files
├── docker-compose.yml            # Docker services
├── requirements.txt              # Python dependencies
├── run_app.py                    # Application runner
├── train_model.py                # Model training script
└── README.md                     # This file
```

## Additional Features

### Ollama Setup (Local LLM for AI Features)

1. **Download Ollama for Windows**:
   - Visit [ollama.ai](https://ollama.ai/download/windows)
   - Download Windows installer

2. **Install and pull models**:
   ```cmd
   ollama pull llama2:7b-chat
   ollama pull mistral:7b
   ```

3. **Verify installation**:
   ```cmd
   ollama list
   ```

### Development Tools

#### VS Code Extensions (Recommended)
- Python
- Docker
- PostgreSQL Explorer
- MongoDB for VS Code
- Thunder Client (for API testing)

#### Jupyter Notebook Support
```cmd
pip install jupyter
jupyter notebook
```

## Quick Start Summary

For a new user on Windows, here's the essential steps:

1. **Install Prerequisites**: Python 3.8+, Git, Docker (optional)
2. **Clone Repository**: `git clone <repo-url> && cd Social-support-system`
3. **Setup Virtual Environment**: `python -m venv venv && venv\Scripts\activate`
4. **Install Dependencies**: `pip install -r requirements.txt`
5. **Setup Environment**: `copy .env.example .env` (edit with your settings)
6. **Start Databases**: `docker-compose up postgres mongodb -d` (or install manually)
7. **Run Application**: `python run_app.py`
8. **Access Application**: http://localhost:8501

## Support

For issues and support:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review logs in the `logs/` directory
3. Ensure all prerequisites are properly installed
4. Verify database connections and services are running

## License

This project is licensed under the MIT License - see the LICENSE file for details.