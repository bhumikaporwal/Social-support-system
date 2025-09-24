from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import uvicorn
import logging
from typing import List, Optional, Dict, Any
import uuid
import json
from datetime import datetime
import aiofiles
import os
from pathlib import Path

from src.database import get_db, init_databases
from src.models import (
    Application, Applicant, Document,
    ApplicationCreate, ApplicationResponse,
    ApplicantCreate, ApplicantResponse,
    DocumentResponse, ApplicationStatus
)
from src.agents.orchestrator import SocialSupportOrchestrator, WorkflowResult
from src.services.llm_client import SocialSupportChatbot, OllamaClient
from src.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="UAE Social Support API",
    description="AI-powered social support application processing system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Global variables for AI services
orchestrator: Optional[SocialSupportOrchestrator] = None
chatbot: Optional[SocialSupportChatbot] = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global orchestrator, chatbot

    try:
        # Initialize databases
        init_databases()
        logger.info("Databases initialized")

        # Initialize AI services
        llm_client = OllamaClient()
        orchestrator = SocialSupportOrchestrator(llm_client)
        chatbot = SocialSupportChatbot(llm_client)
        logger.info("AI services initialized")

        # Create upload directories
        os.makedirs(settings.upload_dir, exist_ok=True)
        os.makedirs(settings.temp_dir, exist_ok=True)

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    try:
        if chatbot and hasattr(chatbot.llm_client, 'close'):
            await chatbot.llm_client.close()
        logger.info("Services cleaned up")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "database": "connected",
            "ai_services": "active" if orchestrator else "inactive"
        }
    }

# Applicant endpoints
@app.post("/api/applicants", response_model=ApplicantResponse)
async def create_applicant(
    applicant: ApplicantCreate,
    db: Session = Depends(get_db)
):
    """Create a new applicant"""
    try:
        # Check if applicant already exists
        existing = db.query(Applicant).filter(
            Applicant.emirates_id == applicant.emirates_id
        ).first()

        if existing:
            raise HTTPException(
                status_code=400,
                detail="Applicant with this Emirates ID already exists"
            )

        # Create new applicant
        db_applicant = Applicant(**applicant.dict())
        db.add(db_applicant)
        db.commit()
        db.refresh(db_applicant)

        return db_applicant

    except Exception as e:
        logger.error(f"Error creating applicant: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/applicants/{applicant_id}", response_model=ApplicantResponse)
async def get_applicant(applicant_id: int, db: Session = Depends(get_db)):
    """Get applicant by ID"""
    applicant = db.query(Applicant).filter(Applicant.id == applicant_id).first()
    if not applicant:
        raise HTTPException(status_code=404, detail="Applicant not found")
    return applicant

# Application endpoints
@app.post("/api/applications", response_model=ApplicationResponse)
async def create_application(
    application: ApplicationCreate,
    db: Session = Depends(get_db)
):
    """Create a new application"""
    try:
        # Verify applicant exists
        applicant = db.query(Applicant).filter(
            Applicant.id == application.applicant_id
        ).first()

        if not applicant:
            raise HTTPException(status_code=404, detail="Applicant not found")

        # Generate application number
        app_number = f"APP-{datetime.now().year}-{uuid.uuid4().hex[:6].upper()}"

        # Create application
        db_application = Application(
            **application.dict(),
            application_number=app_number,
            status=ApplicationStatus.SUBMITTED
        )

        db.add(db_application)
        db.commit()
        db.refresh(db_application)

        return db_application

    except Exception as e:
        logger.error(f"Error creating application: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/applications/{application_id}", response_model=ApplicationResponse)
async def get_application(application_id: int, db: Session = Depends(get_db)):
    """Get application by ID"""
    application = db.query(Application).filter(
        Application.id == application_id
    ).first()

    if not application:
        raise HTTPException(status_code=404, detail="Application not found")

    return application

@app.get("/api/applications/number/{application_number}", response_model=ApplicationResponse)
async def get_application_by_number(
    application_number: str,
    db: Session = Depends(get_db)
):
    """Get application by application number"""
    application = db.query(Application).filter(
        Application.application_number == application_number
    ).first()

    if not application:
        raise HTTPException(status_code=404, detail="Application not found")

    return application

# Document upload endpoints
@app.post("/api/applications/{application_id}/documents")
async def upload_document(
    application_id: int,
    background_tasks: BackgroundTasks,
    document_type: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload document for an application"""
    try:
        # Verify application exists
        application = db.query(Application).filter(
            Application.id == application_id
        ).first()

        if not application:
            raise HTTPException(status_code=404, detail="Application not found")

        # Validate file type
        allowed_types = [
            "application/pdf",
            "image/jpeg", "image/jpg", "image/png",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ]

        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type"
            )

        # Generate unique filename
        file_extension = Path(file.filename).suffix
        unique_filename = f"{uuid.uuid4().hex}{file_extension}"
        file_path = Path(settings.upload_dir) / unique_filename

        # Save file
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)

        # Create document record
        document = Document(
            application_id=application_id,
            document_type=document_type,
            file_name=file.filename,
            file_path=str(file_path),
            file_size=len(content),
            mime_type=file.content_type,
            processing_status="pending"
        )

        db.add(document)
        db.commit()
        db.refresh(document)

        # Schedule background processing
        background_tasks.add_task(process_document_background, document.id)

        return {
            "document_id": document.id,
            "message": "Document uploaded successfully",
            "processing_status": "pending"
        }

    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_document_background(document_id: int):
    """Background task to process uploaded document"""
    # This would be implemented to process the document using the AI pipeline
    logger.info(f"Processing document {document_id} in background")

# Application processing endpoints
@app.post("/api/applications/{application_id}/process")
async def process_application(
    application_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Start processing an application"""
    try:
        # Get application with documents
        application = db.query(Application).filter(
            Application.id == application_id
        ).first()

        if not application:
            raise HTTPException(status_code=404, detail="Application not found")

        if application.status != ApplicationStatus.SUBMITTED:
            raise HTTPException(
                status_code=400,
                detail="Application is not in submitted status"
            )

        # Update status to processing
        application.status = ApplicationStatus.PROCESSING
        db.commit()

        # Schedule background processing
        background_tasks.add_task(process_application_workflow, application_id)

        return {
            "message": "Application processing started",
            "application_id": application_id,
            "status": "processing"
        }

    except Exception as e:
        logger.error(f"Error starting application processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_application_workflow(application_id: int):
    """Background task to process application through AI workflow"""
    global orchestrator

    if not orchestrator:
        logger.error("Orchestrator not initialized")
        return

    try:
        # Get database session
        from src.database import SessionLocal
        db = SessionLocal()

        try:
            # Get application and documents
            application = db.query(Application).filter(
                Application.id == application_id
            ).first()

            documents = db.query(Document).filter(
                Document.application_id == application_id
            ).all()

            # Prepare data for orchestrator
            workflow_data = {
                "application_id": application_id,
                "support_type": application.support_type,
                "documents": [
                    {
                        "document_type": doc.document_type,
                        "file_path": doc.file_path,
                        "file_name": doc.file_name
                    }
                    for doc in documents
                ]
            }

            # Process through workflow
            result: WorkflowResult = await orchestrator.process_application(workflow_data)

            # Update application with results
            application.status = ApplicationStatus.APPROVED if result.status.value == "completed" else ApplicationStatus.DECLINED
            application.processed_at = datetime.utcnow()
            application.agent_decisions = result.final_state
            application.confidence_score = result.confidence_score

            if result.final_state.get('final_recommendation'):
                recommendation = result.final_state['final_recommendation']
                application.recommendation = recommendation.get('decision')
                application.eligibility_score = recommendation.get('confidence', 0.5)

            db.commit()
            logger.info(f"Application {application_id} processed successfully")

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error processing application {application_id}: {e}")

# Chat endpoints
@app.post("/api/chat")
async def chat_with_assistant(
    message: Dict[str, Any]
):
    """Chat with AI assistant"""
    global chatbot

    if not chatbot:
        raise HTTPException(status_code=503, detail="Chatbot service not available")

    try:
        user_message = message.get("message", "")
        context = message.get("context", {})

        response = await chatbot.chat(user_message, context)

        return {
            "response": response,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/stream")
async def stream_chat(message: Dict[str, Any]):
    """Stream chat response"""
    # Implementation for streaming would go here
    # For now, return regular chat response
    return await chat_with_assistant(message)

# Analytics endpoints
@app.get("/api/analytics/dashboard")
async def get_dashboard_analytics(db: Session = Depends(get_db)):
    """Get dashboard analytics"""
    try:
        # Total applications
        total_apps = db.query(Application).count()

        # Applications by status
        status_counts = {}
        for status in ApplicationStatus:
            count = db.query(Application).filter(
                Application.status == status
            ).count()
            status_counts[status.value] = count

        # Recent applications (last 7 days)
        from datetime import timedelta
        recent_date = datetime.utcnow() - timedelta(days=7)
        recent_apps = db.query(Application).filter(
            Application.created_at >= recent_date
        ).count()

        return {
            "total_applications": total_apps,
            "status_distribution": status_counts,
            "recent_applications": recent_apps,
            "approval_rate": (
                status_counts.get("approved", 0) / max(total_apps, 1) * 100
            ),
            "average_processing_time": "3.2 minutes"  # Mock data
        }

    except Exception as e:
        logger.error(f"Analytics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Status check endpoint
@app.get("/api/status/{application_number}")
async def check_application_status(
    application_number: str,
    emirates_id: str,
    db: Session = Depends(get_db)
):
    """Check application status by application number and Emirates ID"""
    try:
        # Find application
        application = db.query(Application).join(Applicant).filter(
            Application.application_number == application_number,
            Applicant.emirates_id == emirates_id
        ).first()

        if not application:
            raise HTTPException(
                status_code=404,
                detail="Application not found or Emirates ID mismatch"
            )

        return {
            "application_number": application.application_number,
            "status": application.status,
            "created_at": application.created_at,
            "processed_at": application.processed_at,
            "recommendation": application.recommendation,
            "confidence_score": application.confidence_score,
            "support_type": application.support_type
        }

    except Exception as e:
        logger.error(f"Status check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.environment == "development"
    )