from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from enum import Enum
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

Base = declarative_base()

class ApplicationStatus(str, Enum):
    SUBMITTED = "submitted"
    PROCESSING = "processing"
    APPROVED = "approved"
    DECLINED = "declined"
    PENDING_REVIEW = "pending_review"

class SupportType(str, Enum):
    FINANCIAL = "financial"
    ECONOMIC_ENABLEMENT = "economic_enablement"
    BOTH = "both"

class Application(Base):
    __tablename__ = "applications"

    id = Column(Integer, primary_key=True, index=True)
    applicant_id = Column(Integer, ForeignKey("applicants.id"))
    application_number = Column(String, unique=True, index=True)
    status = Column(String, default=ApplicationStatus.SUBMITTED)
    support_type = Column(String)

    # Financial Information
    monthly_income = Column(Float)
    employment_status = Column(String)
    family_size = Column(Integer)
    dependents = Column(Integer)
    assets_value = Column(Float)
    liabilities_value = Column(Float)

    # Assessment Results
    eligibility_score = Column(Float)
    risk_score = Column(Float)
    recommendation = Column(String)
    confidence_score = Column(Float)

    # AI Processing
    extracted_data = Column(JSON)
    validation_results = Column(JSON)
    agent_decisions = Column(JSON)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    processed_at = Column(DateTime)

    # Relationships
    applicant = relationship("Applicant", back_populates="applications")
    documents = relationship("Document", back_populates="application")

class Applicant(Base):
    __tablename__ = "applicants"

    id = Column(Integer, primary_key=True, index=True)
    emirates_id = Column(String, unique=True, index=True)
    first_name = Column(String)
    last_name = Column(String)
    date_of_birth = Column(DateTime)
    nationality = Column(String)
    gender = Column(String)

    # Contact Information
    email = Column(String, unique=True, index=True)
    phone = Column(String)
    address = Column(Text)
    emirate = Column(String)

    # Profile
    education_level = Column(String)
    marital_status = Column(String)
    profession = Column(String)

    # Verification
    is_verified = Column(Boolean, default=False)
    kyc_status = Column(String, default="pending")

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    applications = relationship("Application", back_populates="applicant")

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    application_id = Column(Integer, ForeignKey("applications.id"))
    document_type = Column(String)  # bank_statement, emirates_id, resume, assets_liabilities, credit_report
    file_name = Column(String)
    file_path = Column(String)
    file_size = Column(Integer)
    mime_type = Column(String)

    # Processing Results
    extracted_text = Column(Text)
    extracted_data = Column(JSON)
    processing_status = Column(String, default="pending")
    confidence_score = Column(Float)

    # Timestamps
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime)

    # Relationships
    application = relationship("Application", back_populates="documents")

# Pydantic Models for API

class ApplicantCreate(BaseModel):
    emirates_id: str
    first_name: str
    last_name: str
    date_of_birth: datetime
    nationality: str
    gender: str
    email: str
    phone: str
    address: str
    emirate: str
    education_level: Optional[str] = None
    marital_status: Optional[str] = None
    profession: Optional[str] = None

class ApplicantResponse(BaseModel):
    id: int
    emirates_id: str
    first_name: str
    last_name: str
    email: str
    phone: str
    is_verified: bool
    kyc_status: str
    created_at: datetime

    class Config:
        from_attributes = True

class ApplicationCreate(BaseModel):
    applicant_id: int
    support_type: SupportType
    monthly_income: Optional[float] = None
    employment_status: Optional[str] = None
    family_size: Optional[int] = None
    dependents: Optional[int] = None

class ApplicationResponse(BaseModel):
    id: int
    application_number: str
    status: ApplicationStatus
    support_type: str
    monthly_income: Optional[float]
    employment_status: Optional[str]
    family_size: Optional[int]
    eligibility_score: Optional[float]
    recommendation: Optional[str]
    confidence_score: Optional[float]
    created_at: datetime

    class Config:
        from_attributes = True

class DocumentUpload(BaseModel):
    document_type: str
    file_name: str

class DocumentResponse(BaseModel):
    id: int
    document_type: str
    file_name: str
    processing_status: str
    confidence_score: Optional[float]
    uploaded_at: datetime

    class Config:
        from_attributes = True