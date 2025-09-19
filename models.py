# models.py - SQLAlchemy Models
from sqlalchemy import Column, Integer, String, DateTime, JSON, Text, Float, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base

class FileUpload(Base):
    """Model for tracking uploaded files"""
    __tablename__ = "file_uploads"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(Text, nullable=False)
    file_size = Column(Integer)
    file_type = Column(String(100))
    status = Column(String(50), default="uploaded")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    processing_jobs = relationship("ProcessingJob", back_populates="file_upload")

class ProcessingJob(Base):
    """Model for tracking AI processing jobs"""
    __tablename__ = "processing_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    file_id = Column(Integer, ForeignKey("file_uploads.id"))
    job_type = Column(String(100), nullable=False)  # genai_processing, etl, etc.
    status = Column(String(50), default="queued")  # queued, processing, completed, failed
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    result = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Relationships
    file_upload = relationship("FileUpload", back_populates="processing_jobs")

class DataTable(Base):
    """Model for tracking data warehouse tables"""
    __tablename__ = "data_tables"
    
    id = Column(Integer, primary_key=True, index=True)
    table_name = Column(String(255), unique=True, nullable=False)
    original_filename = Column(String(255))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    row_count = Column(Integer, default=0)
    column_count = Column(Integer, default=0)
    schema = Column(JSON)  # Store column definitions
    ai_metadata = Column(JSON)  # Store AI-generated insights
    is_active = Column(Boolean, default=True)

class DataQualityReport(Base):
    """Model for storing data quality assessments"""
    __tablename__ = "data_quality_reports"
    
    id = Column(Integer, primary_key=True, index=True)
    table_name = Column(String(255), nullable=False)
    completeness_score = Column(Float)
    consistency_score = Column(Float)
    accuracy_score = Column(Float)
    anomalies_detected = Column(JSON)
    recommendations = Column(JSON)
    generated_at = Column(DateTime(timezone=True), server_default=func.now())

class AIInsight(Base):
    """Model for storing AI-generated insights"""
    __tablename__ = "ai_insights"
    
    id = Column(Integer, primary_key=True, index=True)
    table_name = Column(String(255), nullable=False)
    insight_type = Column(String(100))  # summary, pattern, anomaly, recommendation
    title = Column(String(255))
    content = Column(Text)
    confidence_score = Column(Float)
    metadata = Column(JSON)
    generated_at = Column(DateTime(timezone=True), server_default=func.now())
