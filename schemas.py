# schemas.py - Pydantic Schemas for API
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class FileUploadResponse(BaseModel):
    success: bool
    message: str
    files: List[Dict[str, Any]]

class JobStatusResponse(BaseModel):
    job_id: int
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

class TableInfo(BaseModel):
    table_name: str
    created_at: datetime
    row_count: int
    column_count: int
    schema: Dict[str, Any]

class VisualizationRequest(BaseModel):
    chart_type: str = Field(..., description="Type of chart: bar, line, pie, scatter")
    x_axis: str = Field(..., description="Column name for X-axis")
    y_axis: Optional[str] = Field(None, description="Column name for Y-axis")

class DataInsight(BaseModel):
    type: str
    title: str
    content: str
    confidence: float
    generated_at: datetime

class DataQualityReport(BaseModel):
    table_name: str
    completeness_score: float
    consistency_score: float
    recommendations: List[str]
    generated_at: datetime
