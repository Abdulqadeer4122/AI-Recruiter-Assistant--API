from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.params import Form

from ..schemas.schema import AnalysisResult,JobDescription
from ..services import analyze_resume_service

router = APIRouter()

# Mock job database

@router.post("/analyze_resume/", response_model=AnalysisResult)
async def analyze_resume(
    job_description: str = Form(...),  # Using Form to receive long text input
    file: UploadFile = File(...)  # File upload for the resume
):
    # Pass job_description and file to the service
    return await analyze_resume_service(file, job_description)