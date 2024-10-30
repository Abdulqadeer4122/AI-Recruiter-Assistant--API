from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.params import Form

from ..schemas.schema import AnalysisResult,Questionnaire
from ..services import analyze_resume_service,analyze_personality

router = APIRouter()

# Mock job database

@router.post("/analyze_resume/", response_model=AnalysisResult)
async def analyze_resume(
    job_description: str = Form(...),  # Using Form to receive long text input
    file: UploadFile = File(...)  # File upload for the resume
):
    # Pass job_description and file to the service
    return await analyze_resume_service(file, job_description)

@router.post("/analyz_personality/")
async def personality_analyzer(questions: Questionnaire):

    # Assuming `questions` is a list of Question objects
    questions_data = [question.model_dump() for question in questions.questions]
    print(questions_data)
    return  analyze_personality(questions_data)