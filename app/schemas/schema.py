from pydantic import BaseModel

class AnalysisResult(BaseModel):
    score: float
    recommendation_status: str
    description: str
class JobDescription(BaseModel):
    job_description: str

