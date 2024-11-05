from pydantic import BaseModel
from typing import Dict, Any, List

class AnalysisResult(BaseModel):
    score: float
    recommendation_status: str
    description: str
class Question(BaseModel):
    id: int
    question: str
    options: Dict[int, str]  # Using a dictionary to represent options
    answer: int

class Questionnaire(BaseModel):
    questions: List[Question]
class Message(BaseModel):
    role: str
    content: str


class MessageList(BaseModel):
    messages: List[Message]

