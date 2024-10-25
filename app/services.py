import tempfile
from fastapi import UploadFile, HTTPException
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END

# Set up the credentials for the new model
GROQ_API_KEY = "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"
GROQ_MODEL_ID = "llama-3.1-70b-versatile"

# Create an instance of the Ollama model with the GROQ API key
llm = ChatGroq(model=GROQ_MODEL_ID, api_key=GROQ_API_KEY)


# Define the state of the graph
class ResumeState(TypedDict):
    resume_text: Annotated[str, "The extracted resume text"]
    job_description: Annotated[str, "The job description"]
    score: Annotated[float, "Matching score for the resume"]
    description: Annotated[str, "Analysis of the resume"]
    recommendation_status: Annotated[str, "Whether the candidate is recommended or not"]


# Function to analyze the resume
def analyze_resume(state: ResumeState) -> ResumeState:
    score_prompt = f"""Act like a percentage calculator and provide only the integer value based on the given job details:
    Job Description: {state['job_description']}
    Resume Text: {state['resume_text']}
    Please analyze in detail and provide the percentage in integer with accuracy. only integer value return not anything else 
    """
    score_output = llm.invoke(score_prompt)
    score = float(score_output.content.strip())

    description_prompt = f"""Analyze the following resume for the given job description:
    Job Description: {state['job_description']}
    Resume Text: {state['resume_text']}

    Please do a detailed analysis of the resume's fit for the job and provide a short comparison of the job details and the candidate's resume.
    """
    description_output = llm.invoke(description_prompt)
    description = description_output.content.strip()

    recommendation_status = "Recommended" if score >= 85 else "Not Recommended"

    return {
        **state,  # Include all existing state keys
        "score": score,
        "description": description,
        "recommendation_status": recommendation_status
    }


# Create the graph
workflow = StateGraph(ResumeState)

# Add node for resume analysis
workflow.add_node("analyze_resume", analyze_resume)

# Add edges
workflow.add_edge("analyze_resume", END)

# Set the entry point
workflow.set_entry_point("analyze_resume")

# Compile the graph
graph = workflow.compile()


def run_resume_analysis(resume_text: str, job_description: str):
    try:
        result = graph.invoke({
            "resume_text": resume_text,
            "job_description": job_description,
            "score": 0.0,
            "description": "",
            "recommendation_status": ""
        })
        return {"score": result["score"], "recommendation_status":result['recommendation_status'], "description": result["description"]}
    except Exception as e:
        print(f"Error running workflow: {e}")
        return None


async def analyze_resume_service(file: UploadFile, job_description:str):
    # Load and process the resume based on file type
    if file.filename.endswith('.pdf'):
        with tempfile.NamedTemporaryFile(delete=False) as temp_pdf:
            temp_pdf.write(await file.read())
            temp_pdf_path = temp_pdf.name
        loader = PyPDFLoader(temp_pdf_path)
    elif file.filename.endswith('.docx'):
        with tempfile.NamedTemporaryFile(delete=False) as temp_docx:
            temp_docx.write(await file.read())
            temp_docx_path = temp_docx.name
        loader = Docx2txtLoader(temp_docx_path)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")

    # Extract text from the resume
    documents = loader.load()
    resume_text = " ".join([doc.page_content for doc in documents])
    response = run_resume_analysis(resume_text=resume_text, job_description=job_description)
    return response











    # # Create the prompt for the LLM
    # prompt = f"""
    # Analyze the following resume based on the job description provided:
    #
    # Job Title: {job['title']}
    # Job Description: {job['job_description']}
    # Key Responsibilities: {job['key_responsibilities']}
    # Qualifications Required: {job['qualifications_required']}
    #
    # Resume Text: {resume_text}
    #
    # Provide a matching score from 0 to 100 and a detailed analysis of the resume's fit for this job.
    # """
    #
    # result = llm.invoke(prompt)
    # # Extract score and description from the LLM response
    # try:
    #     analysis = result.content
    #     score = float(analysis.split('Score:')[1].split('\n')[0].strip().replace("/100**","").replace("**",''))
    #     description = analysis.split('Detailed Analysis:')[1].strip()
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail="Error processing LLM output: " + str(e))
    # # Provide recommendation based on the score
    # if score < 80:
    #     recommendation = "Not recommended. " + description
    # else:
    #     recommendation = "Recommended. " + description

