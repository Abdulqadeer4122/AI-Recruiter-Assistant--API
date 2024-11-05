import tempfile
from fastapi import UploadFile, HTTPException
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from typing import List, Dict, Any # Ensure you have these imports
from typing import TypedDict, Optional
import json
import pdfplumber
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
# Set up the credentials for the new model
GROQ_API_KEY = "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"
GROQ_MODEL_ID = "llama-3.1-70b-versatile"

# Initialize SentenceTransformer model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = SentenceTransformerEmbeddings(model_name=model_name)

# Directory and file configuration
PDF_DIR = "/home/datics/data_pdf/"  # Replace with your PDF directory path
PDF_FILE_NAME = "policy_document.pdf"  # Replace with your PDF file name
VECTOR_STORE_PATH = "faiss_db"


# Create an instance of the Ollama model with the GROQ API key
llm = ChatGroq(model=GROQ_MODEL_ID, api_key=GROQ_API_KEY)


class ResumeState(TypedDict):
    resume_text: str
    job_description: str
    score: float
    is_recommended: bool
    detailed_analysis: str
    recommendation_status: str

def extract_score_from_response(response_text: str) -> float:
    """Extract numerical score from LLM response with robust error handling"""
    try:
        # Remove any non-numeric characters except decimal points
        cleaned_text = ''.join(char for char in response_text if char.isdigit() or char == '.')
        # Convert to float and ensure it's between 0 and 100
        score = float(cleaned_text)
        return min(max(score, 0), 100)
    except (ValueError, TypeError):
        print(f"Warning: Could not parse score from response: {response_text}")
        return 0.0
def calculate_match_score(state: ResumeState) -> ResumeState:
    """Calculate only the match score between resume and job description"""
    score_prompt = f"""As an expert resume analyzer, calculate a percentage match score (0-100) between this resume and job description.
    Consider only relevant skills, experience, and qualifications.

    Job Description: {state['job_description']}
    Resume Text: {state['resume_text']}

    Return only an integer between 0 and 100 representing the match percentage.
    """

    score_output = llm.invoke(score_prompt)
    try:
        score = extract_score_from_response(score_output.content)
        is_recommended = score >= 80
    except ValueError:
        score = 0
        is_recommended = False

    return {
        **state,
        "score": score,
        "is_recommended": is_recommended
    }


def perform_detailed_analysis(state: ResumeState) -> ResumeState:
    """Perform detailed analysis based on the calculated score"""

    analysis_prompt = f"""As an expert resume analyzer, provide a detailed analysis of this candidate's fit for the role.
    The candidate received a match score of {state['score']}% and is {"recommended" if state['is_recommended'] else "not recommended"}.

    Job Description: {state['job_description']}
    Resume Text: {state['resume_text']}

    Please provide a detailed analysis covering:
    1. Key matching qualifications and skills
    2. Missing or mismatched requirements
    3. Relevant experience alignment
    4. Areas where the candidate exceeds requirements
    5. Specific justification for the recommendation decision

    Structure your analysis in clear sections with specific examples from both the resume and job description.
    """

    analysis_output = llm.invoke(analysis_prompt)
    detailed_analysis = analysis_output.content.strip()

    recommendation_status = "Recommended" if state['is_recommended'] else "Not Recommended"

    return {
        **state,
        "detailed_analysis": detailed_analysis,
        "recommendation_status": recommendation_status
    }


def create_resume_workflow() -> StateGraph:
    """Create and return the resume analysis workflow"""
    workflow = StateGraph(ResumeState)

    # Add nodes
    workflow.add_node("calculate_score", calculate_match_score)
    workflow.add_node("perform_detailed_analysis", perform_detailed_analysis)

    # Add edges
    workflow.add_edge("calculate_score", "perform_detailed_analysis")
    workflow.add_edge("perform_detailed_analysis", END)

    # Set entry point
    workflow.set_entry_point("calculate_score")

    return workflow.compile()


def run_resume_analysis(resume_text: str, job_description: str) -> dict:
    """Run the resume analysis workflow with error handling"""
    try:
        graph = create_resume_workflow()

        initial_state = {
            "resume_text": resume_text,
            "job_description": job_description,
            "score": 0.0,
            "is_recommended": False,
            "detailed_analysis": "",  # Keep this as is for internal state
            "recommendation_status": ""
        }

        result = graph.invoke(initial_state)

        # Map the internal state names to your API response model names
        return {
            "score": result["score"],
            "recommendation_status": result["recommendation_status"],
            "description": result["detailed_analysis"]  # Here we map detailed_analysis to description
        }

    except Exception as e:
        print(f"Error running resume analysis workflow: {e}")
        return {
            "score": 0.0,
            "recommendation_status": "error",
            "description": f"Error analyzing resume: {str(e)}"
        }

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

def analyze_personality(questions_data):
    llm = ChatGroq(model=GROQ_MODEL_ID, api_key=GROQ_API_KEY)

    # Create a simplified data string focusing on questions and answers
    formatted_responses = "\n".join([
        f"Q{item['id']}: {item['question']} - Answer: {item['answer']}/5"
        for item in questions_data
    ])

    # Create a focused prompt for the LLM
    system_prompt = """You are a personality assessment analyzer. Analyze the given responses and calculate percentages for the Big Five personality traits:
    1. Openness - Questions about creativity, curiosity, and new experiences
    2. Conscientiousness - Questions about organization, planning, and discipline
    3. Extraversion - Questions about social interaction and energy
    4. Agreeableness - Questions about cooperation and consideration for others
    5. Neuroticism - Questions about stress sensitivity and emotional stability

    Return ONLY a JSON object with the five traits and their percentage scores. Example:
    {
        "Openness": 75,
        "Conscientiousness": 80,
        "Extraversion": 60,
        "Agreeableness": 85,
        "Neuroticism": 45
    }"""

    # Map relevant question IDs to traits for focused analysis
    trait_mapping = """
    Key questions for each trait:
    - Openness: 15, 16, 17, 18, 19 (creativity, curiosity, openness to experiences)
    - Conscientiousness: 27, 28, 29, 30 (discipline, organization, planning)
    - Extraversion: 8, 9, 10, 11, 12 (social interaction, energy)
    - Agreeableness: 20, 21, 22, 23, 24, 25, 26 (cooperation, consideration)
    - Neuroticism: 1, 2, 3, 4, 5, 6, 7 (emotional stability, stress response)
    """

    # Get analysis from LLM
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""
            Analyze these personality assessment responses:
            {formatted_responses}

            Question groupings for reference:
            {trait_mapping}

            Provide the personality trait percentages based on the answers.
        """)
    ]

    try:
        response = llm.invoke(messages)

        # Extract the JSON part from the response
        start_idx = response.content.find('{')
        end_idx = response.content.rfind('}') + 1

        # Make sure the response contains valid JSON
        json_string = response.content[start_idx:end_idx].strip()

        # Validate and load JSON
        if json_string:
            results = json.loads(json_string)
        else:
            raise ValueError("No valid JSON found in the response.")

        return results

    except json.JSONDecodeError as json_err:
        return f"JSON decoding error: {str(json_err)}"
    except Exception as e:
        return f"Error analyzing personality: {str(e)}"





#///////////////////////////

# Pydantic model for messages
import re


def minimal_cleaning(text):
    text = re.sub(r'(?<=\w)(?=[A-Z])', ' ', text)
    return ' '.join(text.split())


def process_pdf(pdf_file_path):
    pages = []
    with pdfplumber.open(pdf_file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            raw_text = page.extract_text()
            if raw_text:
                cleaned_text = minimal_cleaning(raw_text)
                pages.append({"page_number": i + 1, "content": cleaned_text})

    return pages


def get_chunks(pages):
    documents = "\n\n".join(page["content"] for page in pages)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ",", " "]
    )
    chunks = text_splitter.split_text(documents)
    return chunks


def create_vector_store(chunks):
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local("faiss_db")


def load_vector_store():
    return FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)


def get_response(messages):
    vector_store = load_vector_store()
    user_query=messages[-1]['content']
    question_embedding = embeddings.embed_query(user_query)
    relevant_docs = vector_store.similarity_search_by_vector(question_embedding, k=4)
    # Combine previous messages into context
    combined_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    context_with_metadata = []
    for doc in relevant_docs:
        page_num = doc.metadata.get("page_number", "unknown")
        content = getattr(doc, 'content', None) or getattr(doc, 'page_content', None) or getattr(doc, 'text', None)
        if content is not None:
            context_with_metadata.append(f"[Page {page_num}]: {content}")

    context = "\n\n".join(context_with_metadata)
    print(context)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert assistant specializing in interpreting and clarifying company policy documents.

            Your task is to analyze the provided context and respond to user questions with precision and clarity. Always reference the relevant sections or page numbers from the policy document to ensure your answers are well-supported.

            Please format your response as follows:
            1. **Direct Answer**: Provide a  answer to the user's question, directly addressing their inquiry.
            2. **Supporting Details**: Offer any additional context, explanations, or references to specific sections or pages within the policy document that substantiate your answer.

            **Context**:
            {context}
            """),
        ("human", "{combined_context}")
    ])

    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.invoke({"combined_context": combined_context,"context": context})

    return response["text"]

import os
# Initialize vector store at startup
def initialize_vector_store():
    pdf_file_path = os.path.join(PDF_DIR, PDF_FILE_NAME)

    # Check if the vector store exists
    if not os.path.exists(VECTOR_STORE_PATH):
        print("Vector store not found. Processing PDF...")

        # Process the PDF file
        pages = process_pdf(pdf_file_path)

        # Get chunks from the processed pages
        chunks = get_chunks(pages)

        # Create the vector store with the chunks
        create_vector_store(chunks)
    else:
        print("Vector store loaded successfully.")


# Call the initialization function on startup
initialize_vector_store()


