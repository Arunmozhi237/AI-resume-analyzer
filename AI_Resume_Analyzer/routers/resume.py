from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from services.extractor import extract_text
from services.preprocessor import preprocess
from services.skill_analyzer import analyze_skills
from services.matcher import match_resume_to_job
from services.rag import generate_suggestions
from services.rag_store import build_index
from models.schemas import ExtractionResponse, SkillAnalysisResponse, JobMatchResponse, SuggestionResponse, RAGIndexResponse
import json
import os

router = APIRouter(prefix="/resume", tags=["Resume"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILLS_DB_PATH = os.path.join(BASE_DIR, "data", "skills_db.json")
with open(SKILLS_DB_PATH, "r") as f:
    SKILLS_DB = json.load(f)

@router.post("/extract", response_model=ExtractionResponse)
async def extract_resume(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith((".pdf", ".docx")):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")

    file_bytes = await file.read()
    result = extract_text(file_bytes, file.filename)

    if result["status"].startswith("error"):
        raise HTTPException(status_code=422, detail=result["status"])

    return ExtractionResponse(
        filename=file.filename,
        file_type=result["file_type"],
        raw_text=result["text"],
        word_count=len(result["text"].split()),
        status=result["status"]
    )

@router.post("/analyze", response_model=SkillAnalysisResponse)
async def analyze_resume(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith((".pdf", ".docx")):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")

    file_bytes = await file.read()
    result = extract_text(file_bytes, file.filename)

    if result["status"].startswith("error"):
        raise HTTPException(status_code=422, detail=result["status"])

    preprocessed = preprocess(result["text"])
    skill_result = analyze_skills(preprocessed["cleaned_text"])

    return SkillAnalysisResponse(
        filename=file.filename,
        found_skills=skill_result["found_skills"],
        total_skills_found=skill_result["total_skills_found"],
        categories_found=skill_result["categories_found"],
        total_categories=skill_result["total_categories"],
        status="success"
    )

@router.post("/match", response_model=JobMatchResponse)
async def match_resume(
    file: UploadFile = File(...),
    job_description: str = Form(...)
):
    if not file.filename or not file.filename.lower().endswith((".pdf", ".docx")):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")

    file_bytes = await file.read()
    result = extract_text(file_bytes, file.filename)

    if result["status"].startswith("error"):
        raise HTTPException(status_code=422, detail=result["status"])

    preprocessed = preprocess(result["text"])
    skill_result = analyze_skills(preprocessed["cleaned_text"])
    all_resume_skills = []
    for skills in skill_result["found_skills"].values():
        all_resume_skills.extend(skills)

    match_result = match_resume_to_job(
        resume_text=preprocessed["cleaned_text"],
        job_description=job_description,
        resume_skills=all_resume_skills,
        skills_db=SKILLS_DB
    )

    return JobMatchResponse(
        filename=file.filename,
        match_score=match_result["match_score"],
        matching_skills=match_result["matching_skills"],
        missing_skills=match_result["missing_skills"],
        required_skills_in_jd=match_result["required_skills_in_jd"],
        recommendation=match_result["recommendation"],
        status="success"
    )

@router.post("/suggest", response_model=SuggestionResponse)
async def suggest_improvements(
    file: UploadFile = File(...),
    job_description: str = Form(...)
):
    if not file.filename or not file.filename.lower().endswith((".pdf", ".docx")):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")

    # Step 1: Extract
    file_bytes = await file.read()
    result = extract_text(file_bytes, file.filename)

    if result["status"].startswith("error"):
        raise HTTPException(status_code=422, detail=result["status"])

    # Step 2: Preprocess
    preprocessed = preprocess(result["text"])

    # Step 3: Skill analysis
    skill_result = analyze_skills(preprocessed["cleaned_text"])
    all_resume_skills = []
    for skills in skill_result["found_skills"].values():
        all_resume_skills.extend(skills)

    # Step 4: Match
    match_result = match_resume_to_job(
        resume_text=preprocessed["cleaned_text"],
        job_description=job_description,
        resume_skills=all_resume_skills,
        skills_db=SKILLS_DB
    )

    # Step 5: Generate RAG suggestions
    rag_result = generate_suggestions(
        resume_text=preprocessed["cleaned_text"],
        job_description=job_description,
        matching_skills=match_result["matching_skills"],
        missing_skills=match_result["missing_skills"],
        match_score=match_result["match_score"]
    )

    return SuggestionResponse(
        filename=file.filename,
        match_score=match_result["match_score"],
        matching_skills=match_result["matching_skills"],
        missing_skills=match_result["missing_skills"],
        recommendation=match_result["recommendation"],
        resume_suggestions=rag_result["resume_suggestions"],
        skill_gap_analysis=rag_result["skill_gap_analysis"],
        learning_roadmap=rag_result["learning_roadmap"],
        interview_questions=rag_result["interview_questions"],
        retrieved_context=rag_result["retrieved_context"],
        generation_mode=rag_result["generation_mode"],
        notice=rag_result["notice"],
        status="success"
    )

@router.post("/rag/reindex", response_model=RAGIndexResponse)
def reindex_knowledge_base():
    """Rebuild FAISS after editing data/career_knowledge.md."""
    return RAGIndexResponse(indexed_chunks=build_index(force=True), status="success")
