from pydantic import BaseModel
from typing import List, Dict, Optional

class ExtractionResponse(BaseModel):
    filename: str
    file_type: str
    raw_text: str
    word_count: int
    status: str

class PreprocessResponse(BaseModel):
    cleaned_text: str
    total_tokens: int
    total_sentences: int
    tokens_without_stopwords: List[str]
    lemmas: List[str]

class SkillAnalysisResponse(BaseModel):
    filename: str
    found_skills: Dict[str, List[str]]
    total_skills_found: int
    categories_found: List[str]
    total_categories: int
    status: str

class JobMatchResponse(BaseModel):
    filename: str
    match_score: float
    matching_skills: List[str]
    missing_skills: List[str]
    required_skills_in_jd: List[str]
    recommendation: str
    status: str

class SuggestionResponse(BaseModel):
    filename: str
    match_score: float
    matching_skills: List[str]
    missing_skills: List[str]
    recommendation: str
    resume_suggestions: List[str]
    skill_gap_analysis: List[str]
    learning_roadmap: List[str]
    interview_questions: List[str]
    retrieved_context: List[Dict]
    generation_mode: str
    notice: Optional[str] = None
    status: str

class RAGIndexResponse(BaseModel):
    indexed_chunks: int
    status: str
