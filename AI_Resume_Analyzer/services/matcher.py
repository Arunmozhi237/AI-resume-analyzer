from functools import lru_cache
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

@lru_cache(maxsize=1)
def get_embedding_model():
    """Lazy loading keeps lightweight endpoints responsive until embeddings are needed."""
    return SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text: str):
    """Convert text to embedding vector"""
    return get_embedding_model().encode([text], normalize_embeddings=True)[0]

def calculate_match_score(resume_text: str, job_description: str) -> float:
    """Calculate semantic similarity score between resume and JD"""
    resume_embedding = get_embedding(resume_text)
    jd_embedding = get_embedding(job_description)

    score = cosine_similarity(
        [resume_embedding],
        [jd_embedding]
    )[0][0]

    return round(float(score) * 100, 2)

def extract_jd_skills(jd_text: str, skills_db: dict) -> list:
    """Extract required skills from job description"""
    jd_lower = jd_text.lower()
    required_skills = []

    for category, skills in skills_db.items():
        for skill in skills:
            pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            if re.search(pattern, jd_lower):
                required_skills.append(skill)

    return required_skills

def find_matching_skills(resume_skills: list, jd_skills: list) -> dict:
    """Find matching and missing skills"""
    resume_skills_lower = [s.lower() for s in resume_skills]
    jd_skills_lower = [s.lower() for s in jd_skills]

    matching = [s for s in jd_skills if s.lower() in resume_skills_lower]
    missing = [s for s in jd_skills if s.lower() not in resume_skills_lower]

    return {
        "matching_skills": matching,
        "missing_skills": missing
    }

def get_recommendation(score: float) -> str:
    """Generate recommendation based on match score"""
    if score >= 80:
        return "Excellent match! Highly recommend applying."
    elif score >= 60:
        return "Good match! Consider applying with minor improvements."
    elif score >= 40:
        return "Moderate match. Upskill in missing areas before applying."
    else:
        return "Low match. Significant skill gaps need to be addressed."

def match_resume_to_job(
    resume_text: str,
    job_description: str,
    resume_skills: list,
    skills_db: dict
) -> dict:
    """Main matching function"""
    match_score = calculate_match_score(resume_text, job_description)
    jd_skills = extract_jd_skills(job_description, skills_db)
    skill_match = find_matching_skills(resume_skills, jd_skills)
    recommendation = get_recommendation(match_score)

    return {
        "match_score": match_score,
        "matching_skills": skill_match["matching_skills"],
        "missing_skills": skill_match["missing_skills"],
        "required_skills_in_jd": jd_skills,
        "recommendation": recommendation
    }
