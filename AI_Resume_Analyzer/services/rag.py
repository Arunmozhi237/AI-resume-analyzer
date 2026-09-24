"""Grounded LLM generation for resume analysis using a local FAISS retriever."""
import json
import os

import requests
from dotenv import load_dotenv

from services.rag_store import retrieve

load_dotenv()
API_URL = os.getenv("HF_API_URL", "https://router.huggingface.co/hf-inference/v1/chat/completions")
HF_MODEL = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")


def build_prompt(resume_text, job_description, matching_skills, missing_skills, match_score, context):
    """Make retrieved evidence explicit and prohibit invented candidate facts."""
    evidence = "\n\n".join(f"[Knowledge {item['id']}] {item['text']}" for item in context)
    return f"""You are a careful career coach. Use only candidate/job facts and retrieved knowledge.
Do not invent experience, credentials, achievements, or missing skills.

RETRIEVED KNOWLEDGE:
{evidence}

RESUME: {resume_text[:1800]}
JOB DESCRIPTION: {job_description[:1400]}
MATCH SCORE: {match_score}%
MATCHING SKILLS: {', '.join(matching_skills) or 'None identified'}
MISSING SKILLS: {', '.join(missing_skills) or 'None identified'}

Return valid JSON only with keys resume_suggestions (3 strings), skill_gap_analysis
(strings), learning_roadmap (3 strings: weeks 1-4, 5-8, 9-12), and
interview_questions (4 strings)."""


def _fallback(missing_skills, matching_skills):
    """Grounded response when a token/model is unavailable."""
    targets = missing_skills[:3] or ["the highest-priority job requirements"]
    return {
        "resume_suggestions": [
            "Add truthful project bullets that connect relevant technologies to measurable outcomes.",
            "Mirror job-description terminology only where it accurately reflects your experience.",
            "Prioritize your strongest matching skills near the top of the resume.",
        ],
        "skill_gap_analysis": [f"Prioritize evidence of: {', '.join(targets)}.",
                               f"Current matching evidence: {', '.join(matching_skills[:5]) or 'none identified'}."],
        "learning_roadmap": [
            f"Weeks 1-4: learn fundamentals of {', '.join(targets)} and complete guided practice.",
            "Weeks 5-8: build a small documented project applying the target skills.",
            "Weeks 9-12: test, polish, document trade-offs, and practise interviews.",
        ],
        "interview_questions": [
            "Describe a project where you used your most relevant technical skill.",
            "How did you test or evaluate the quality of that work?",
            "What trade-off did you make, and why?",
            f"How would you approach learning and applying {targets[0]} in this role?",
        ],
    }


def query_llm(prompt):
    """Use Hugging Face's OpenAI-compatible API; return a reason on failure."""
    token = os.getenv("HF_TOKEN")
    if not token or token == "your_huggingface_token_here":
        return None, "HF_TOKEN is not configured; returned the local grounded fallback."
    try:
        response = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"model": HF_MODEL, "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 900, "temperature": 0.3,
                  "response_format": {"type": "json_object"}}, timeout=60,
        )
        response.raise_for_status()
        parsed = json.loads(response.json()["choices"][0]["message"]["content"])
        expected = {"resume_suggestions", "skill_gap_analysis", "learning_roadmap", "interview_questions"}
        if not expected.issubset(parsed):
            raise ValueError("LLM response is missing expected fields")
        return parsed, None
    except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError) as exc:
        return None, f"LLM unavailable ({exc}); returned the local grounded fallback."


def generate_suggestions(resume_text, job_description, matching_skills, missing_skills, match_score):
    """Retrieve semantic knowledge chunks, then ground the LLM on that context."""
    context = retrieve(f"{job_description}\nMissing skills: {', '.join(missing_skills)}")
    result, notice = query_llm(build_prompt(
        resume_text, job_description, matching_skills, missing_skills, match_score, context
    ))
    return {
        **(result or _fallback(missing_skills, matching_skills)),
        "retrieved_context": context,
        "generation_mode": "huggingface_llm" if result else "local_grounded_fallback",
        "notice": notice,
    }
