import json, os, re
from services.ml_skill_classifier import predict_skill_category

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(BASE, 'data', 'skills_db.json'), encoding='utf-8') as f:
    SKILLS_DB = json.load(f)

def extract_skills(text: str):
    found=[]; low=text.lower()
    for skill in {s for vals in SKILLS_DB.values() for s in vals}:
        if re.search(r'(?<!\w)'+re.escape(skill.lower())+r'(?!\w)', low): found.append(skill)
    return sorted(set(found))

def categorize_skills(skills):
    out={}
    for skill in skills:
        out.setdefault(predict_skill_category(skill), []).append(skill)
    return out

def analyze_skills(text: str):
    skills=extract_skills(text); cats=categorize_skills(skills)
    return {'found_skills':cats,'total_skills_found':len(skills),
            'categories_found':sorted(cats),'total_categories':len(cats),'status':'success'}
