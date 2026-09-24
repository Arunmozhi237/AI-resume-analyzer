from fastapi import FastAPI
from routers import resume

app = FastAPI(
    title="AI Resume Analyzer",
    description="NLTK, ML, semantic matching, FAISS RAG, and Hugging Face LLM resume analysis",
    version="2.0.0"
)

app.include_router(resume.router)

@app.get("/")
def root():
    return {"message": "Resume Analyzer API is running!"}
