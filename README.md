# AI Resume Analyzer — NLP, ML, RAG and LLM

A FastAPI service that extracts PDF/DOCX resumes, identifies and categorizes
skills, calculates resume–job semantic similarity, and produces grounded career
guidance. It is designed as a clear demonstration of **NLTK + classical ML +
embeddings + retrieval-augmented generation + LLM** working together.

## Architecture

```text
Resume PDF/DOCX + Job Description
              |
     Text extraction (PyMuPDF / python-docx)
              |
 NLTK cleaning, tokenization, stopword removal, lemmatization
              |
 skills_db.json extraction -> TF-IDF + LinearSVC skill classification
              |
 SentenceTransformer embeddings -> cosine-similarity match score
              |
 job requirements + gaps -> chunked career_knowledge.md
              |                    |
              +---- FAISS vector search / relevant knowledge chunks
                                   |
                        Hugging Face LLM (or grounded fallback)
                                   |
 Resume suggestions, skill-gap analysis, 12-week roadmap, interview questions
```

### What makes this a real RAG pipeline?

`data/career_knowledge.md` is split into overlapping chunks. The project embeds
those chunks with `all-MiniLM-L6-v2`, stores the normalized vectors in a local
FAISS index, retrieves the most relevant chunks for each job/gap query, and
passes those chunks to the LLM as explicit evidence. The generated FAISS index
and metadata live in `data/vector_store/`, are rebuilt automatically, and are
not shipped in the ZIP.

## Clean setup

Prerequisite: Python 3.10 or later.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m nltk.downloader punkt_tab stopwords wordnet omw-1.4
copy .env.example .env
uvicorn main:app --reload
```

On macOS/Linux, replace `copy` with `cp`. Add a Hugging Face token to `.env` to
enable hosted generation. Without a token, `/resume/suggest` still returns a
useful local fallback grounded in the retrieved knowledge; it does not make a
network call. Open `http://127.0.0.1:8000/docs` for Swagger UI.

The first semantic-match or RAG request downloads the Sentence Transformer model
if it is not already cached. The first RAG request also builds the local FAISS
index. Both are normal first-run steps and no generated artifacts need to be
committed or distributed.

## Train the skill classifier

The skill classifier is a TF-IDF + LinearSVC pipeline. It auto-trains on the
starter labeled dataset the first time skill analysis runs, or train it before
starting the API:

```bash
python train_skill_model.py
```

Expand `data/skill_training_data.csv` with more accurately labeled skills before
using the classifier for serious evaluation. The generated `ml/*.joblib` model is
ignored so every clean distribution can reproduce it.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Health check |
| POST | `/resume/extract` | Extract raw PDF/DOCX text |
| POST | `/resume/analyze` | Extract and ML-categorize skills |
| POST | `/resume/match` | Semantic match score, matching and missing skills |
| POST | `/resume/suggest` | RAG-grounded advice, roadmap, and interview questions |
| POST | `/resume/rag/reindex` | Rebuild FAISS after editing the knowledge base |

`/resume/match` and `/resume/suggest` accept multipart form fields named `file`
and `job_description`. The suggestion response exposes `retrieved_context` and
`generation_mode`, making the grounding path inspectable.

## Customizing knowledge

Edit `data/career_knowledge.md` to add organization- or domain-specific career
guidance. Call `POST /resume/rag/reindex` afterwards, or remove the generated
`data/vector_store/` folder and let the next request rebuild it.

## Important implementation notes

- NLTK is used throughout; spaCy is not required.
- `services.preprocessor.preprocess()` supplies the dictionary API expected by
  the router. This fixes the former `preprocess`/`preprocess_text` import
  incompatibility.
- Semantic score is cosine similarity of pretrained sentence embeddings, not a
  hiring-outcome prediction or hiring decision.
- The LLM is instructed not to invent candidate claims. Review all generated
  advice before placing it on a resume.

## Distribution hygiene

Never include `.env`, virtual environments, `.git`, Python caches, FAISS index
files, or `ml/*.joblib` in a distributable archive. `.gitignore` covers those
local artifacts.
