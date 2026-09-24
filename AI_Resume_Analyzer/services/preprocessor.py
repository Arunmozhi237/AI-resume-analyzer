import re
from functools import lru_cache
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def clean_text(text: str) -> str:
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\b[\w.-]+@[\w.-]+\.\w+\b', ' ', text)
    text = re.sub(r'\+?\d[\d\s().-]{7,}\d', ' ', text)
    text = re.sub(r'[^A-Za-z0-9+#.\-/ ]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

@lru_cache(maxsize=1)
def _nltk_resources():
    """Load NLTK resources once and give a useful setup error when absent."""
    try:
        return set(stopwords.words("english")), WordNetLemmatizer()
    except LookupError as exc:
        raise RuntimeError(
            "NLTK data is missing. Run: python -m nltk.downloader punkt_tab stopwords wordnet omw-1.4"
        ) from exc


def preprocess(text: str) -> dict:
    """Clean, tokenize, remove stopwords, and lemmatize resume/JD text.

    This dictionary interface is intentionally used by the FastAPI router.  It
    replaces the old, incompatible ``preprocess_text``-only implementation.
    """
    cleaned = clean_text(text)
    stop_words, lemmatizer = _nltk_resources()
    tokens = word_tokenize(cleaned.lower())
    retained = [
        token for token in tokens
        if token not in stop_words and re.search(r"[a-zA-Z0-9]", token)
    ]
    lemmas = [lemmatizer.lemmatize(token) for token in retained]
    return {
        "cleaned_text": " ".join(lemmas),
        "total_tokens": len(tokens),
        "total_sentences": len(re.findall(r"[.!?]+", text)),
        "tokens_without_stopwords": retained,
        "lemmas": lemmas,
    }


def preprocess_text(text: str) -> str:
    """Backward-compatible shortcut for callers that only need cleaned text."""
    return preprocess(text)["cleaned_text"]
