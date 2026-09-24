import csv
import os
from functools import lru_cache
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data', 'skill_training_data.csv')
MODEL = os.path.join(BASE, 'ml', 'skill_classifier.joblib')

def train_skill_classifier():
    texts, labels = [], []
    with open(DATA, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            texts.append(row['skill']); labels.append(row['category'])
    model = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2))),
                      ('classifier', LinearSVC())])
    model.fit(texts, labels)
    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    joblib.dump(model, MODEL)
    return model

@lru_cache(maxsize=1)
def load_skill_classifier():
    """Load once per process; train only when a clean checkout has no model."""
    return joblib.load(MODEL) if os.path.exists(MODEL) else train_skill_classifier()

def predict_skill_category(skill: str):
    return load_skill_classifier().predict([skill])[0]
