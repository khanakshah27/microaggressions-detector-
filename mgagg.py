# microagg.py

import pandas as pd
import numpy as np
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline as hf_pipeline
import re
import string

# Downloads
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text_textblob(text):
    try:
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        blob = TextBlob(text)
        return " ".join([
            lemmatizer.lemmatize(word) for word in blob.words
            if word not in stop_words and len(word) > 1
        ])
    except:
        return text.lower()

class TextBlobPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        return [preprocess_text_textblob(text) for text in X]

def train_and_save_pipeline():
    df = pd.read_csv("micro_agg.csv", encoding='ISO-8859-1')
    df['ptext'] = df['speech'].apply(preprocess_text_textblob)

    X = df['speech']
    y = df['label']

    pipeline = Pipeline([
        ('textblob', TextBlobPreprocessor()),
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95, sublinear_tf=True)),
        ('sgd', SGDClassifier(loss='log_loss', penalty='elasticnet', alpha=1e-5,
                              max_iter=1000, class_weight='balanced', random_state=42))
    ])

    pipeline.fit(X, y)
    joblib.dump(pipeline, "microagg_model.pkl")
    print("Model saved as microagg_model.pkl")

def load_pipeline():
    return joblib.load("microagg_model.pkl")

def load_kb_embedder_faiss():
    kb_df = pd.read_csv("rag_kb.csv", encoding='ISO-8859-1')
    kb = kb_df['explanation'].dropna().tolist()
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    kb_embeddings = embedder.encode(kb)
    index = faiss.IndexFlatL2(kb_embeddings.shape[1])
    index.add(kb_embeddings)
    return kb, embedder, index

def load_rephraser():
    try:
        return hf_pipeline("text2text-generation", model="t5-base", tokenizer="t5-base")
    except:
        try:
            return hf_pipeline("text2text-generation", model="t5-small", tokenizer="t5-small")
        except:
            return None

def enhanced_rule_based_rephrasing(original):
    text = original.strip()
    patterns = {
        r'\bwomen do\b': 'What is the involvement of women in',
        r'\bcan women\b': 'What are women\'s capabilities regarding',
        r'\bdo women\b': 'What is women\'s experience with',
        r'\bgirls and\b': 'What about female participation in',
        r'\bwhy don\'t women\b': 'What factors might influence women\'s participation in',
        r'\bwhy can\'t women\b': 'What are the barriers women face in',
        r'\byou people\b': 'individuals from diverse backgrounds',
        r'\byour kind\b': 'people in similar situations',
        r'\bthose people\b': 'individuals from that community',
        r'\bwomen don\'t\b': 'some women may not',
        r'\bwomen can\'t\b': 'there may be barriers preventing women from',
        r'\bwomen aren\'t\b': 'women may not always be',
    }

    rephrased = text
    for pattern, replacement in patterns.items():
        rephrased = re.sub(pattern, replacement, rephrased, flags=re.IGNORECASE)

    if rephrased != text:
        if '?' in text and '?' not in rephrased:
            rephrased += '?'
        return rephrased[0].upper() + rephrased[1:]

    if '?' in text:
        return f"I'd like to understand more about {text.replace('?', '')}. Could you help me?"
    return f"Could you help me understand: {text}?"

def clean_generated_text(text):
    text = re.sub(r'^(rephrase|rewrite|question|answer):\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text.strip())
    if text and not text[0].isupper():
        text = text[0].upper() + text[1:]
    if text and text[-1] not in '.!?':
        text += '.'
    return text

def generate_rephrasing(text, rephraser_pipeline):
    rule_based = enhanced_rule_based_rephrasing(text)
    if rephraser_pipeline:
        try:
            prompts = [f"Rewrite: {text}", f"Improve: {text}"]
            for prompt in prompts:
                response = rephraser_pipeline(prompt, max_length=50, num_return_sequences=1)
                result = clean_generated_text(response[0]['generated_text'])
                if result.lower() != text.lower():
                    return result
        except:
            pass
    return rule_based

def classify_and_explain(text, pipeline, embedder, kb, index):
    prediction = pipeline.predict([text])[0]
    if prediction == 1:
        query_vec = embedder.encode([text])
        _, I = index.search(np.array(query_vec), k=1)
        explanation = kb[I[0][0]]
        return "Microaggression", explanation
    return "Not a microaggression", None

