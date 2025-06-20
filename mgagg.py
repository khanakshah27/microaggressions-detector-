import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
import joblib
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline as hf_pipeline
import re

# 🔹 Load data
nlp = spacy.load("en_core_web_sm")
df = pd.read_csv("micro_agg.csv", encoding='ISO-8859-1')
df['ptext'] = df['speech'].apply(lambda x: " ".join(
    [token.lemma_ for token in nlp(x.lower()) if not token.is_stop and not token.is_punct]))

X = df['speech']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class SpacyPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        return [" ".join([token.lemma_ for token in nlp(text.lower()) if not token.is_punct]) for text in X]

# 🔹 Train model
pipeline = Pipeline([
    ('spacy', SpacyPreprocessor()),
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95, sublinear_tf=True)),
    ('sgd', SGDClassifier(loss='log_loss', penalty='elasticnet', alpha=1e-5,
    max_iter=1000, class_weight='balanced', random_state=42))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# 🔹 RAG Embedding Setup
kb_df = pd.read_csv("rag_kb.csv", encoding='ISO-8859-1')
kb = kb_df['explanation'].dropna().tolist()
embedder = SentenceTransformer('all-MiniLM-L6-v2')
kb_embeddings = embedder.encode(kb)
index = faiss.IndexFlatL2(kb_embeddings.shape[1])
index.add(kb_embeddings)

# 🔹 Improved T5 Rephrasing Setup
print("Loading T5 model for rephrasing...")
try:
    # Use T5-base which is better than T5-small for text generation
    rephraser_pipeline = hf_pipeline(
        "text2text-generation",
        model="t5-base",
        tokenizer="t5-base"
    )
    print("Successfully loaded T5-base model")
except Exception as e:
    print(f"T5-base failed: {e}")
    try:
        rephraser_pipeline = hf_pipeline(
            "text2text-generation",
            model="t5-small",
            tokenizer="t5-small"
        )
        print("Successfully loaded T5-small model")
    except Exception as e2:
        print(f"T5-small also failed: {e2}")
        rephraser_pipeline = None

def preprocess_for_rephrasing(text):
    """Preprocess text to make it more suitable for T5 rephrasing"""
    # Remove extra spaces and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    
    # If it's a very short phrase, expand it slightly for better context
    if len(text.split()) <= 3:
        if '?' in text:
            return f"How can I ask about {text.replace('?', '')} in a more respectful way"
        else:
            return f"Rephrase this statement respectfully: {text}"
    
    return text

def generate_rephrasing(original):
    # First, try rule-based rephrasing as it's more reliable for microaggressions
    rule_based_result = enhanced_rule_based_rephrasing(original)
    
    # If rule-based found a good transformation, use it
    if rule_based_result and rule_based_result.lower() != original.lower():
        # Check if it's not just a generic wrapper
        generic_phrases = ["could you help me understand", "i'd like to understand", "consider rephrasing"]
        if not any(phrase in rule_based_result.lower() for phrase in generic_phrases):
            return rule_based_result
    
    # Only try T5 if rule-based didn't work well
    if rephraser_pipeline is not None:
        try:
            # Simpler, more direct prompts for T5
            simple_prompts = [
                f"Question: {original} Answer:",
                f"Rewrite: {original}",
                f"{original} Better version:"
            ]
            
            for prompt in simple_prompts:
                try:
                    response = rephraser_pipeline(
                        prompt,
                        max_length=50,
                        min_length=5,
                        num_return_sequences=1,
                        do_sample=True,
                        top_k=30,
                        top_p=0.9,
                        temperature=0.5,
                        early_stopping=True
                    )
                    
                    result = response[0]['generated_text'].strip()
                    cleaned_result = clean_generated_text(result)
                    
                    # Check if the result is actually different and useful
                    if (cleaned_result and 
                        len(cleaned_result) > 5 and 
                        cleaned_result.lower() != original.lower() and
                        not cleaned_result.startswith(prompt.split(':')[0]) and
                        len(cleaned_result.split()) >= 3):
                        return cleaned_result
                        
                except Exception as e:
                    continue
        except Exception as e:
            print(f"T5 rephrasing failed: {e}")
    
    # Fallback to rule-based result (even if generic)
    return rule_based_result if rule_based_result else f"Could you rephrase this more respectfully: {original}"

def clean_generated_text(text):
    """Clean up the generated text"""
    import re
    
    # Remove common T5 artifacts and prompts
    text = re.sub(r'^(rephrase|rewrite|improve|make this more inclusive|question|answer):\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^(.*?)(rephrase|rewrite|improve|make this more inclusive|question|answer):\s*', '', text, flags=re.IGNORECASE)
    
    # Remove repeated phrases
    parts = text.split()
    if len(parts) > 6:
        # Check for repetition in the middle
        mid = len(parts) // 2
        first_half = ' '.join(parts[:mid])
        second_half = ' '.join(parts[mid:])
        if first_half.lower() == second_half.lower():
            text = first_half
    
    # Clean up spacing
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Ensure proper capitalization
    if text and not text[0].isupper():
        text = text[0].upper() + text[1:]
    
    # Ensure proper punctuation
    if text and text[-1] not in '.!?':
        text += '?'
    
    return text

def enhanced_rule_based_rephrasing(original):
    """Enhanced rule-based rephrasing with better patterns"""
    try:
        text = original.strip()
        lower_text = text.lower()
        
        # Enhanced respectful transformations
        respectful_patterns = {
            # Question patterns
            r'\bwomen do\b': 'What is the involvement of women in',
            r'\bcan women\b': 'What are women\'s capabilities regarding',
            r'\bdo women\b': 'What is women\'s experience with',
            r'\bgirls and\b': 'What about female participation in',
            r'\bwhy don\'t women\b': 'What factors might influence women\'s participation in',
            r'\bwhy can\'t women\b': 'What are the barriers women face in',
            
            # Problematic phrases
            r'\byou people\b': 'individuals from diverse backgrounds',
            r'\byour kind\b': 'people in similar situations',
            r'\bthose people\b': 'individuals from that community',
            
            # Assumptions
            r'\bwomen don\'t\b': 'some women may not',
            r'\bwomen can\'t\b': 'there may be barriers preventing women from',
            r'\bwomen aren\'t\b': 'women may not always be',
        }
        
        rephrased = text
        for pattern, replacement in respectful_patterns.items():
            rephrased = re.sub(pattern, replacement, rephrased, flags=re.IGNORECASE)
        
        # If we made changes, clean up the result
        if rephrased != text:
            # Ensure it's a proper question if it was originally
            if '?' in original and '?' not in rephrased:
                rephrased += '?'
            
            # Proper capitalization
            if rephrased and not rephrased[0].isupper():
                rephrased = rephrased[0].upper() + rephrased[1:]
            
            return rephrased
        
        # If no specific pattern matched, create a more inclusive version
        if '?' in text:
            return f"I'd like to understand more about {text.replace('?', '').strip()}. Could you help me learn about this?"
        else:
            return f"Could you help me understand: {text}?"
            
    except Exception as e:
        return f"Could you please rephrase this more respectfully: {original}"

def classify_and_explain(text):
    prediction = pipeline.predict([text])[0]
    if prediction == 1:
        query_vec = embedder.encode([text])
        _, I = index.search(np.array(query_vec), k=1)
        explanation = kb[I[0][0]]
        return "Microaggression", explanation
    else:
        return "Not a microaggression", None

# 🔹 Run test predictions with multiple examples
test_samples = [
    "Women do ML?"
]

print("\n" + "="*60)
print("TESTING REPHRASING FUNCTIONALITY")
print("="*60)

for sample_text in test_samples:
    print(f"\nOriginal: '{sample_text}'")
    prediction_status, explanation = classify_and_explain(sample_text)
    print(f"Prediction: {prediction_status}")
    
    if prediction_status == "Microaggression":
        print(f"Explanation: {explanation}")
        rephrased = generate_rephrasing(sample_text)
        print(f"Suggested Rephrasing: '{rephrased}'")
    else:
        print("No rephrasing needed as it's not a microaggression.")
    print("-" * 40)

# 🔹 Save model
joblib.dump(pipeline, "microagg_model.pkl")
print("\nModel saved successfully!")