import pandas as pd
from textblob import TextBlob
import nltk
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
import string

try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    pass

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text_textblob(text):
    """Preprocess text using TextBlob and NLTK - equivalent to spaCy preprocessing"""
    try:
    
        text = text.lower()
        
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        blob = TextBlob(text)
        
        processed_tokens = []
        for word in blob.words:
            if word not in stop_words and len(word) > 1:
                lemmatized = lemmatizer.lemmatize(word)
                processed_tokens.append(lemmatized)
        
        return " ".join(processed_tokens)
    except:
        return text.lower()

df = pd.read_csv("micro_agg.csv", encoding='ISO-8859-1')
df['ptext'] = df['speech'].apply(preprocess_text_textblob)

X = df['speech']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class TextBlobPreprocessor(BaseEstimator, TransformerMixin):
    """Custom preprocessor using TextBlob instead of spaCy"""
    def fit(self, X, y=None): 
        return self
    
    def transform(self, X):
        processed_texts = []
        for text in X:
            try:
                text = text.lower()
                text = text.translate(str.maketrans('', '', string.punctuation))
                
              
                blob = TextBlob(text)
                
                tokens = []
                for word in blob.words:
                    if len(word) > 1:  # Remove single characters
                        lemmatized = lemmatizer.lemmatize(word)
                        tokens.append(lemmatized)
                
                processed_texts.append(" ".join(tokens))
            except:
                processed_texts.append(text.lower())
        
        return processed_texts

pipeline = Pipeline([
    ('textblob', TextBlobPreprocessor()),
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95, sublinear_tf=True)),
    ('sgd', SGDClassifier(loss='log_loss', penalty='elasticnet', alpha=1e-5,
    max_iter=1000, class_weight='balanced', random_state=42))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

kb_df = pd.read_csv("rag_kb.csv", encoding='ISO-8859-1')
kb = kb_df['explanation'].dropna().tolist()
embedder = SentenceTransformer('all-MiniLM-L6-v2')
kb_embeddings = embedder.encode(kb)
index = faiss.IndexFlatL2(kb_embeddings.shape[1])
index.add(kb_embeddings)

print("Loading T5 model for rephrasing...")
try:
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
    text = re.sub(r'\s+', ' ', text.strip())
    
    if len(text.split()) <= 3:
        if '?' in text:
            return f"How can I ask about {text.replace('?', '')} in a more respectful way"
        else:
            return f"Rephrase this statement respectfully: {text}"
    
    return text

def generate_rephrasing(original):
    rule_based_result = enhanced_rule_based_rephrasing(original)
    
    if rule_based_result and rule_based_result.lower() != original.lower():
        generic_phrases = ["could you help me understand", "i'd like to understand", "consider rephrasing"]
        if not any(phrase in rule_based_result.lower() for phrase in generic_phrases):
            return rule_based_result
    
    if rephraser_pipeline is not None:
        try:
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
    
    return rule_based_result if rule_based_result else f"Could you rephrase this more respectfully: {original}"

def clean_generated_text(text):
    """Clean up the generated text"""
    import re
    
    text = re.sub(r'^(rephrase|rewrite|improve|make this more inclusive|question|answer):\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^(.*?)(rephrase|rewrite|improve|make this more inclusive|question|answer):\s*', '', text, flags=re.IGNORECASE)
    
    parts = text.split()
    if len(parts) > 6:
        mid = len(parts) // 2
        first_half = ' '.join(parts[:mid])
        second_half = ' '.join(parts[mid:])
        if first_half.lower() == second_half.lower():
            text = first_half
    
    text = re.sub(r'\s+', ' ', text.strip())
    
    if text and not text[0].isupper():
        text = text[0].upper() + text[1:]
    
    if text and text[-1] not in '.!?':
        text += '?'
    
    return text

def enhanced_rule_based_rephrasing(original):
    """Enhanced rule-based rephrasing with better patterns"""
    try:
        text = original.strip()
        lower_text = text.lower()
        
        respectful_patterns = {
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
        for pattern, replacement in respectful_patterns.items():
            rephrased = re.sub(pattern, replacement, rephrased, flags=re.IGNORECASE)
        
        if rephrased != text:
            if '?' in original and '?' not in rephrased:
                rephrased += '?'
            
            if rephrased and not rephrased[0].isupper():
                rephrased = rephrased[0].upper() + rephrased[1:]
            
            return rephrased
        
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

# Test samples
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

# Save the model
joblib.dump(pipeline, "microagg_model.pkl")
print("\nModel saved successfully!")
