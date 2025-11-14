"""
Train and save the sentiment analysis model properly
"""
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from collections import Counter
from pathlib import Path

from text_processing import preprocess_texts

print("Loading data...")
data = pd.read_csv('artifacts/sentiment_analysis.csv')

print("Applying preprocessing pipeline...")
clean_tweets = preprocess_texts(data['tweet'])

print("Building vocabulary...")
vocab = Counter()
for sentence in clean_tweets:
    vocab.update(sentence.split())

# Keep tokens that appear more than the minimum frequency threshold
MIN_TOKEN_FREQUENCY = 1
tokens = [key for key in vocab if vocab[key] >= MIN_TOKEN_FREQUENCY]
print(f"Vocabulary size: {len(tokens)} tokens")

VOCAB_PATH = Path('static/model/vocabulary.txt')
VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
with VOCAB_PATH.open('w', encoding='utf-8') as vocab_file:
    vocab_file.write('\n'.join(tokens))
print(f"Vocabulary saved to {VOCAB_PATH}")

print("Vectorizing data...")
def vectorizer(ds, vocabulary):
    vectorized_lst = []
    for sentence in ds:
        sentence_lst = np.zeros(len(vocabulary))
        for i in range(len(vocabulary)):
            if vocabulary[i] in sentence.split():
                sentence_lst[i] = 1
        vectorized_lst.append(sentence_lst)
    return np.asarray(vectorized_lst, dtype=np.float32)

X = vectorizer(clean_tweets, tokens)
y = data['label']

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluate
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Training accuracy: {train_score:.4f}")
print(f"Test accuracy: {test_score:.4f}")

print("Saving model to static/model/model.pickle...")
with open('static/model/model.pickle', 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully!")
print("\nYou can now run the Flask app with: python app.py")
