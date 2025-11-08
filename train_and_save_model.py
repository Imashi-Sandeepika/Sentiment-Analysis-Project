"""
Train and save the sentiment analysis model properly
"""
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from collections import Counter

print("Loading data...")
data = pd.read_csv('artifacts/sentiment_analysis.csv')

print("Building vocabulary...")
vocab = Counter()
for sentence in data['tweet']:
    vocab.update(sentence.split())

# Keep tokens that appear more than 10 times
tokens = [key for key in vocab if vocab[key] > 10]
print(f"Vocabulary size: {len(tokens)} tokens")

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

X = vectorizer(data['tweet'], tokens)
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
