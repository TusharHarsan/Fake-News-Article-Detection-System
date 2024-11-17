import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Download NLTK data
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv('news.csv')

# Split the data
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = vectorizer.fit_transform(X_train)
tfidf_test = vectorizer.transform(X_test)

# Save the vectorizer
pickle.dump(vectorizer, open('vector.pkl', 'wb'))

# Dictionary to store models and their names
models = {
    'Passive Aggressive Classifier': PassiveAggressiveClassifier(max_iter=50),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Support Vector Machine': LinearSVC(random_state=42),
    'Naive Bayes': MultinomialNB()
}

# Train and save each model
for name, model in models.items():
    print(f"\nTraining {name}...")
    # Train the model
    model.fit(tfidf_train, y_train)
    
    # Evaluate
    y_pred = model.predict(tfidf_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")
    
    # Save model
    filename = name.lower().replace(' ', '_') + '_model.pkl'
    pickle.dump(model, open(filename, 'wb'))
    print(f"Model saved as {filename}")

print("\nAll models have been trained and saved!")
