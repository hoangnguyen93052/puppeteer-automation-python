import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
import joblib

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Preprocessing text
def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lower case
    text = text.lower()
    # Tokenization
    tokens = nltk.word_tokenize(text)
    # Remove stop words
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Preparing the dataset
def prepare_dataset(data):
    data['cleaned_text'] = data['text'].apply(preprocess_text)
    X = data['cleaned_text']
    y = data['label']
    return X, y

# Splitting the dataset
def split_dataset(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the model
def create_model():
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    return model

# Training the model
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

# Evaluating the model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))

# Saving the model
def save_model(model, filename):
    joblib.dump(model, filename)

# Main function
def main(file_path):
    data = load_data(file_path)
    X, y = prepare_dataset(data)
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    model = create_model()
    trained_model = train_model(model, X_train, y_train)
    evaluate_model(trained_model, X_test, y_test)

    save_model(trained_model, 'text_classifier_model.pkl')

if __name__ == "__main__":
    main('data/text_data.csv')

# Sample test code
def test_model():
    model = joblib.load('text_classifier_model.pkl')
    sample_texts = [
        "This is a great movie!",
        "I didn't like the film at all.",
        "The plot was boring and predictable."
    ]
    sample_preprocessed = [preprocess_text(text) for text in sample_texts]
    predictions = model.predict(sample_preprocessed)
    for text, prediction in zip(sample_texts, predictions):
        print(f'Text: "{text}" => Prediction: {prediction}')

if __name__ == "__main__":
    test_model()

# Utility function to show available classes
def show_classes(data):
    print(data['label'].value_counts())

if __name__ == "__main__":
    data = load_data('data/text_data.csv')
    show_classes(data)