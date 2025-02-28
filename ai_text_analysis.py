import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

class TextAnalyzer:
    def __init__(self, data):
        self.data = data
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(stop_words=self.stop_words)
        self.model = None

    def clean_text(self, text):
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\@w+|\#','', text)
        text = re.sub(r'\d+', '', text)
        text = text.lower()
        text = ' '.join(word for word in text.split() if word not in self.stop_words)
        return text

    def preprocess_data(self):
        self.data['cleaned_text'] = self.data['text'].apply(self.clean_text)
    
    def split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(self.data['cleaned_text'], 
                                                            self.data['label'], 
                                                            test_size=0.2, 
                                                            random_state=42)
        return X_train, X_test, y_train, y_test

    def vectorize_data(self, X_train, X_test):
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)
        return X_train_vectorized, X_test_vectorized

    def train_model(self, X_train_vectorized, y_train):
        self.model = MultinomialNB()
        self.model.fit(X_train_vectorized, y_train)

    def evaluate_model(self, X_test_vectorized, y_test):
        y_pred = self.model.predict(X_test_vectorized)
        print(classification_report(y_test, y_pred))
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        plt.show()

    def sentiment_analysis(self, text):
        sia = SentimentIntensityAnalyzer()
        sentiment = sia.polarity_scores(text)
        return sentiment

    def topic_modeling(self, n_topics=5):
        tfidf = self.vectorizer.fit_transform(self.data['cleaned_text'])
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(tfidf)
        return lda

    def print_topics(self, model, num_words=5):
        for index, topic in enumerate(model.components_):
            print(f'Topic {index}:')
            print(" ".join([self.vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-num_words:]]))

def main():
    data_path = 'data.csv'  # path to CSV file containing text data
    text_data = pd.read_csv(data_path)
    analyzer = TextAnalyzer(text_data)
    analyzer.preprocess_data()
    X_train, X_test, y_train, y_test = analyzer.split_data()
    X_train_vectorized, X_test_vectorized = analyzer.vectorize_data(X_train, X_test)
    analyzer.train_model(X_train_vectorized, y_train)
    analyzer.evaluate_model(X_test_vectorized, y_test)

    sample_text = "I love programming in Python!"
    sentiment = analyzer.sentiment_analysis(sample_text)
    print(f'Sentiment analysis: {sentiment}')

    lda_model = analyzer.topic_modeling(3)
    analyzer.print_topics(lda_model)

if __name__ == "__main__":
    main()