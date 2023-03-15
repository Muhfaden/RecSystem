#import libiries
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the Data
df = pd.read_csv('C:/Users/Noodi/OneDrive/Desktop/jawalan/cities.csv', encoding='unicode_escape')

# Preprocess the Data
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))

#NLP
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())

    # Remove stopwords and punctuation
    tokens = [token for token in tokens if token not in stop_words and token not in punctuation]

    # Join the tokens back into a string
    preprocessed_text = " ".join(tokens)

    return preprocessed_text

# Vectorize the Data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Description"].apply(preprocess_text))

#  Build the Recommendation System Algorithm
def get_recommendations(disability, X):
    # Preprocess the disability input
    preprocessed_disability = preprocess_text(disability)

    # Vectorize the preprocessed disability input
    vectorized_disability = vectorizer.transform([preprocessed_disability])

    # Calculate cosine similarities between the disability input and all descriptions
    similarities = cosine_similarity(X, vectorized_disability)

    # Get the indices of the places with the highest cosine similarities
    indices = similarities.argsort(axis=0)[::-1].flatten()

    # Return the top 6 recommended places
    return df.iloc[indices[:6]]["Place"]

disability = "deaf/nonverbal"
recommendations = get_recommendations(disability, X)
print(recommendations)




