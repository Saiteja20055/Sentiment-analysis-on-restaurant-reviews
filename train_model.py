import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re

nltk.download('stopwords')

# Load the dataset using a raw string literal for the file path
df = pd.read_csv(r'C:\Users\Bhavana\OneDrive\Desktop\analysis\Restaurant_Reviews .tsv', delimiter='\t', quoting=3)

# Text preprocessing function
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
    return ' '.join(text)

# Preprocess the reviews
df['Review'] = df['Review'].apply(preprocess_text)

# Features and Labels
X = df['Review']
y = df['Liked']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Create a pipeline with a CountVectorizer and Multinomial Naive Bayes classifier
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Train the model
pipeline.fit(X_train, y_train)

# Save the model to disk
with open('sentiment_model.pkl', 'wb') as model_file:
    pickle.dump(pipeline, model_file)

print("Model trained and saved successfully.")
