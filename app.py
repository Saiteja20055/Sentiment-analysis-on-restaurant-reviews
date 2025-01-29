from flask import Flask, request, render_template
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import sklearn
nltk.download('stopwords')

app = Flask(__name__)

# Load the trained model
with open('sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Text preprocessing function
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
    return ' '.join(text)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        processed_review = preprocess_text(review)
        prediction = model.predict([processed_review])
        result = 'Positive' if prediction == 1 else 'Negative'
        return render_template('result.html', review=review, result=result)

if __name__ == '__main__':
    app.run(debug=True)
