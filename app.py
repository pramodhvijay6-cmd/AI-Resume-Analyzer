from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    resume_file = request.files['resume_file']
    job_description = request.form['job_description']

    resume_text = resume_file.read().decode('utf-8', errors='ignore')

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_description])
    similarity = cosine_similarity(vectors)[0][1]

    score = round(similarity * 100, 2)

    return render_template('result.html', score=score)

if __name__ == '__main__':
    app.run(debug=True)