from flask import Flask, request

from keybert import KeyBERT

model = KeyBERT('distilbert-base-nli-mean-tokens')


app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/extract", methods=["POST"])
def extract():
    if request.method == 'POST':
        text = request.form['text']
        keywords_with_scores = model.extract_keywords(
            text, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=3
        )
        keywords = [keyword for keyword, _ in keywords_with_scores]

        return {"keywords": keywords}
    else:
        return "Unsupported method"
