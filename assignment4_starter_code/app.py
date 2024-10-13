from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)


# TODO: Fetch dataset, initialize vectorizer and LSA here

# fetching dataset
newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

# getting NLTK stopwords
stop_words = list(stopwords.words('english'))

# initializing vectorizer
vectorizer = TfidfVectorizer(stop_words=stop_words)
tfidf_matrix = vectorizer.fit_transform(documents)

# defining LSA
n_components = 100  # You can adjust this as needed
svd = TruncatedSVD(n_components=n_components)

# LSA function
def perform_lsa(tfidf_matrix):
    lsa_matrix = svd.fit_transform(tfidf_matrix)
    terms = vectorizer.get_feature_names_out()
    components = svd.components_
    return lsa_matrix, terms, components

# performing LSA
lsa_matrix, terms, components = perform_lsa(tfidf_matrix)


def search_engine(query):
    """
    Function to search for top 5 similar documents given a query
    Input: query (str)
    Output: documents (list), similarities (list), indices (list)
    """
    # TODO: Implement search engine here
    # return documents, similarities, indices 

    query_tfidf = vectorizer.transform([query])
    
    query_lsa = svd.transform(query_tfidf)
    
    similarities = cosine_similarity(query_lsa, lsa_matrix).flatten()
    
    top_indices = similarities.argsort()[-5:][::-1]
    
    similar_documents = [documents[i] for i in top_indices]
    similar_scores = [similarities[i] for i in top_indices]
    
    return similar_documents, similar_scores, top_indices



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents, similarities, indices = search_engine(query)
    indices = indices.tolist()
    return jsonify({'documents': documents, 'similarities': similarities, 'indices': indices}) 

if __name__ == '__main__':
    app.run(debug=True)
