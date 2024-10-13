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

# initializing vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)

# defining LSA
n_components = 100  # You can adjust this as needed
svd = TruncatedSVD(n_components=n_components)

# LSA function
def perform_lsa(tfidf_matrix, n_components=2):
    lsa_matrix = svd.fit_transform(tfidf_matrix)
    
    terms = vectorizer.get_feature_names_out()
    components = svd.components_
    
    return lsa_matrix, terms, components

# performing LSA
lsa_matrix, terms, components = perform_lsa(tfidf_matrix, n_components)


def search_engine(query):
    """
    Function to search for top 5 similar documents given a query
    Input: query (str)
    Output: documents (list), similarities (list), indices (list)
    """
    # TODO: Implement search engine here
    # return documents, similarities, indices 

    # Step 1: Transform the query into TF-IDF representation
    query_tfidf = vectorizer.transform([query])
    
    # Step 2: Reduce the dimensionality of the query using SVD
    query_lsa = svd.transform(query_tfidf)
    
    # Step 3: Compute cosine similarity between the query and document LSA representations
    similarities = cosine_similarity(query_lsa, lsa_matrix).flatten()
    
    # Step 4: Get the indices of the top 5 most similar documents
    top_indices = similarities.argsort()[-5:][::-1]
    
    # Step 5: Retrieve the similar documents and their scores
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
    return jsonify({'documents': documents, 'similarities': similarities, 'indices': indices}) 

if __name__ == '__main__':
    app.run(debug=True)
