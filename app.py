from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import string
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from gensim.models import Word2Vec, KeyedVectors # Import KeyedVectors for the Google model
from annoy import AnnoyIndex

# ==============================================================================
# ONE-TIME SETUP: Load all new models and data
# ==============================================================================
print("Loading backend models and data... This may take a moment.")

# --- The vector size is now 300 (from the Google model) ---
VECTOR_SIZE = 300

# --- Load the final movie data created by the combined script ---
try:
    df = pd.read_csv('final_movie_data_combined.csv')
except FileNotFoundError:
    print("FATAL ERROR: 'final_movie_data_combined.csv' not found. Please run the combined build script.")
    exit()

# --- Load BOTH the custom model and the Google News model ---
try:
    custom_model = Word2Vec.load("movie_word2vec.model")
    google_model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
except FileNotFoundError as e:
    print(f"FATAL ERROR: A model file was not found: {e}. Please ensure all model files are in the folder.")
    exit()

# --- Load the NEW Annoy search index ---
try:
    search_index = AnnoyIndex(VECTOR_SIZE, 'angular')
    search_index.load('movie_index_combined.ann')
    print("Backend loaded and ready! ✅")
except FileNotFoundError:
    print("FATAL ERROR: 'movie_index_combined.ann' not found. Please run the combined build script.")
    exit()

# ==============================================================================
# HELPER FUNCTIONS (Updated to use both models)
# ==============================================================================

def preprocess_text(text):
    """Cleans and tokenizes text for processing."""
    if pd.isna(text): return []
    tokens = word_tokenize(str(text).lower())
    tagged_tokens = pos_tag(tokens)
    stop_words = set(stopwords.words('english'))
    punct = string.punctuation
    return [(word, tag) for word, tag in tagged_tokens if word not in stop_words and word not in punct and not word.isdigit()]

def create_combined_vector(tokens, google_model, custom_model):
    """Creates a single vector using the fallback model strategy."""
    words = [word for word, tag in tokens]
    word_vectors = []
    for word in words:
        if word in google_model:
            word_vectors.append(google_model[word])
        elif word in custom_model.wv:
            # Pad our smaller vector to match Google's 300 dimensions
            custom_vector = custom_model.wv[word]
            padded_vector = np.pad(custom_vector, (0, VECTOR_SIZE - len(custom_vector)), 'constant')
            word_vectors.append(padded_vector)

    if not word_vectors:
        return np.zeros(VECTOR_SIZE)
    return np.mean(word_vectors, axis=0)

# ==============================================================================
# FLASK APPLICATION
# ==============================================================================

app = Flask(__name__)

@app.route('/')
def home():
    """Renders the main search page."""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """Handles the search query and displays results."""
    user_query = request.form['query']

    if not user_query:
        return render_template('index.html', error="Please enter a search term.")

    # 1. Preprocess the user's query
    query_tokens = preprocess_text(user_query)

    # 2. Convert the query to a vector using the COMBINED model function
    query_vector = create_combined_vector(query_tokens, google_model, custom_model)

    # 3. Check if the query vector is "blank" (all zeros)
    if np.all(query_vector == 0):
        results = []
    else:
        # 4. If the vector is valid, search the Annoy index
        result_indices = search_index.get_nns_by_vector(query_vector, 10)
        results_df = df.iloc[result_indices]
        results = results_df.to_dict('records')

    return render_template('results.html', results=results, query=user_query)

if __name__ == '__main__':
    app.run(debug=True)

