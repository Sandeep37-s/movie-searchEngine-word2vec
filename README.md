Semantic Movie Search Engine
This project is a movie search engine that finds movies based on the meaning of a query, not just keywords. You can describe a plot, a theme, or a concept, and the engine will return the most relevant movies from a dataset of over 34,000 films.

How It Works
The search engine uses a combination of Natural Language Processing (NLP) techniques and a fast vector search index to understand and retrieve movies.

Data Preprocessing: Movie plot data from the Wiki Plots dataset is cleaned by removing common "stopwords" (like 'the', 'a', 'is'), punctuation, and converting text to lowercase using the NLTK library.

Word Embeddings (Word2Vec): To understand the meaning of words, a hybrid approach is used:

A powerful, pre-trained Google News model (trained on 100 billion words) provides a broad understanding of the English language.

A custom Word2Vec model, trained on our specific movie plot data, acts as a fallback for movie-specific terms.

This combination ensures both general knowledge and domain-specific expertise.

Vector Indexing: Each movie plot is converted into a single numerical vector (a "topical fingerprint"). These vectors are then organized into a highly efficient search index using Spotify's Annoy library for instant lookups.

Web Application: A Flask web server provides a simple user interface. When you enter a query:

The query is converted into a vector using the same hybrid model.

The Annoy index finds the movie vectors that are mathematically closest to your query's vector.

The corresponding movie titles and plots are displayed as results.

Setup and Installation
To run this project, you will need to install the required Python libraries.

# Install all necessary packages
pip install pandas nltk gensim annoy Flask

You will also need to download the NLTK data and the pre-trained Google News model:

# Run this in a Python shell
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

Note: The GoogleNews-vectors-negative300.bin.gz file is required and must be in the main project folder.

How to Run
Build the Backend: Run the master Python script to process the data and create the necessary model files (.model, .ann, .csv).

Start the Web Server: Once the backend files are created, run the Flask application from your terminal:

python app.py

Use the Search Engine: Open your web browser and navigate to http://127.0.0.1:5000.
