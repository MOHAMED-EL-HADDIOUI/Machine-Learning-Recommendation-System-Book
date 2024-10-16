# Imports
from flask import Flask, request, jsonify
from fastai.collab import load_learner, CollabDataLoaders, collab_learner
from fastai.tabular.all import *
from annoy import AnnoyIndex
import pandas as pd
import pickle
import os
import torch

# Ensure KMP_DUPLICATE_LIB_OK is set to avoid issues with MKL
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Flask app setup
app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Set device based on GPU availability
print(f"Using device: {device}")
print(torch.cuda.is_available())
print(torch.version.cuda)

# Data loading functions
def load_preprocessed_data(filename='preprocessed_data.csv'):
    return pd.read_csv(filename)

def load_dataloader(filename='dataloader.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def load_data_and_dataloader():
    all_ratings = load_preprocessed_data()
    data = load_dataloader()
    return all_ratings, data

def load_books_data(filename='books_data.csv'):
    return pd.read_csv(filename)

# Annoy and TF-IDF functions
def create_tfidf_matrix(df, max_features=10000):
    tfidf = TfidfVectorizer(stop_words='english', max_features=max_features)
    return tfidf.fit_transform(df['features']), tfidf

def build_annoy_index(tfidf_matrix, n_trees=10):
    dim = tfidf_matrix.shape[1]  # Dimension of each TF-IDF vector
    annoy_index = AnnoyIndex(dim, 'angular')  # Using angular distance (similar to cosine)
    for i in range(tfidf_matrix.shape[0]):
        annoy_index.add_item(i, tfidf_matrix[i].toarray()[0])  # Add the TF-IDF vectors to the index
    annoy_index.build(n_trees)  # Build the index with n trees
    return annoy_index

def recommend_similar_books_annoy(isbn, df, annoy_index, n=5):
    try:
        idx = df[df['ISBN'] == isbn].index[0]  # Find the index of the book by ISBN
        book_indices = annoy_index.get_nns_by_item(idx, n + 1)[1:]  # Get the n nearest neighbors (excluding the book itself)
        return df.iloc[book_indices][['ISBN']].to_dict(orient='records')
    except IndexError:
        return "ISBN not found in dataset."

# Loading models and indices
def load_annoy_index(file_path, dim):
    annoy_index = AnnoyIndex(dim, 'angular')  # Dimension must be the same as when you created the index
    annoy_index.load(file_path)  # Load the Annoy index
    return annoy_index

def load_tfidf_vectorizer(file_path):
    with open(file_path, 'rb') as file:
        tfidf_vectorizer = pickle.load(file)
    return tfidf_vectorizer

# Load the TF-IDF vectorizer and Annoy index
tfidf_vectorizer = load_tfidf_vectorizer('tfidf_vectorizer.pkl')
dim = tfidf_vectorizer.max_features  # Dimension used during index creation
annoy_index = load_annoy_index('annoy_index.ann', dim)
books = load_books_data()

# Load the data and model
all_ratings, data = load_data_and_dataloader()

# Recreate the Learner object with the data
learn = collab_learner(data, n_factors=20, y_range=(0., 10.0), wd=1e-1)

# Load the model onto the appropriate device
learn.load('book_recommendation_model', device=device)

# Flask routes
@app.route('/book/<isbn>', methods=['GET'])
def get_book_info(isbn):
    similar_books = recommend_similar_books_annoy(isbn, books, annoy_index, n=5)
    return jsonify({"similar_books": similar_books})

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_id = data.get('userId')
    if user_id is None:
        return jsonify({"error": "userId is required"}), 400

    try:
        user_id = int(user_id)
    except ValueError:
        return jsonify({"error": "userId must be an integer"}), 400

    # Get all unique ISBNs
    books = all_ratings['ISBN'].unique()

    # Get books that the user has not rated yet
    rated_books = all_ratings[all_ratings['userId'] == user_id]['ISBN']
    unrated_books = list(set(books) - set(rated_books))

    # Create a DataFrame for predictions
    to_predict = pd.DataFrame({
        'userId': [user_id] * len(unrated_books),
        'ISBN': unrated_books
    })

    # Create a DataLoader for predictions
    dl = learn.dls.test_dl(to_predict)

    # Get predictions
    preds, _ = learn.get_preds(dl=dl)

    # Convert predictions to a list of tuples (ISBN, prediction)
    predicted_ratings = list(zip(unrated_books, preds.squeeze().tolist()))

    # Sort by predicted rating
    recommendations = sorted(predicted_ratings, key=lambda x: x[1], reverse=True)

    # Take the top 5 recommendations
    top_recommendations = recommendations[:20]

    # Format recommendations for the JSON response
    formatted_recommendations = [{"ISBN": isbn, "predicted_rating": rating} for isbn, rating in top_recommendations]

    return jsonify(recommendations=formatted_recommendations)

# Run the Flask app
if __name__ == '__main__':
    app.run(port=5000, debug=True)
