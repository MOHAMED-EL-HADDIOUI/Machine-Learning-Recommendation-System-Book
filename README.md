# **Book Recommendation System**

## **Overview**

This project implements a **Book Recommendation System** that recommends books based on the content of their titles, authors, and publishers. It uses a combination of **TF-IDF (Term Frequency-Inverse Document Frequency)** for feature extraction and **Annoy (Approximate Nearest Neighbors)** for fast similarity searching. The system can recommend books similar to a given one, based on textual features like book title, author, and publisher.

## **Table of Contents**

- [Overview](#overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Features](#features)
- [Usage](#usage)
- [Preprocessing](#preprocessing)
- [Recommendation System](#recommendation-system)
- [Model Persistence](#model-persistence)
- [Acknowledgments](#acknowledgments)

## **Installation**

To run this project locally, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/book-recommendation-system.git
    cd book-recommendation-system
    ```

2. **Create a Virtual Environment (Optional but Recommended)**:
    ```bash
    python -m venv env
    source env/bin/activate  # For Unix/macOS
    env\Scripts\activate     # For Windows
    ```

3. **Install the Required Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Install Annoy**:
    Annoy is a library used for fast nearest neighbor search.
    ```bash
    pip install annoy
    ```

5. **Install scikit-learn**:
    ```bash
    pip install scikit-learn
    ```

## **Project Structure**

```
📦book-recommendation-system
 ┣ 📂data
 ┃ ┣ 📜books.csv                # Raw dataset containing books information
 ┃ ┗ 📜preprocessed_books.csv    # Preprocessed dataset
 ┣ 📂models
 ┃ ┣ 📜annoy_index.ann           # Saved Annoy index
 ┃ ┗ 📜tfidf_vectorizer.pkl       # Saved TF-IDF vectorizer
 ┣ 📜book_recommendation.py      # Main script containing the recommendation logic
 ┣ 📜README.md                   # Project documentation
 ┗ 📜requirements.txt            # List of required Python packages

```

## **Features**

- **Preprocessing**: Cleans and prepares the book dataset by combining the title, author, and publisher into a single text feature.
- **TF-IDF Vectorization**: Converts book textual features into a numerical format using TF-IDF.
- **Annoy Index**: Builds an efficient index for fast book similarity searching using angular distance.
- **Recommendations**: Recommends similar books to a given ISBN using the Annoy index.

## **Usage**

1. **Preprocessing Data**:
    To preprocess the book dataset, run the following code inside `book_recommendation.py`:
    ```python
    books_ = preprocess_data(books)
    ```

2. **Creating the TF-IDF Matrix**:
    Create a TF-IDF matrix for book features:
    ```python
    tfidf_matrix, tfidf_vectorizer = create_tfidf_matrix(books_)
    ```

3. **Building the Annoy Index**:
    Build the Annoy index for fast recommendations:
    ```python
    annoy_index = build_annoy_index(tfidf_matrix)
    ```

4. **Saving the Preprocessed Data and Models**:
    You can save the preprocessed data, Annoy index, and TF-IDF vectorizer for future use:
    ```python
    save_annoy_index(annoy_index, 'models/annoy_index.ann')
    save_tfidf_vectorizer(tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')
    books_.to_csv('data/preprocessed_books.csv', index=False)
    ```

5. **Recommending Similar Books**:
    Use the following command to get book recommendations based on a specific ISBN:
    ```python
    similar_books = recommend_similar_books_annoy(isbn, books_, annoy_index, n=5)
    ```

6. **Example**:
    ```python
    isbn = "0312195516"
    recommendations = recommend_similar_books_annoy(isbn, books_, annoy_index, n=5)
    for book in recommendations:
        print(book)
    ```
- [Resources]([https://pandas.pydata.org/](https://drive.google.com/drive/folders/1QzwikZnHYsfS-eE8frA-PbIZj95XuPB_?usp=drive_link)) : Resources
  
## **Preprocessing**

- The dataset is cleaned by filling missing values for book titles, authors, and publishers with the placeholder "Unknown".
- Features for each book are created by concatenating the title, author, and publisher into a single string.

## **Recommendation System**

- **TF-IDF** is used to transform book features into a matrix of numerical values representing the importance of words.
- **Annoy** builds an index for fast search based on angular distance, which is similar to cosine similarity. This allows for quick retrieval of books that are similar to a given one.

## **Model Persistence**

- The **Annoy index** and **TF-IDF vectorizer** can be saved to disk and reloaded later to avoid rebuilding the index and vectorizer from scratch every time.
- The `annoy_index.ann` and `tfidf_vectorizer.pkl` files contain the saved index and vectorizer.

## **Acknowledgments**


This project is built using the following libraries:

- [Pandas](https://pandas.pydata.org/): For data manipulation.
- [Resources](https://drive.google.com/drive/folders/1QzwikZnHYsfS-eE8frA-PbIZj95XuPB_?usp=drive_link): Additional resources.
- [Scikit-learn](https://scikit-learn.org/): For building the TF-IDF matrix.
- [Annoy](https://github.com/spotify/annoy): For fast approximate nearest neighbors search.
- [Pickle](https://docs.python.org/3/library/pickle.html): For saving and loading models.

