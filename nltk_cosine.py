import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Preprocessing
def preprocess_text(text):
    # Implement preprocessing steps, such as:
    # Tokenization, stopword removal, stemming or lemmatization
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.lower() not in stopwords.words('portuguese')]  # Remove stopwords
    return ' '.join(tokens)

# Calculate similarity
def calculate_similarity(query, products):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([query] + products)
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
    return cosine_similarities[0][1:]

# Search and rank
def search_products(query, products):
    preprocessed_query = preprocess_text(query)
    preprocessed_products = [preprocess_text(product) for product in products]
    similarities = calculate_similarity(preprocessed_query, preprocessed_products)
    ranked_products = [(product, similarity) for product, similarity in zip(products, similarities)]
    ranked_products.sort(key=lambda x: x[1], reverse=True)
    return ranked_products

# Client description
client_desc = "TESOURA FINA BICUDA 115MM"

# DB descriptions
db_descriptions = ["TESOURA IRIS CURVA 115MM ", "TESOURA FINA P.RCT.BICUDA 115MM", "TESOURA IRIS RETA 130MM"]

# Search and rank products
ranked_products = search_products(client_desc, db_descriptions)

# Print the client description and the best matching description from the database
print(f"Client Description: {client_desc}")
print(f"Best Match in DB: {ranked_products[0][0]} (Score: {ranked_products[0][1]})")