import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Client description query
query = "TESOURA FINA PCT BICUDA 115MM"

# Tokenize the query
tokens = word_tokenize(query)
print("Tokens:", tokens)

# Create a TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Fit the vectorizer to the query and transform it into a vector
#Sparse matrix result 
vector = vectorizer.fit_transform([query]) 

# Get the feature names (i.e., the tokenized words)
feature_names = vectorizer.get_feature_names()
print("Feature Names:", feature_names)

# Print the vectorized query
#dense vector 
print("Vectorized Query:", vector.toarray())