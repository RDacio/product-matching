from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = KeyedVectors.load_word2vec_format('C:/Users/Rodrigo Dácio/Documents/MyCode/GoogleNews-vectors-negative300.bin', binary=True)

query = "TESOURA FINA PCT BICUDA 115MM"
description = "TESOURA FINA P.RCT.BICUDA 130MM"

stop_words = set(stopwords.words('portuguese'))

query_words = [word for word in word_tokenize(query) if word not in stop_words]
description_words = [word for word in word_tokenize(description) if word not in stop_words]

# Create vectors for the words present in the model
query_vector = [model[word] for word in query_words if word in model]
description_vector = [model[word] for word in description_words if word in model]

# Check if there are any valid vectors
if query_vector and description_vector:
    # Compute similarity
    similarity = cosine_similarity([np.mean(query_vector, axis=0)], [np.mean(description_vector, axis=0)])
    print("Similarity:", similarity[0][0])
else:
    print("No valid words found in the model for comparison.")

similarity = cosine_similarity(query_vector, description_vector)

print("Similarity:", similarity)
