from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

model = KeyedVectors.load_word2vec_format('C:/Users/Rodrigo Dácio/Documents/MyCode/GoogleNews-vectors-negative300.bin', binary=True)

query = "TESOURA FINA PCT BICUDA 115MM"
description = "TESOURA FINA P.RCT.BICUDA 115MM"

stop_words = set(stopwords.words('portuguese'))

query_tokens = [word for word in word_tokenize(query) if word not in stop_words]
description_tokens = [word for word in word_tokenize(description) if word not in stop_words]

query_vectors = [model[word] for word in query_tokens if word in model]
description_vectors = [model[word] for word in description_tokens if word in model]

similarity = cosine_similarity(query_vectors, description_vectors)

print("Similarity:", similarity)
