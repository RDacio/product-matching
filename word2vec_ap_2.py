from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

model = KeyedVectors.load_word2vec_format('C:/Users/Rodrigo Dácio/Documents/MyCode/GoogleNews-vectors-negative300.bin', binary=True)

query = "TESOURA FINA PCT BICUDA 115MM"
description = "TTESOURA FINA P.RCT.BICUDA 130MM"

stop_words = set(stopwords.words('portuguese'))

query_words = word_tokenize(query)
description_words = word_tokenize(description)

query_vector = [model[word] for word in query_words if word in model]
description_vector = [model[word] for word in description_words if word in model]

similarity = cosine_similarity(query_vector, description_vector)

print("Similarity:", similarity)
