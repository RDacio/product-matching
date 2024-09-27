
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

#Modelo - generico 
model = SentenceTransformer('all-MiniLM-L6-v2')

#query = "TESOURA IRIS RETA 115MM" 0.55
query= "TESOURA FINA PCT BICUDA 115MM"
description = "TTESOURA FINA P.RCT.BICUDA 115MM"

#Convert
embeddings1 = model.encode(query)
embeddings2 = model.encode(description)

#cosine similarity
cosine_similarity = util.cos_sim(embeddings1, embeddings2)

print("Cosine Similarity:", cosine_similarity.item())