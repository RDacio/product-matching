from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

#Model - pt-br(bert)
model = SentenceTransformer('neuralmind/bert-base-portuguese-cased')

query= "TESOURA FINA PCT BICUDA 115MM"
description = "TESOURA FINA P.RCT.BICUDA 115MM"

#Convert
client_embed = model.encode(query)
description_embed = model.encode(description)

#Cosine similarity
cosine_similarity = util.cos_sim(client_embed, description_embed)

print("Cosine Similarity:", cosine_similarity.item())