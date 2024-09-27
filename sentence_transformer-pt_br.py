
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel

#Model - pt-br
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

query= "TESOURA FINA PCT BICUDA 115MM"
description = "TESOURA FINA P.RCT.BICUDA 115MM"

#convert
client_embed = model.encode(query)
description_embed = model.encode(description)

#cosine similarity
cosine_similarity = util.cos_sim(client_embed, description_embed)

print("Cosine Similarity:", cosine_similarity.item())