
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define the two sentences
#query = "TESOURA IRIS RETA 115MM" 0.55
query= "TESOURA FINA PCT BICUDA 115MM"
description = "TTESOURA FINA P.RCT.BICUDA 115MM"

# Convert the sentences to embeddings
embeddings1 = model.encode(query)
embeddings2 = model.encode(description)

# Calculate the cosine similarity between the embeddings
cosine_similarity = util.cos_sim(embeddings1, embeddings2)

print("Cosine Similarity:", cosine_similarity.item())