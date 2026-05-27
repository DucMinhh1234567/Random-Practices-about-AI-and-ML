from datetime import date
import json
import numpy as np
from sentence_transformers import SentenceTransformer
 
# Restore the embedding matrix, deserialize the metadata, rebuild the index
embeddings_loaded = np.load("ticket_embeddings.npy")
 
with open("ticket_metadata.json") as f:
    tickets_loaded = json.load(f)
for t in tickets_loaded:
    t["created"] = date.fromisoformat(t["created"])
 
model = SentenceTransformer("all-MiniLM-L6-v2")
index = ContextAwareIndex(embeddings_loaded, tickets_loaded)
 
print(f"Reloaded: {embeddings_loaded.shape[0]} docs, {embeddings_loaded.shape[1]}D.")