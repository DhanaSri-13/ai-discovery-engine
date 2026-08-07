from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

print("Loading model and data...")
model = SentenceTransformer('all-MiniLM-L6-v2')
products = pd.read_csv('products_processed.csv')
product_embeddings = np.load('product_embeddings.npy')

def search(query_text, top_k=5):
    query_vector = model.encode([query_text])[0]
    scores = cosine_similarity([query_vector], product_embeddings)[0]
    top_indices = scores.argsort()[::-1][:top_k]

    results = products.iloc[top_indices].copy()
    results['similarity_score'] = scores[top_indices]
    return results[['name', 'category', 'brand', 'price', 'color', 'similarity_score']]

# Test it
if __name__ == "__main__":
    query = input("Search for a product: ")
    results = search(query)
    print("\nTop matches:\n")
    print(results.to_string(index=False))