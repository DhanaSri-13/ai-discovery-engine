from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')
products = pd.read_csv('products_processed.csv')
product_embeddings = np.load('product_embeddings.npy')

def get_session_vector(clicked_ids):
    indices = products[products['id'].isin(clicked_ids)].index
    if len(indices) == 0:
        return None
    return product_embeddings[indices].mean(axis=0)

def recommend(query_text="", clicked_ids=None, top_k=5):
    clicked_ids = clicked_ids or []

    query_vec = model.encode([query_text])[0] if query_text.strip() else np.zeros(384)
    session_vec = get_session_vector(clicked_ids)
    if session_vec is None:
        session_vec = np.zeros(384)

    # Weighted blend: 60% search intent, 40% click behavior
    final_vec = 0.6 * query_vec + 0.4 * session_vec

    scores = cosine_similarity([final_vec], product_embeddings)[0]
    top_indices = scores.argsort()[::-1][:top_k]

    results = products.iloc[top_indices].copy()
    results['similarity_score'] = scores[top_indices]
    return results[['id', 'name', 'category', 'brand', 'price', 'color', 'similarity_score']]

if __name__ == "__main__":
    # Simulate: user searched "shoes" AND previously clicked sporty items
    results = recommend(query_text="comfortable footwear", clicked_ids=[1, 2, 5])
    print(results.to_string(index=False))