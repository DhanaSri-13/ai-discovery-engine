import pandas as pd
import numpy as np

products = pd.read_csv('products_processed.csv')
product_embeddings = np.load('product_embeddings.npy')

def get_session_vector(clicked_ids):
    """
    Given a list of product IDs the user clicked,
    return the average embedding representing their current interest.
    """
    indices = products[products['id'].isin(clicked_ids)].index
    if len(indices) == 0:
        return None
    return product_embeddings[indices].mean(axis=0)

# Quick test
if __name__ == "__main__":
    # Simulate a user clicking Nike shoes, then Adidas shoes, then something else
    clicked_ids = [1, 2, 3]  # adjust to real IDs from your CSV
    vec = get_session_vector(clicked_ids)
    if vec is not None:
        print("Session vector generated, shape:", vec.shape)
    else:
        print("No matching products found for those IDs.")