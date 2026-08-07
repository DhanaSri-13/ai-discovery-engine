from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

print("Loading embedding model... (this may take a moment the first time)")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Reading products.csv...")
products = pd.read_csv('products.csv')

products['combined_text'] = (
    products['name'].astype(str) + ' ' +
    products['category'].astype(str) + ' ' +
    products['brand'].astype(str) + ' ' +
    products['color'].astype(str) + ' ' +
    products['material'].astype(str) + ' ' +
    products['description'].astype(str)
)

print("Generating embeddings for all products...")
embeddings = model.encode(products['combined_text'].tolist(), show_progress_bar=True)

print("Embeddings shape:", embeddings.shape)

np.save('product_embeddings.npy', embeddings)
products.to_csv('products_processed.csv', index=False)

print("Done! Saved product_embeddings.npy and products_processed.csv")