import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from purchase_history import load_co_purchase_table, get_also_bought
from llm_rerank import rerank_with_llm
import json

st.set_page_config(page_title="AI Product Discovery", layout="wide")

# ---------- Load everything once ----------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_data():
    products = pd.read_csv('products_processed.csv')
    embeddings = np.load('product_embeddings.npy')
    co_purchase = load_co_purchase_table()
    return products, embeddings, co_purchase

model = load_model()
products, product_embeddings, co_purchase = load_data()

# ---------- Session state (memory across reruns) ----------
if 'clicked_ids' not in st.session_state:
    st.session_state.clicked_ids = []

# ---------- Core recommend function ----------
def recommend(query_text="", clicked_ids=None, top_k=5):
    clicked_ids = clicked_ids or []
    query_vec = model.encode([query_text])[0] if query_text.strip() else np.zeros(384)

    if clicked_ids:
        indices = products[products['id'].isin(clicked_ids)].index
        session_vec = product_embeddings[indices].mean(axis=0) if len(indices) > 0 else np.zeros(384)
    else:
        session_vec = np.zeros(384)

    final_vec = 0.6 * query_vec + 0.4 * session_vec
    scores = cosine_similarity([final_vec], product_embeddings)[0]
    top_indices = scores.argsort()[::-1][:top_k]
    return products.iloc[top_indices]

# ---------- UI ----------
st.title("🛍️ AI Product Discovery Engine")

query = st.text_input("Search for something", placeholder="e.g. black formal shoes")

# ---------- Recommendations (moved above the grid) ----------
if query or st.session_state.clicked_ids:
    st.subheader("✨ Recommended for you")

    with st.spinner("Thinking..."):
        try:
            llm_result = rerank_with_llm(query, st.session_state.clicked_ids)
            recs = json.loads(llm_result)
            for r in recs:
                st.markdown(f"**{r['name']}** — {r['reason']}")
        except Exception as e:
            st.warning(" ")
            recs = recommend(query, st.session_state.clicked_ids)
            for _, r in recs.iterrows():
                st.markdown(f"**{r['name']}** — ₹{r['price']}")

if st.session_state.clicked_ids:
    st.subheader("🛒 Frequently bought together")
    last_clicked = st.session_state.clicked_ids[-1]
    also_bought = get_also_bought(last_clicked, co_purchase)
    if also_bought:
        for pid, count in also_bought:
            name = products[products['id'] == pid]['name'].values
            if len(name) > 0:
                st.markdown(f"- {name[0]} (bought together {count} times)")
    else:
        st.caption("No purchase history data for this product yet.")

st.divider()

# ---------- Product browsing grid (moved below) ----------
st.subheader("Browse products (click 'View' to simulate interest)")
cols = st.columns(5)
for idx, row in products.iterrows():
    with cols[idx % 5]:
        st.image(row['image_url'], use_container_width=True)
        st.caption(f"{row['name']} — ₹{row['price']}")
        if st.button("View", key=f"click_{row['id']}"):
            if row['id'] not in st.session_state.clicked_ids:
                st.session_state.clicked_ids.append(row['id'])
            st.rerun()