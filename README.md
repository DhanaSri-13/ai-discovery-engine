# AI Product Discovery Engine

An LLM + RAG powered product discovery system that recommends products based on user search queries, click behavior, purchase history, and product similarity.

## How it works

1. **Product embeddings** — every product's text (name, category, brand, color, material, description) is converted into a vector using `sentence-transformers`.
2. **Search** — a user's search query is embedded the same way, then compared against all product vectors using cosine similarity to find the closest matches.
3. **Click tracking (in progress)** — as a user clicks products, their embeddings are averaged into a "session vector" representing current interest.
4. **Recommendation blending (in progress)** — search intent and click behavior are combined into one vector to produce personalized results.

## Tech stack

- Python 3.13
- `sentence-transformers` (all-MiniLM-L6-v2) — text embeddings
- `scikit-learn` — cosine similarity search
- `pandas` / `numpy` — data handling
- Streamlit (planned) — UI
- Using Google Gemini API (free tier) — LLM reranking and explanations

## Project status

- [x] Product data prepared (`products.csv`)
- [x] Embedding pipeline (`embed.py`)
- [x] Text-based search (`search.py`)
- [x] Click-based session vector (`session.py`)
- [x] Combined recommendation engine (`recommend.py`)
- [x] Purchase history / co-occurrence logic
- [x] LLM reranking layer (RAG)
- [ ] Streamlit UI

## Setup

\`\`\`bash
python -m venv venv
venv\\Scripts\\Activate.ps1     # Windows
pip install -r requirements.txt
\`\`\`

## Usage

\`\`\`bash
python embed.py       # generate product embeddings (run once, or after data changes)
python search.py      # test text-based search
\`\`\`
