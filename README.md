# AI Product Discovery Engine

An LLM + RAG powered product discovery system that recommends products based on user search queries, click behavior, purchase history, and product similarity.

## How it works

1. **Product embeddings** — every product's text (name, category, brand, color, material, description) is converted into a vector using `sentence-transformers`.
2. **Search** — a user's search query is embedded the same way, then compared against all product vectors using cosine similarity to find the closest matches.
3. **Click tracking** — as a user clicks products, their embeddings are averaged into a "session vector" representing current interest.
4. **Recommendation blending** — search intent and click behavior are combined into one vector to produce personalized results.
5. **Purchase history** — a co-occurrence table tracks which products are frequently bought together, shown as a separate "Frequently bought together" section.
6. **LLM reranking (RAG)** — the top candidate products are passed to an LLM (Google Gemini), which picks the best matches and generates a natural-language explanation for each.

## Tech stack

- Python 3.13
- `sentence-transformers` (all-MiniLM-L6-v2) — text embeddings
- `scikit-learn` — cosine similarity search
- `pandas` / `numpy` — data handling
- `streamlit` — interactive UI
- Google Gemini API (free tier) — LLM reranking and explanations

## Project status

- [x] Product data prepared (`products.csv`)
- [x] Embedding pipeline (`embed.py`)
- [x] Text-based search (`search.py`)
- [x] Click-based session vector (`session.py`)
- [x] Combined recommendation engine (`recommend.py`)
- [x] Purchase history / co-occurrence logic (`purchase_history.py`)
- [x] LLM reranking layer (RAG) (`llm_rerank.py`)
- [x] Streamlit UI (`app.py`)

## Notes on implementation choices

- **LLM provider:** Using Google Gemini API (free tier) instead of OpenAI/Anthropic due to free-tier availability without requiring billing setup.
- **Product images:** Sourced dynamically by category using a placeholder image service, since this project uses simulated product data rather than a real product catalog with real photos.

## Setup

\`\`\`bash
python -m venv venv
venv\Scripts\Activate.ps1     # Windows
pip install -r requirements.txt
\`\`\`

Create a `.env` file in the project root with your Gemini API key:
\`\`\`
GEMINI_API_KEY=your-key-here
\`\`\`

## Usage

\`\`\`bash
python embed.py           # generate product embeddings (run once, or after data changes)
python search.py          # test text-based search from the terminal
streamlit run app.py      # launch the full interactive app
\`\`\`
