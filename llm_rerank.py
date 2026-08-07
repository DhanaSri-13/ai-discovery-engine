import os
from dotenv import load_dotenv
from google import genai
from recommend import recommend

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def rerank_with_llm(query_text, clicked_ids, top_k=5):
    candidates = recommend(query_text=query_text, clicked_ids=clicked_ids, top_k=10)

    candidate_list = "\n".join([
        f"- ID {row['id']}: {row['name']} | {row['brand']} | ₹{row['price']} | {row['color']}"
        for _, row in candidates.iterrows()
    ])

    prompt = f"""A user searched for: "{query_text}"
They previously clicked on products with these IDs: {clicked_ids}

Here are 10 candidate products:
{candidate_list}

Pick the best 5 for this user and explain in one short sentence each why you recommend it.
Respond ONLY with valid JSON in this exact format, no other text:
[{{"id": 1, "name": "...", "reason": "..."}}, ...]
"""

    response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt
)
    return response.text

if __name__ == "__main__":
    result = rerank_with_llm(query_text="comfortable footwear", clicked_ids=[1, 2, 5])
    print(result)