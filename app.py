from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import re
import os

# -------------------------
# INITIAL SETUP
# -------------------------

print("Starting Semantic Search API...")

# Load FAISS index
index = faiss.read_index("product_index.faiss")

# Load product data
products = pickle.load(open("products.pkl", "rb"))

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
print("GROQ KEY PRESENT:", os.getenv("GROQ_API_KEY") is not None)

# Create FastAPI app
app = FastAPI()

# -------------------------
# ENABLE CORS
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# HELPER FUNCTIONS
# -------------------------

def parse_query(query: str):
    budget = None
    match = re.search(r'under\s?(\d+)', query)
    if match:
        budget = int(match.group(1))

    return {
        "budget": budget,
        "use_case": query
    }


def llm_explanation(product, query):
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": f"""
User query: {query}

Product:
Name: {product['name']}
Brand: {product['brand']}
Rating: {product['rating']}

Explain in 1–2 lines why this product matches the user's intent.
"""
                }
            ],
            max_tokens=80,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return "AI explanation unavailable"


# -------------------------
# API ROUTES
# -------------------------

@app.get("/")
def home():
    return {"status": "Semantic Search API is running"}

@app.get("/search")
def search(query: str):
    parsed = parse_query(query)

    query_embedding = model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, 10)

    results = []

    for i, idx in enumerate(indices[0]):
        product = products.iloc[idx]

        relevance_score = float(
            round(max(0.0, 100.0 - float(distances[0][i])), 2)
        )

        rating = product["reviews.rating"]
        rating = float(rating) if rating == rating else None

        product_data = {
            "name": str(product["name"]),
            "brand": str(product["brand"]),
            "rating": rating,
            "relevance": relevance_score,
        }

        # ✅ LLM FOR ALL RESULTS
        product_data["why_matched"] = llm_explanation(
            product_data, query
        )

        results.append(product_data)

    return {
        "query": query,
        "parsed_query": parsed,
        "processing_status": "Indexed 57,278 products",
        "results": results
    }
