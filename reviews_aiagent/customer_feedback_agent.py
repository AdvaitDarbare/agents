# -------------------------
# Imports
# -------------------------

import os  # for loading environment variables
import time  # optional: could be used for delays or timing
import json  # for printing JSON output
import string  # for text preprocessing
import requests  # to make HTTP requests to Apify
import numpy as np  # numerical operations
import faiss  # for fast similarity search over dense vectors

# External libraries
from dotenv import load_dotenv  # to load API keys from a .env file
from openai import OpenAI  # OpenAI API client
from sklearn.feature_extraction.text import TfidfVectorizer  # text vectorization
from sklearn.cluster import KMeans  # clustering reviews
from sentence_transformers import SentenceTransformer  # to generate dense embeddings

# -------------------------
# Step 1: Load environment variables & API keys
# -------------------------

load_dotenv()  # Load variables from .env

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_TOKEN = os.getenv("APIFY_API_TOKEN")

client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------
# Step 2: Set up Apify parameters
# -------------------------

ACTOR_ID = 'junglee~amazon-reviews-scraper'
PRODUCT_URL = 'https://www.amazon.com/dp/B08BHHSB6M'  # Product page (not used in script)
MAX_REVIEWS = 5
BASE_URL = 'https://api.apify.com/v2'

# -------------------------
# Step 3: Fetch latest actor run and get dataset
# -------------------------

# Get latest run for the actor
runs_url = f"{BASE_URL}/acts/{ACTOR_ID}/runs/last?token={API_TOKEN}"
runs_response = requests.get(runs_url)
run_data = runs_response.json().get("data", {})

# Get dataset ID from the run
dataset_id = run_data.get("defaultDatasetId", None)
if not dataset_id:
    raise Exception("Dataset ID not found. Ensure that the actor run produced a dataset.")

# Fetch reviews from dataset
dataset_items_url = f"{BASE_URL}/datasets/{dataset_id}/items?token={API_TOKEN}&limit={MAX_REVIEWS}"
print("Fetching up to 5 scraped reviews...")
dataset_response = requests.get(dataset_items_url)
review_items = dataset_response.json()

# Print raw review data
print("Scraped Reviews (limit 5):")
print(json.dumps(review_items, indent=2))

# -------------------------
# Step 4: Text Analysis and Clustering
# -------------------------

# Extract review descriptions or titles
reviews_texts = []
for review in review_items:
    text = review.get("reviewDescription") or review.get("reviewTitle") or ""
    reviews_texts.append(text)

print("\nExtracted Review Texts:")
for text in reviews_texts:
    print(text)

# Basic preprocessing: lowercase and remove punctuation
def preprocess(text):
    return text.lower().translate(str.maketrans("", "", string.punctuation))

cleaned_reviews = [preprocess(text) for text in reviews_texts]

# Convert to TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(cleaned_reviews)

# Cluster reviews into 1–2 groups
n_clusters = 2 if len(cleaned_reviews) >= 2 else 1
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X)

# Show clustering results
print("\nClustering Results:")
for i, text in enumerate(reviews_texts):
    print(f"Review (first 50 chars): {text[:50]}... -> Cluster: {clusters[i]}")

# Function to get top TF-IDF keywords per cluster
def get_top_keywords(cluster_label, n_keywords=3):
    cluster_indices = [i for i, label in enumerate(clusters) if label == cluster_label]
    if not cluster_indices:
        return []
    X_cluster = X[cluster_indices]
    centroid = X_cluster.mean(axis=0).tolist()[0]
    indices = sorted(range(len(centroid)), key=lambda i: centroid[i], reverse=True)[:n_keywords]
    return [vectorizer.get_feature_names_out()[i] for i in indices]

# Print top keywords for each cluster
print("\nTop Keywords per Cluster:")
for cl in set(clusters):
    print(f"Cluster {cl}: {get_top_keywords(cl)}")

# -------------------------
# Step 5: Use OpenAI LLM to suggest product improvements
# -------------------------

# Use GPT to suggest improvements based on clustered reviews
def suggest_improvements(reviews, cluster_id):
    text_sample = "\n".join(reviews)
    prompt = (
        f"Review the following customer feedback for cluster {cluster_id}:\n\n"
        f"{text_sample}\n\n"
        "Based on this feedback, provide a list of specific, actionable suggestions to improve the product. "
        "For example, if customers mention battery issues or design flaws, suggest changes or enhancements.\n\n"
        "Suggestions:"
    )

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert product strategist helping to derive actionable improvement suggestions from customer feedback."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

# Map reviews to their clusters
cluster_reviews = {cl: [] for cl in set(clusters)}
for idx, label in enumerate(clusters):
    cluster_reviews[label].append(reviews_texts[idx])

# Generate and print improvement suggestions
print("\nImprovement Suggestions for each Cluster:")
for cl, reviews_list in cluster_reviews.items():
    suggestions = suggest_improvements(reviews_list, cl)
    print(f"\nSuggestions for Cluster {cl}:\n{suggestions}\n")

# -------------------------
# Step 6: Compute Dense Embeddings & Store in FAISS
# -------------------------

# Load sentence transformer model
dense_model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate dense embeddings from cleaned review text
dense_embeddings = dense_model.encode(cleaned_reviews, convert_to_numpy=True)
embedding_dim = dense_embeddings.shape[1]

# Create FAISS index (Euclidean distance)
faiss_index = faiss.IndexFlatL2(embedding_dim)
faiss_index.add(dense_embeddings)

print("\nDense embeddings computed and stored in FAISS index.")
