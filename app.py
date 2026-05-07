from flask import Flask, render_template, request
import pandas as pd
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

CSV_URL = "movies_metadata (2).csv"

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "minimax-m2.1:cloud"

movies_df = None
movie_embeddings = None


def get_embedding(text):
    payload = {
        "model": EMBED_MODEL,
        "prompt": text
    }

    response = requests.post(OLLAMA_EMBED_URL, json=payload, timeout=60)
    response.raise_for_status()

    return response.json()["embedding"]


def load_movies():
    df = pd.read_csv(CSV_URL, low_memory=False)

    useful_columns = ["title", "overview", "genres", "release_date", "vote_average"]
    df = df[useful_columns].copy()

    df["title"] = df["title"].fillna("Unknown Title")
    df["overview"] = df["overview"].fillna("")
    df["genres"] = df["genres"].fillna("")
    df["release_date"] = df["release_date"].fillna("Unknown")
    df["vote_average"] = df["vote_average"].fillna("N/A")

    df = df[df["overview"].str.strip() != ""].head(1000)

    df["rag_text"] = (
        "Title: " + df["title"].astype(str) + "\n" +
        "Overview: " + df["overview"].astype(str) + "\n" +
        "Genres: " + df["genres"].astype(str) + "\n" +
        "Release Date: " + df["release_date"].astype(str) + "\n" +
        "Vote Average: " + df["vote_average"].astype(str)
    )

    return df


def build_embeddings():
    global movies_df, movie_embeddings

    movies_df = load_movies()
    embeddings = []

    print("Building movie embeddings. This may take a little while...")

    for text in movies_df["rag_text"]:
        embedding = get_embedding(text)
        embeddings.append(embedding)

    movie_embeddings = np.array(embeddings)

    print("Embeddings are ready.")


def retrieve_movies(question, top_k=5):
    question_embedding = np.array(get_embedding(question)).reshape(1, -1)

    similarities = cosine_similarity(question_embedding, movie_embeddings).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]

    results = movies_df.iloc[top_indices].copy()
    results["score"] = similarities[top_indices]

    return results


def build_context(rows):
    context = []

    for _, row in rows.iterrows():
        context.append(
            f"Title: {row['title']}\n"
            f"Overview: {row['overview']}\n"
            f"Genres: {row['genres']}\n"
            f"Release Date: {row['release_date']}\n"
            f"Vote Average: {row['vote_average']}\n"
        )

    return "\n---\n".join(context)


def ask_ollama(question, context):
    prompt = f"""
You are a movie metadata RAG assistant.

Answer the user's question using only the retrieved movie context below.
If the retrieved context is not enough to answer, say that clearly.
Do not invent movies or facts that are not in the context.

Retrieved Context:
{context}

User Question:
{question}

Grounded Answer:
"""

    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "No answer returned.")
    except requests.exceptions.RequestException as e:
        return f"Ollama/API error: {e}"


@app.route("/", methods=["GET", "POST"])
def index():
    question = ""
    answer = None
    retrieved_rows = None

    if request.method == "POST":
        question = request.form.get("question", "").strip()

        if question == "":
            answer = "Please enter a movie-related question."
        else:
            try:
                retrieved_rows = retrieve_movies(question)
                context = build_context(retrieved_rows)
                answer = ask_ollama(question, context)
            except Exception as e:
                answer = f"Error: {e}"

    return render_template(
        "index.html",
        question=question,
        answer=answer,
        retrieved_rows=retrieved_rows
    )


if __name__ == "__main__":
    build_embeddings()
    app.run(debug=True, port=5005)
