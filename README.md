# Movie Metadata RAG Assistant

## Overview
This project is a Flask-based web application that implements a Retrieval-Augmented Generation (RAG) system using Ollama. The application allows users to ask movie-related questions and returns answers grounded in a dataset of movie metadata.

The system retrieves the most relevant movie records using embedding similarity and then sends the retrieved context to an Ollama-hosted language model to generate a concise, grounded response.

---

## Features
- Retrieval-Augmented Generation (RAG) pipeline
- Embedding-based search using Ollama (`nomic-embed-text`)
- LLM response generation using Ollama (`minimax-m2.1:cloud`)
- Flask web interface for user interaction
- Displays retrieved movie rows with similarity scores
- Provides grounded answers based only on retrieved data

---

## Project Structure
Marino_Movie_metadata/
│
├── app.py
├── requirements.txt
├── README.md
├── movies_metadata (2).csv
└── templates/
└── index.html

---

## Setup Instructions

### 1. Clone the Repository
git clone https://github.com/aleysha23/Marino_Movie_metadata.git

cd Marino_Movie_metadata

### 2. Install Dependencies
pip install -r requirements.txt

### 3. Install and Run Ollama
Make sure Ollama is installed and running.

Pull required models:
ollama pull nomic-embed-text
ollama pull minimax-m2.1:cloud

Start Ollama server:
ollama serve

---

## Running the Application

In the project directory, run:
python app.py

Then open your browser and go to:
http://127.0.0.1:5005/


---

## How It Works

1. User enters a movie-related question.
2. The app generates an embedding for the question using Ollama.
3. The app compares the question embedding to movie embeddings.
4. The top 5 most similar movie records are retrieved.
5. Retrieved data is formatted into a context block.
6. The context and question are sent to the Ollama LLM.
7. The LLM returns a grounded answer.

---

## Example Query

**Input:**
Find movies about animals


**Output:**
- Table of retrieved movie records (title, overview, genres, etc.)
- Generated answer based only on retrieved context

---

## Technologies Used
- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- Ollama (LLM + embeddings)

---

## Notes
- The application runs locally on port 5005.
- Embeddings are generated on startup and may take a few minutes initially.
- The model is instructed to only answer using retrieved context to prevent hallucinations.

---

## Author
Aleysha Marino