import json
import numpy as np
import fastembed
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Function to split text into chunks
def split_text_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 50) -> list:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return text_splitter.split_text(text)

# Load dataset and split into chunks
with open('src/bert.md', 'r', encoding="utf-8") as f:
    text = f.read()

chunks = split_text_into_chunks(text)

# Load test data (questions and expected chunks)
with open('src/test_data.json', 'r', encoding="utf-8") as f:
    test_data = json.load(f)

questions = [entry["question"] for entry in test_data]
expected_chunks = [entry["answer"] for entry in test_data]

text_embedding_support_model = fastembed.TextEmbedding.list_supported_models()

# save names into a text file for later use
# with open('src/text_embedding_support_models.txt', 'w') as f:
#     for model in text_embedding_support_model:
#         f.write(model["model"] + "\n")

try:
    with open('src/results.json', 'r') as f:
        existing_results = json.load(f)
except FileNotFoundError:
    existing_results = {}
            
for model_data in text_embedding_support_model:
    model_name = model_data["model"]
    if model_name in existing_results:
        print(f"Skipping model {model_name} as it has already been evaluated.")
        continue

    # Initialize FastEmbed Model
    # model_name = "BAAI/bge-large-en-v1.5"
    try:
        embed_model = fastembed.TextEmbedding(model_name=model_name)

        # Generate embeddings for chunks (Fix: Convert generator to NumPy array)
        chunk_embeddings = np.array(list(embed_model.embed(chunks)))

        # Cosine Similarity Function
        def cosine_similarity(A, B):
            return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

        # Evaluate retrieval performance
        retrieved_chunks = []
        correct = 0
        top_k = 3  # Set Top-K for evaluation

        for i, question in enumerate(questions):
            # Convert question to embedding
            q_embedding = np.array(list(embed_model.embed([question]))[0])

            # Compute cosine similarity with each chunk
            similarities = np.array([cosine_similarity(chunk_emb, q_embedding) for chunk_emb in chunk_embeddings])

            # Find the best-matching chunk
            best_chunk_idx = np.argmax(similarities)
            retrieved_chunk = chunks[best_chunk_idx]
            retrieved_chunks.append(retrieved_chunk)

            # Exact Match Evaluation
            if retrieved_chunk.strip() == expected_chunks[i].strip():
                correct += 1

        # Compute Accuracy Metrics
        exact_match_accuracy = correct / len(questions)

        # Compute Top-K Accuracy
        top_k_correct = 0
        for i, question in enumerate(questions):
            q_embedding = np.array(list(embed_model.embed([question]))[0])
            similarities = np.array([cosine_similarity(chunk_emb, q_embedding) for chunk_emb in chunk_embeddings])
            top_k_indices = np.argsort(similarities)[-top_k:][::-1]  # Top-K indices

            if any(chunks[idx].strip() == expected_chunks[i].strip() for idx in top_k_indices):
                top_k_correct += 1

        top_k_accuracy = top_k_correct / len(questions)

        # Print results
        print("Results for {}".format(model_name))
        print(f"Exact Match Accuracy: {exact_match_accuracy * 100:.2f}%")
        print(f"Top-{top_k} Accuracy: {top_k_accuracy * 100:.2f}%")

        # Save results to JSON
        results = {
            model_name: {
            "exact_match_accuracy": round(exact_match_accuracy, 2),
            f"top_{top_k}_accuracy": round(top_k_accuracy, 2)
            }
        }

        # Update results with new data
        existing_results.update(results)

        # Save updated results to JSON file
        with open('src/results.json', 'w') as f:
            json.dump(existing_results, f, indent=4)
    except Exception as e:
        print(f"Error occurred for model {model_name}: {e}")
