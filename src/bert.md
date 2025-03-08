# ColBERT: Contextualized Late Interaction over BERT

## References
https://jina.ai/news/what-is-colbert-and-late-interaction-and-why-they-matter-in-search/

## Introduction
ColBERT (stands for Contextualized Late Interaction over BERT), is a dense retrieval model designed to efficiently perform information retrieval by leveraging the representations learned by BERT.

ColBERT aims to combine the contextual power of BERT-style models with the efficiency needed for large-scale document retrieval.

## What's the Problem ColBERT is Solving?
Traditionally, information retrieval systems like BM25 (a popular keyword-based retrieval algorithm) perform well in finding relevant documents but struggle when complex semantics or context are necessary to understand a user query in relation to the documents.

On the other hand, dense retrieval models leverage powerful transformer-based models like BERT to produce more contextually aware representations. However, existing dense retrieval approaches can become extremely computationally expensive when handling large document collections because they often rely on heavy cross-attention mechanisms between the query and each document in the collection. This results in slow latency when ranking documents.

The key problem ColBERT addresses is this trade-off between efficiency (speed) and effectiveness (accuracy). That is, ColBERT aims to enable efficient retrieval while still leveraging powerful BERT-based contextual embeddings for better document ranking.

## High-Level Overview of ColBERT
ColBERT introduces an innovative architecture that leverages BERT-style embeddings while utilizing a "Late Interaction" technique to maintain efficiency.

Here's a step-by-step breakdown of how standard BERT-based retrieval systems work, and how ColBERT differs:

### 1. Full-Interaction Models (Traditional BERT for Retrieval):
In many dense retrieval settings using BERT, a query and a document are passed through BERT together to produce a score for their relevance. This requires "cross-attention" between the query and the document, meaning that each token in the query attends to every token in the document, which results in:
- High computational cost.
- Query-document interactions can only be computed at re-ranking time, making it impractical for large datasets, as you can't precompute huge document embeddings independently of a query.

### 2. ColBERT's Late Interaction Mechanism:
ColBERT is designed to sidestep the computational issues associated with cross-attention by decoupling query and document contextualization using BERT (or any transformer-based model). Each query and each document is encoded independently, meaning that you avoid the expensive joint BERT computation for each query-document pair. The result is that you can precompute document embeddings once and use them to quickly match any future queries.

The Late Interaction mechanism allows ColBERT to:
- Precompute document representations offline, which is great for efficiency.
- Perform interaction between the query and document embeddings at inference time, at the granularity of individual terms, without the need for full cross-attention (e.g., through costly matrix operations).

## Architecture of ColBERT
Let's walk through the architecture in more detail:

### 1. BERT Encoding Stage (Contextualized Embeddings):

#### Query Processing:
- A query (e.g., "What is machine learning?") is tokenized and processed independently by BERT.
- BERT produces a set of contextualized embeddings for each query token.
- Example: "What" → embedding_1, "is" → embedding_2, "machine" → embedding_3, etc.

#### Document Processing:
- Similarly, each document is processed separately by BERT.
- Each document is tokenized into word or subword tokens and passed through BERT, which generates contextualized embeddings for each token, just like for queries.
- Since this "document encoding" operation can be done offline (precomputed), it doesn't need to be done during real-time query evaluation, making retrieval more efficient during inference.

At this point, both the query and documents are represented by sets of dense contextualized vectors, but there hasn't yet been any interaction between the query and document.

### 2. Late Interaction Step
Unlike traditional retrieval approaches (e.g., vanilla BERT retrieval) where query-document interaction happens during the transformer encoding process (via cross-attention), ColBERT uses a form of lightweight interaction mechanism at the end of the pipeline.

Specifically, ColBERT performs MaxSim matching between the embeddings of the query tokens and the embeddings of the document tokens.

#### a. MaxSim (Maximum Similarity):
After BERT has independently encoded the query and the document into their respective embeddings, ColBERT uses MaxSim to calculate the match between each query token embedding and the best (i.e., most similar) document token embedding.

For each token in the query, you compute the maximum similarity it has with any token in the document:
- For example, for a query token "machine", ColBERT will compute the cosine similarity (or dot product) between "machine" and all tokens in the document, and simply take the maximum similarity score.
- This process is repeated for all query tokens, so that each query token is matched to the most relevant token in the document.

#### b. Final Relevance Score (Aggregation):
After computing these MaxSim scores for each query token, the final query-document interaction score is computed as an aggregate of all these MaxSim scores (typically using summation).

The intuition behind this is that we care about how well each query token is covered by some part of the document, and taking the maximum ensures that the best possible match is taken into account.

Once each query token has matched to its best corresponding document token, those scores are aggregated to produce a final relevance score for the query-document pair.

### Why "Late Interaction"?
The name "late interaction" comes from the fact that the interaction between a query and document happens very "late" in the pipeline, only in the final stage, based on the token embeddings, not during the expensive BERT encoding process.

This results in considerable computational savings and the ability to manipulate and index document embeddings efficiently for retrieval.

## Key Features of ColBERT

### a. Efficiency:
Since ColBERT decouples the computation of the query and the document embeddings, document representations can be precomputed and stored for future queries. When a new query comes in, you only need to process the query and then perform inexpensive MaxSim operations with stored document embeddings.

This is a huge win in scalability compared to traditional dense retrieval models (e.g., vanilla BERT, where you would have to reprocess both the query and the document jointly during retrieval time.

### b. Contextualization:
By leveraging BERT, ColBERT produces contextualized embeddings. Thus, the model can capture the rich semantics and dependencies that original retrieval algorithms like BM25 (which are based purely on keyword matching) struggle to encode. This makes the system more powerful for tasks requiring an understanding of nuanced language or complex queries.

### c. Effectiveness:
With the MaxSim mechanism, ColBERT ensures that each query token is matched with the most relevant document token, making it well-suited for scoring relevance.

The combination of the fine-grained token-level interaction and the efficiency-preserving late interaction paradigm results in ColBERT performing well for tasks like passage retrieval and document search in large datasets.

## Summary of ColBERT's Key Innovation:
- **Late Interaction**: Instead of using full cross-attention between query and document tokens inside the BERT layers (like in early dense retrieval models), ColBERT decouples query and document encoding by using independent BERT passes for both, followed by MaxSim-based ranking at the token level in the final step.

- **Efficiency through Precomputation**: ColBERT's approach allows precomputed contextualized document embeddings, which means that these embeddings can be stored and reused for future queries. When a new query comes in, you only need to compute the query embeddings and perform a fast late interaction to find the most relevant document embeddings.

- **Token-level Interaction**: Instead of treating entire documents as homogeneous units, ColBERT leverages token-level matches between queries and documents. Each query token is compared to every document token in the final retrieval step, ensuring that fine-grained interactions are preserved.

## Use Cases for ColBERT:
- **Document Retrieval**: Finding relevant documents for a given query from a large corpus of documents.
- **Passage Retrieval**: Retrieving small sections (passages) of documents that are most relevant to a query.
- **Question Answering**: ColBERT is ideal for retrieving paragraphs or documents that are relevant to a specific question, which can then be passed to a QA model for extracting specific answers.

## Core Differences Between Embedding-Based Retrieval and ColBERT's Late Interaction:

| Aspect | Embedding-Based Retrieval | Late Interaction ColBERT |
|--------|---------------------------|--------------------------|
| Query-Document Interaction | Interaction happens only in vector space. The embeddings (for both document and query) are compared as holistic chunks | Interaction happens in a token-level, granular stage. Query tokens are compared to document tokens late in the process post-BERT encoding |
| Document Representation | Documents are embedded into a single dense vector that represents the full document chunk | Documents are represented as multiple dense vectors (one for each token in the document) |
| Efficiency | Efficient because document embeddings can be precomputed; but might lose fine-grained relevancy | More computationally heavy, but captures more fine-grained token-based matching and interactions |
| Precomputation of Document Embeddings | Yes, document embeddings can be precomputed and stored in a vector DB (e.g., FAISS) | Yes, document token representations can also be precomputed, but interaction happens at a finer grain (token level) |

## ColBERT v2 Improvements Over v1

- **Embedding Compression**
  - v2 uses learned CNN-based compression
  - Reduces embedding size from 768D to 32D
  - Significant reduction in storage requirements

- **Asymmetric In-batch Negative Training**
  - More efficient use of training data
  - Improves model's discriminative power

- **Residual Compression**
  - Preserves more information during compression
  - Enhances retrieval quality despite dimensionality reduction

- **End-to-End Training**
  - Jointly optimizes encoding and compression
  - Better alignment between compressed representations and retrieval task

- **Pruning Mechanism**
  - Removes less important tokens from document representations
  - Further reduces storage and improves retrieval speed

- **Query Augmentation**
  - Enhances query robustness during training
  - Improves model's ability to handle diverse queries

- **Scalability**
  - Compressed embeddings allow for larger document collections
  - More practical for real-world, large-scale applications

- **Performance**
  - Achieves better retrieval accuracy despite compression
  - Faster inference due to reduced embedding dimensions

- **Implementation Improvements**
  - Enhanced tooling and easier deployment
  - Better integration with existing IR systems

## Denoised Supervision in ColBERT v2

### Overview
Denoised Supervision is a technique used in ColBERT v2 to improve the quality of training signals in information retrieval, reducing noise in supervision data and enhancing model performance.

### Key Aspects

#### 1. Motivation
- Traditional IR often uses binary relevance labels (relevant/not relevant)
- These labels can be noisy or incomplete, especially in large-scale datasets

#### 2. Core Idea
- Assigns different weights or importances to training examples
- Identifies which positive examples are likely truly relevant
- Determines which negative examples are more informative for training

#### 3. Implementation in ColBERT v2
- Uses "Asymmetric In-batch Negatives"

### How it Works

#### 1. In-batch Negatives
- Each batch contains multiple queries and their positive documents
- Other documents in the batch serve as negative examples

#### 2. Asymmetric Scoring
- Computes relevance scores between each query and all batch documents
- Positive pairs scored using full ColBERT model
- Negative pairs scored with simplified, efficient function

#### 3. Dynamic Weighting
- Adjusts importance of each training example based on scores
- Emphasizes hard negatives (high-scoring but not labeled positive)
- De-emphasizes easy negatives (clearly don't match the query)

#### 4. Noise Reduction
- Focuses on hard negatives and potentially misclassified positives
- Helps model make finer distinctions
- Reduces impact of noisy or incorrectly labeled data

### Benefits
- **Improved Discrimination**: Better distinction between closely related documents
- **Robustness**: More resilient to noise in training data
- **Efficiency**: More effective learning from each data batch
- **Better Generalization**: Improved performance on unseen queries and documents

### Example
Query: "python programming language"
- Positive: Detailed article about Python (high importance)
- Hard negative: Article about programming in general (increased importance)
- Easy negative: Article about snakes (reduced importance)

## Conclusion
Denoised Supervision in ColBERT v2 dynamically adjusts training example importance, focusing on informative distinctions and leading to better overall performance in information retrieval tasks.