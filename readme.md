embedding, that takes a 

- model
- dataset of documents
- Chunks
- Send LLM → Ask it create question whose answer is the chunk
- Set of Questions
- Each Question → Right Answer   - Test Data

Each Question → RAG (Calculate Embedding) → similarity Search → Find the best chunk. → compare actual output → get accuracy
