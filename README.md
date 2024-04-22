# retrieva
Retrieva: Smart Documentation Retrieval !


![Alt text](retrieva_demo.gif)


# Setup
- .env with the openai apikey
- folder with the data to create the embeddings


# Features:

- [x] RAG Pipeline to query documentation using LLamaIndex
   - [ ] prompt engineering (role, output format, negative prompt)
   - [ ] preventing hallucinations ()
- [ ] Prompting through FastAPI with streaming
   - [ ] Middleware for auth
- [ ] Embeddings saved in local weaviate service
   - [ ] Load it! https://docs.llamaindex.ai/en/stable/examples/vector_stores/WeaviateIndexDemo/
- [ ] Containerized deployment