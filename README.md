# retrieva
Retrieva: Smart Documentation Retrieval based on LlamaIndex.

![Alt text](retrieva_demo.gif)


# Setup
- .env file with:

   - OPENAI_API_KEY=<OPENAI_API_KEY>
   - RAG_API_URL=http://localhost:3333  # or whatever the address is
   - WEAVIATE_URL=http://localhost:10080  # or other weaviate db url
   - DATA_FOLDER_PATH=./artifacts/sagemaker_documentation  # or other path to the docs
   - USE_CLOUD_PIPELINE=0 # or 1 to use the local models

- "artifacts" folder with the data at root

# Running the project

The easiest way is to `docker compose up` and it will start up every service needed.

- API swagger at *http://localhost:3333/docs*.
- webapp available at *http://localhost:4444*.

NOTE: the notebooks folder has examples of the main objects of the project:
- you can make requests to the api in the [testing_api](./notebooks/testing_api.ipynb)
- the notebook [handler_testing](./notebooks/handler_testing.ipynb) shows how to use the RAG handler object, that contains all the logic

# Project Features:

- [x] RAG Pipeline to query documentation using LLamaIndex
   - [x] Option to use local models or cloud ones
   - [x] Prompt engineering (role, negative prompt)
   - [x] Embeddings saved in local weaviate service
   - [ ] Production monitoring
- [x] Prompting through FastAPI with streaming
   - [ ] Middleware for auth
   - [ ] Routes to update indexes
   - [ ] Add https
- [x] Demo webapp
   - [ ] Add https
- [x] Containerized deployment
   - [ ] Infrastructure as code (ideally with CI/CD)
