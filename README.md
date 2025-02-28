# Medical Assistant for Dermatoscopic Images

This is a **Medical Assistant for Dermatoscopic Images** application designed to assist in diagnosing skin lesions. 
It uses **Vector Similarity Search** (powered by [**Qdrant**](https://qdrant.tech/)) for image comparison and provides an AI-formatted diagnosis based on the uploaded image.
The demo utilizes the **HAM_10000** dataset, which can be found on [HuggingFace](https://huggingface.co/datasets/marmal88/skin_cancer).

> **Disclaimer**: This application is a demo tool and is not a substitute for professional medical advice or diagnosis.

## Demo Workflow:
![Demo GIF](https://storage.googleapis.com/demo-skin-cancer/git-gif-skin-demo.gif)

## Features:
1. **Vector Similarity Search**: Get insights and compare skin lesion images based on cosine similarity with a database of known cases.
2. **AI Assistant for Diagnosis**: An AI assistant provides a potential diagnosis for the uploaded image based on KNN classification from the **HAM_10000** dataset.
3. **Local Hosting**: The app can be hosted fully locally, ensuring medical data protection. 
   - Can be hosted on [Qdrant's Private Cloud](https://qdrant.tech/documentation/private-cloud/)
   - Uses a local embedding model ([**Dino v2** from Hugging Face Transformers](https://huggingface.co/facebook/dinov2-large)) and a local RAG model ([**ollama's deepseek-llm**](https://ollama.com/library/deepseek-llm)).

> **Note**: Qdrant is model-agnostic, meaning you can use any state-of-the-art medical imagery model (e.g., **medVit**, **skinVit**, **radDino**, **UNI**, etc.). For this demo, **DinoV2** is used for simplicity.

### Functionality:
1. **Top-5 Cosine Similarity Vector Search**: Compares the uploaded image against a dataset using cosine similarity with CLS image embeddings.
2. **RAG AI Assistant Diagnosis**: Provides a diagnosis based on KNN classification (k=10) run on Qdrant with the **HAM_10000** dataset.
   - **KNN-Classifier Process**:
     1. **First-stage retriever**: Uses mean pooled image patch embeddings.
     2. **Reranker**: Uses multi-vector image patch embeddings.
   - **Accuracy**: The classifier achieved **0.805 precision** on k=10 using the test dataset (no fine-tuning).
  
### Further Customizations:
The app can be extended with additional features:

- Filterable Semantic Search: Utilize Qdrant’s custom filterable vector index for better search customization. [Learn More](https://qdrant.tech/articles/filtrable-hnsw/)
- Anomaly & Outliers Detection: Implement anomaly detection for medical image analysis. [Learn More](https://www.youtube.com/live/_BQTnXpuH-E)
- Pattern Discovery: Explore pattern discovery within your data. [Learn More](https://qdrant.tech/articles/discovery-search/)

## Files in this Repository:
- **indexing.py**: Used for indexing the **HAM_10000** dataset into Qdrant.
- **evaluating.py**: Evaluates the KNN classification performance using different retrieval strategies.
- **app.py**: The main Streamlit app.
- **config.ini**: Contains Qdrant cloud credentials (for use with Qdrant Cloud instead of local hosting).
- **requirements.txt**: Lists all dependencies for the app.

