# Robust Privacy-Preserving RAG System for Personal Documents

This project implements a Retrieval-Augmented Generation (RAG) system designed to securely handle and query personal documents. It leverages advanced language models and vector databases to provide accurate and context-aware responses while ensuring data privacy.

We use local deployment of language models (like LLaMA 3) and vector databases (like FAISS) to avoid reliance on external APIs, ensuring that sensitive information remains on the user's device.

All the documents are stored and processed locally, and no data is sent to external servers, making this solution suitable for users with strict privacy requirements. The PDF and text files are ingested, and their embeddings are created using local models, the files will be stored in "data" folder from here the files will be read and processed.

## Environment Setup

1. **Install Python 3.13**  
    Ensure Python 3.13 is installed on your system. You can download it from the [official Python website](https://www.python.org/downloads/).

2. **Install `uv`**  
    Follow the instructions to install `uv` from its [official repository](https://github.com/uv-mpm/uv).

    Installed uv on mac using homebrew:
    ```bash
    brew install uv
    ```
    Navigate to your project directory, and run the following command to initialize `uv` with Python 3.13:
    ```bash
    then run:
    ```bash
    uv init --python=python3.13
    ```

3. **Create a Portable Environment**  
    Run the following command to create a new environment:
    ```bash
    uv create -n rag_env
    ```

4. **Activate the Environment**  
    Activate the newly created environment:
    ```bash
    uv venv
    source .venv/bin/activate
    ```

5. **Install Dependencies**  
    Use the `requirements.txt` file to install the necessary dependencies, including LangChain, chromadb, pypdf,  etc
    modules:
    ```bash
    uv add -r requirements.txt
    ```

    uv add ipykernel -- to use jupyter notebooks

6. **Verify Installation**  
    Ensure all dependencies are installed correctly:
    ```bash
    uv pip list
    ```

Your environment is now set up and ready for the RAG project.

To run the application, llama3 must be installed on your system. Follow the instructions in the llama3 repository to set it up.

Run ollama3 server:
```bash
ollama run llama3
```

To start the RAG application, execute:
```bash
streamlit run streamlit_app.py
```
This starts the Streamlit application, allowing you to interact with the RAG system through a web interface.

To update files (PDF, TXT), place them in the `data` folder and hit `Reload documents` button in the Streamlit app to refresh the document embeddings.