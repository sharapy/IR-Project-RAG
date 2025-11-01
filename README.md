# Setting Up the Environment for RAG Project

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