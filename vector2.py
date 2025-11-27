from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os
import pandas as pd
import numpy as np # Used for checking null values

# --- Configuration ---
DATA_FILE = "data/csv/bike_safety.csv"
DB_LOCATION = "./faiss_llama_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Ensure the data directory exists before trying to read the CSV
if not os.path.exists(os.path.dirname(DATA_FILE)):
    # Assuming 'data/csv/' is your intended path
    print(f"Directory '{os.path.dirname(DATA_FILE)}' does not exist. Please check your file path.")
    exit()

# --- 1. Load Data ---
try:
    # FIX: Use 'sep="|"'. We assume the source CSV uses commas within the text 
    # but is intended to be a single column. Using a non-existent separator 
    # treats the whole line as one column, then we use the first column by name.
    # Alternatively, you can use quoting=3 (csv.QUOTE_NONE) or engine='python'.
    # For robust reading of a file with commas inside fields, let's use the 'python' engine and ensure quotes are ignored.
    df = pd.read_csv(DATA_FILE, engine='python', sep='^', header=None, skipinitialspace=True)
    # Since we are using a weird separator (^) to force a single column, we need to rename it
    df = df.iloc[1:] # Drop the header row if it exists, assuming row 0 is the "Safety_Tip" header
    df.columns = ['text_content'] # Give the single column a name
    
except FileNotFoundError:
    print(f"Error: CSV file not found at '{DATA_FILE}'. Please ensure it exists.")
    exit()

# --- 2. Initialize Components ---
embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
db_exists = os.path.exists(DB_LOCATION)

# --- 3. Prepare Documents (MUST be done before the conditional logic) ---
documents = []

for i, row in df.iterrows():
    # We now access the column by name, which contains the full line of text.
    text_content = row['text_content']
    
    # Check if content is valid (e.g., not NaN/null) before creating a document
    if pd.notna(text_content):
        document = Document(
            # Convert text content to string if needed
            page_content=str(text_content).strip(), 
            metadata={"source": "bicycle_safety_tips", "row_id": str(i)}
            # NOTE: LangChain Document objects do not typically take an 'id' argument directly
        )
        documents.append(document)
    
print(f"Prepared {len(documents)} documents for processing.")


# --- 4. Core Logic: Create or Update ---
if db_exists:
    # Fix 1: Correct logic for loading and updating
    print(f"Loading existing vectorstore from '{DB_LOCATION}'")
    
    # Use allow_dangerous_deserialization=True as per previous fix
    vectorstore = FAISS.load_local(DB_LOCATION, embedding, allow_dangerous_deserialization=True)
    
    print(f"Adding {len(documents)} new chunks to existing vectorstore")
    # This adds documents to the existing index
    vectorstore.add_documents(documents) 
    
else:
    # Fix 3: This branch is for first-time creation
    print(f"Creating new vectorstore from {len(documents)} chunks at '{DB_LOCATION}'")
    # The 'documents' list is correctly populated in Step 3
    vectorstore = FAISS.from_documents(documents, embedding)


# --- 5. Save/Persist ---
vectorstore.save_local(DB_LOCATION)
print(f"\nVectorstore saved/updated to '{DB_LOCATION}'.")
print(f"Total vectors in the store: {vectorstore.index.ntotal}")

# --- 6. Initialize Retriever ---
# Fix 4: Correct the typo in the keyword argument (search_kwards -> search_kwargs)
retriever = vectorstore.as_retriever(
      search_kwargs = {"k": 5}
)

print("Retriever successfully initialized.")