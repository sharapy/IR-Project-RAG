from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("data/csv/bike_safety.csv")

embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db_location = "./faiss_llama_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        document = Document(
            page_content=row,
            metadata={"source": "bicycle_safety_tips"},
            id = str(i)
        )
        ids.append(str(i))
        documents.append(document)

if add_documents:
            print(f"Loading existing vectorstore from '{db_location}'")
            vectorstore = FAISS.load_local(db_location, embedding, allow_dangerous_deserialization=True)
            print(f"Adding {len(documents)} new chunks to existing vectorstore")
            vectorstore.add_documents(documents)
else:
    print("Creating new vectorstore from chunks")
    vectorstore = FAISS.from_documents(documents, embedding)

vectorstore.save_local(db_location)
print(f"Vectorstore saved to '{db_location}' with total {vectorstore.index.ntotal} vectors")

#vector_store = FAISS.from_documents(documents, embedding)
#vector_store.save_local(db_location)

retriever = vectorstore.as_retriever(
      search_kwards = {"k":5}
)