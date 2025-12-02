"""
Document processing module
Contains semantic chunking 
    We use Langchain semantic chunker for this purpose for simplicity
    instead of using cosine similarity to find the similarity between chunks
HuggingFace Embedding using given FAISS vectorstore
The embeddings will need internet connection on first go, as with other setup. 
"""
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFDirectoryLoader,
    PyMuPDFLoader
)
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Union
from pathlib import Path
import os, shutil
from dotenv import load_dotenv

# load env
load_dotenv()

class DocProcessor:
    """Loads and processes documents from data directory with semantic chunking"""

    def __init__(self):
        """Initialize with embedding model for semantic chunking"""
        self.data_dir = os.getenv("DATA_DIR", "data")
        # self.embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")
        # self.embedding = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self.embedding = OllamaEmbeddings(model=self.embedding_model)
        # self.chunker = SemanticChunker(self.embedding) # disabled as semantic chunker is slow when corpora increases
        self.chunker = RecursiveCharacterTextSplitter(
                            chunk_size=1500,
                            chunk_overlap=300,
                            separators=["\n\n", "\n", ".", " "],
                        )
        self.vectorstore = None
        self.retriever = None

    def load_documents_from_data_dir(self, data_dir: str = "data") -> List[Document]:
        """
        Load all documents (PDF, TXT) from data directory recursively.
        Args: data_dir: Directory path to scan for documents    
        Returns: List of Document objects
        """
        all_documents = []
        data_path = Path(data_dir)
        
        # Supported file extensions and their loaders
        file_extensions = {
            '.pdf': PyMuPDFLoader,
            '.txt': TextLoader
        }
        # Get all supported files
        supported_files = []
        for ext, loader in file_extensions.items():
            files = list(data_path.glob(f"**/*{ext}"))
            supported_files.extend(files)
        print(f"Found {len(supported_files)} documents in {data_dir}")
        
        # Load documents based on file type
        for file_path in supported_files:
            ext = file_path.suffix.lower()
            loader_class = file_extensions.get(ext)
            
            if loader_class:
                print(f"Loading {file_path.name} ({ext})...")
                try:
                    if ext == '.pdf':
                        loader = loader_class(str(file_path))
                    elif ext == '.txt':
                        loader = loader_class(str(file_path), encoding='utf-8')
                    # load it 
                    docs = loader.load()
                    # Add source file metadata to all documents from this file (see if this neede on UI)
                    for doc in docs:
                        doc.metadata['source_file'] = file_path.name
                        doc.metadata['source_path'] = str(file_path)
                        doc.metadata['file_type'] = ext
                    
                    all_documents.extend(docs)
                    print(f"  -> Loaded {len(docs)} pages/documents")
                    
                except Exception as e:
                    print(f"  -> Error loading {file_path.name}: {e}")
            else:
                print(f"Unsupported file type: {file_path.name}")
        print(f"\nTotal documents loaded: {len(all_documents)}")
        return all_documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into semantically meaningful chunks.
        Args: documents: List of Document objects to split   
        Returns: List of semantic Document chunks
        """
        chunks = self.chunker.split_documents(documents)
        print(f"Created {len(chunks)} semantic chunks")
        return chunks

    def process_docs(self, data_dir: str = "data") -> List[Document]:
        """
        Complete pipeline: Load documents → Semantic chunking → Return processed chunks
        Args: data_dir: Directory containing documents to process
        Returns: List of semantically chunked Document objects ready for embedding
        """
        print("=== Starting document processing pipeline ===")
        documents = self.load_documents_from_data_dir(data_dir)
        chunks = self.split_documents(documents)
        print("=== Document processing complete ===")
        return chunks
    
    def create_update_vectorstore_faiss(self, chunks: List[Document], save_dir: str = "../../faiss_vectorstore") -> FAISS:
        """
        Create or update a FAISS vector store with new chunks.
        If a vectorstore exists in save_dir, load it and add new chunks,
        otherwise create a new vectorstore from chunks.
        Args:
            chunks: List of Document chunks to add.
            save_dir: Directory path to save/load vectorstore.
        Returns:
            FAISS vectorstore instance.
        """
        # save_dir = os.getenv("VECTORSTORE_DIR")
        print(save_dir)
        if os.path.exists(save_dir) and os.path.isdir(save_dir):
            # print(f"Loading existing vectorstore from '{save_dir}'")
            # self.vectorstore = FAISS.load_local(save_dir, self.embedding, allow_dangerous_deserialization=True)
            # print(f"Adding {len(chunks)} new chunks to existing vectorstore")
            # self.vectorstore.add_documents(chunks)

            # Since deduplication is not present at the moment, will recreate store 
            shutil.rmtree(save_dir)
            self.vectorstore = FAISS.from_documents(chunks, self.embedding)
        
        print("Creating new vectorstore from chunks")
        self.vectorstore = FAISS.from_documents(chunks, self.embedding)
            
        # self.retriever = self.vectorstore.as_retriever()
        self.vectorstore.save_local(save_dir)
        print(f"Vectorstore saved to '{save_dir}' with total {self.vectorstore.index.ntotal} vectors")
        return self.vectorstore

    def create_update_vectorstore(
        self,
        chunks: List[Document],
        save_dir: str = "chroma_vectorstore",
        collection_name: str = "rag_collection",
    ) -> Chroma:
        """
        Create or update a Chroma vector store with new chunks.
        If a vectorstore exists in save_dir, delete and recreate it (no dedup).
        Otherwise create a new vectorstore from chunks.
        """
        persist_dir = os.path.abspath(save_dir)
        print(persist_dir)

        if os.path.exists(persist_dir) and os.path.isdir(persist_dir):
            print("Loading existing Chroma store")
            self.vectorstore = Chroma(
                embedding_function=self.embedding,
                collection_name=collection_name,
                persist_directory=persist_dir,
            )
            print(f"Adding {len(chunks)} new chunks")
            self.vectorstore.add_documents(chunks)
        else:
            print("Creating new Chroma vectorstore from chunks")
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding,
                collection_name=collection_name,
                persist_directory=persist_dir,
            )
        print(f"Chroma vectorstore saved to '{persist_dir}'")
        return self.vectorstore
    
    def _clean_text(self, text: str) -> str:
        # if process_pdf from ipynb is used for metadata
        text = " ".join(text.split())
        text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
        return text

    def get_retriever(self, k: int = 4):
        """
        Get the retriever instance
        Returns: Retriever instance
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
        
    
    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve relevant documents for a query
        Args: query: Search query
            k: Number of documents to retrieve    
        Returns: List of relevant documents
        """
        retriever = self.get_retriever(k=k)
        return retriever.invoke(query)
    
# Module Sample run:
# if __name__ == "__main__":
#     processor = DocProcessor()
#     processed_chunks = processor.process_docs("../../data/pdf/")
#     print(f"\nFinal result: {len(processed_chunks)} semantic chunks ready for vectorstore!")
#     #creating vector store or loading from it
#     processor.create_update_vectorstore(processed_chunks)
    
#     docs = processor.retrieve("Bike Security?", k=3)
#     # for d in docs:
#     #     print(d.metadata.get("source_file"), "->", d.page_content[:200])
#     formatted = []
#     for i, doc in enumerate(docs):
#         source = doc.metadata.get("source_file", "Unknown")
#         formatted.append(f"Document {i+1} (Source: {source}):\n{doc.page_content}")
#     print("\n\n".join(formatted))