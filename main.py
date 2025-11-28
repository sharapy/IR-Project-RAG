"""Main application entry point for RAG system"""

import sys
from pathlib import Path
from langchain_ollama.llms import OllamaLLM
import os
from dotenv import load_dotenv

# load env
load_dotenv()
# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.doc_ingestion.doc_processor import DocProcessor
from src.graph_builder.graph_builder import GraphBuilder

class RAGAPP:
    """Main RAG application"""
    
    def __init__(self):
        """
        Initialize RAG system
        Args: urls: List of URLs to process (uses defaults if None)
        """
        print("🚀 Initializing Agentic RAG System...")
        
        # Initialize components
        self.llm = OllamaLLM(model=os.getenv("LLM_MODEL", "llama3"))
        self.doc_processor = DocProcessor()   
        # Process documents and create vector store
        self._setup_vectorstore()
        
        # Build graph
        self.graph_builder = GraphBuilder(
            retriever=self.doc_processor.get_retriever(),
            llm=self.llm
        )
        self.graph_builder.build()
        
        print("✅ System initialized successfully!\n")
    
    def _setup_vectorstore(self):
        """Setup vector store with processed documents"""
        processed_chunks = self.doc_processor.process_docs(os.getenv("DATA_DIR")+"/pdf/")
        print(f"📊 Created {len(processed_chunks)} document chunks")
        
        print("🔍 Creating vector store...")
        self.doc_processor.create_update_vectorstore(processed_chunks)
    
    def ask(self, question: str) -> str:
        """
        Ask a question to the RAG system
        Args: question: User question
        Returns: Generated answer
        """
        print(f"❓ Question: {question}\n")
        print("🤔 Processing...")
        
        result = self.graph_builder.run(question)
        answer = result['answer']
        
        print(f"✅ Answer: {answer}\n")
        return answer
    
    def interactive_mode(self):
        """Run in interactive mode"""
        print("💬 Interactive Mode - Type 'quit' to exit\n")
        while True:
            question = input("Enter your question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            if question:
                self.ask(question)
                print("-" * 80 + "\n")

def main():
    """Main function"""    
    # Initialize RAG system
    rag = RAGAPP()
    
    # Example questions
    example_questions = [
        "What is the concept Bike Security?",
        "What are the key components of RAG?",
        "Explain the concept of QR Codes"
    ]
    
    print("=" * 80)
    print("📝 Running example questions:")
    print("=" * 80 + "\n")
    
    for question in example_questions:
        rag.ask(question)
        print("=" * 80 + "\n")
    
    # Optional: Run interactive mode
    print("\n" + "=" * 80)
    user_input = input("Would you like to enter interactive mode? (y/n): ")
    if user_input.lower() == 'y':
        rag.interactive_mode()

if __name__ == "__main__":
    main()