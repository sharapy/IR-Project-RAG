"""Streamlit UI for RAG System """

import streamlit as st
from pathlib import Path
import sys
import time
from langchain_ollama.llms import OllamaLLM
import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# load env
load_dotenv()

# Add src to path
sys.path.append(str(Path(__file__).parent))

# from src.config.config import Config
from src.doc_ingestion.doc_processor import DocProcessor
from src.graph_builder.graph_builder import GraphBuilder

# Page configuration
st.set_page_config(
    page_title="🤖📚 Local RAG Search",
    page_icon="🔍",
    layout="centered"
)

# Simple CSS
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'history' not in st.session_state:
        st.session_state.history = []

@st.cache_resource
def initialize_rag():
    """Initialize the RAG system (cached)"""
    try:
        # Initialize components
        llm = OllamaLLM(model=os.getenv("LLM_MODEL", "llama3"))
        # llm = ChatOllama(model=os.getenv("LLM_MODEL", "llama3"))
        processor = DocProcessor()        
        # Process documents
        processed_chunks = processor.process_docs(os.getenv("DATA_DIR")+"/pdf/")
        # Create vector store
        # processor.create_update_vectorstore(processed_chunks)
        vectorstore = processor.create_update_vectorstore(processed_chunks)
        retriever=processor.get_retriever()
        
        # Build graph
    #     graph_builder = GraphBuilder(
    #         retriever=processor.get_retriever(),
    #         llm=llm
    #     )
    #     graph_builder.build()
    #     return graph_builder, len(processed_chunks)
    # except Exception as e:
    #     st.error(f"Failed to initialize: {str(e)}")
    #     return None, 0
            # Build simple rag chain as a runnable pipeline
        simple_prompt = ChatPromptTemplate.from_template("""Answer the question based only on the following context:
Context: {context}

Question: {question}

Answer:""")
        simple_rag_chain = (
            RunnablePassthrough()  # Input: {"question": "query"}
            | {"context": lambda x: retriever | format_documents | (lambda docs: format_documents(docs))(x["question"]),
                "question": lambda x: x["question"]}
            | simple_prompt
            | llm
            | StrOutputParser()
        )
        
        return simple_rag_chain, len(processed_chunks), retriever
    
    except Exception as e:
        st.error(f"Failed to initialize: {str(e)}")
        return None, 0

def format_documents(docs):
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source_file", "Unknown")
        formatted.append(f"Document {i+1} (Source: {source}):\n{doc.page_content}")
    return "\n\n".join(formatted)

def main():
    """Main application"""
    init_session_state()
    
    # Title
    st.title("🤖📚 RAG Document Search")
    st.markdown("Ask questions about the loaded documents")
    
    # Initialize system
    if not st.session_state.initialized:
        with st.spinner("Loading system..."):
            rag_system, num_chunks, retriever = initialize_rag()
            if rag_system:
                st.session_state.rag_system = rag_system
                st.session_state.retriever = retriever
                st.session_state.initialized = True
                st.success(f"✅ System ready! ({num_chunks} document chunks loaded)")
    
    st.markdown("---")
    
    # Search interface
    with st.form("search_form"):
        question = st.text_input(
            "Enter your question:",
            placeholder="What would you like to know?"
        )
        submit = st.form_submit_button("🔍 Search")
    
    # Process search
    if submit and question:
        if st.session_state.rag_system and st.session_state.retriever:
            with st.spinner("Searching..."):
                start_time = time.time()
                
                # Get answer
                # result = st.session_state.rag_system.run(question)
                answer = st.session_state.rag_system.invoke({"question": question})
                # retriever = st.session_state.retriever
                # retrieved_docs = retriever.invoke(question)
                retrieved_docs = st.session_state.retriever.invoke(question)
                
                elapsed_time = time.time() - start_time
                
                # Add to history
                st.session_state.history.append({
                    'question': question,
                    'answer': answer,
                    'docs': retrieved_docs,
                    'time': elapsed_time
                })
                
                # Display answer
                st.markdown("### 💡 Answer")
                # st.success(result['answer'])
                st.success(answer)
                
                # # Show retrieved docs in expander
                # with st.expander("📄 Source Documents"):
                #     for i, doc in enumerate(result['retrieved_docs'], 1):
                #         st.text_area(
                #             f"Document {i}",
                #             doc.page_content[:300] + "...",
                #             height=100,
                #             disabled=True
                #         )
                
                st.markdown("### 📚 Sources")
                if retrieved_docs:
                    cols = st.columns(min(3, len(retrieved_docs)))  # Max 3 columns
                    for i, (col, doc) in enumerate(zip(cols, retrieved_docs)):
                        with col:
                            st.markdown(f"**Doc {i+1}:** {doc.metadata.get('source_file', 'Unknown')}")
                            st.caption(doc.page_content[:150] + "...")
                else:
                    st.warning("No documents retrieved")
                
                st.caption(f"⏱️ Response time: {elapsed_time:.2f} seconds")
        else:
            st.error("RAG system not initialized. Please refresh.")
    
    # Show history
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 📜 Recent Searches")
        
        for item in reversed(st.session_state.history[-3:]):  # Show last 3
            with st.container():
                st.markdown(f"**Q:** {item['question']}")
                st.markdown(f"**A:** {item['answer'][:200]}...")
                st.caption(f"Time: {item['time']:.2f}s")
                st.markdown("")

if __name__ == "__main__":
    main()