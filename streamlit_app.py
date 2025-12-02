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
from src.doc_ingestion.doc_processor import DocProcessor
# Graph can be added once local llm model supports it. 
#from src.graph_builder.graph_builder import GraphBuilder

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
        processed_chunks = processor.process_docs(os.getenv("DATA_DIR"))
        # Create vector store
        processor.create_update_vectorstore(processed_chunks)
        # vectorstore = processor.create_update_vectorstore(processed_chunks)
        retriever=processor.get_retriever(k=4)
        
        simple_prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant. 
            Use the conversation history only if needed and the context to answer the question.
            If you do not find any document match provide a generalized answer and say you dont know the exact answer:

Conversation history:{history}

Context: {context}

Question: {question}

Answer:""")
        simple_rag_chain = (
            RunnablePassthrough()  # Input: {"question": "query"}
            # Build context by calling the retriever and formatting the returned docs into a string
            | {"context": lambda x: format_documents(retriever.invoke(x["question"])),
               "question": lambda x: x["question"],
               "history": lambda x: x.get("history", "")
            }
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
        # If the retriever returned plain strings, doc will be a str
        if isinstance(doc, str):
            source = "Unknown"
            content = doc
        else:
            # Fallback safely if attributes are missing
            metadata = getattr(doc, "metadata", {}) or {}
            source = metadata.get("source_file", "Unknown")
            content = getattr(doc, "page_content", str(doc))

        formatted.append(f"Document {i+1} (Source: {source}):\n{content}")
    return "\n\n".join(formatted)

def expand_query_with_llm(user_query: str, llm: OllamaLLM) -> str:
    prompt = f"""Rewrite the following user question to be a clear, specific search query for a document retrieval system. 
Keep the same intent but make it more explicit and detailed if needed.

User question:
\"\"\"{user_query}\"\"\"

Rewritten search query:"""
    expanded = llm.invoke(prompt)
    print(expanded)
    return str(expanded).strip()

def main():
    """Main application"""
    init_session_state()
    
    # Title
    st.title("🤖📚 Personal-RAG Document Search")
    st.markdown("Ask questions about the loaded documents")

    # Reload button: clears cached resources and forces re-initialization
    if st.button("🔁 Reload documents"):
        # Clear cached resources created by @st.cache_resource
        try:
            st.cache_resource.clear()
        except Exception:
            # Older Streamlit versions may not have cache_resource; try clearing all cache
            try:
                st.legacy_caching.clear_cache()
            except Exception:
                pass

        # Reset session state and rerun the app so initialize_rag() runs again
        st.session_state.initialized = False
        st.session_state.rag_system = None
        st.session_state.retriever = None
        # st.session_state.history = []
        # experimental_rerun may not exist on all Streamlit versions; fall back to st.stop()
        try:
            st.rerun()
        except AttributeError:
            # st.stop() will halt this run; Streamlit will rerun on the next interaction
            st.stop()
    
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
                # answer = st.session_state.rag_system.invoke({"question": question})
                history_entries = []
                for h in st.session_state.history[-5:]:  # last 5 exchanges
                    history_entries.append(f"User: {h['question']}\nAssistant: {h['answer']}")
                history_str = "\n\n".join(history_entries)

                # Call RAG chain with question + history
                # answer = st.session_state.rag_system.invoke(
                #     {
                #         "question": question,
                #         "history": history_str,
                #     }
                # )

                def stream_answer(question: str, history_str: str):
                    for chunk in st.session_state.rag_system.stream(
                        {"question": question, "history": history_str}
                    ):
                        yield chunk

                st.markdown("### 💡 Answer")
                answer = st.write_stream(stream_answer(question, history_str))


                # retriever = st.session_state.retriever
                # retrieved_docs = retriever.invoke(question)
                llm = OllamaLLM(model=os.getenv("LLM_MODEL", "llama3"))
                # expanded_query = expand_query_with_llm(question, llm)
                # retrieved_docs = st.session_state.retriever.invoke(expanded_query)

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
                # st.markdown("### 💡 Answer")
                # st.success(answer)
                
                
                st.markdown("### 📚 Sources")
                if retrieved_docs:
                    cols = st.columns(min(3, len(retrieved_docs)))  # Max 3 columns
                    for i, (col, doc) in enumerate(zip(cols, retrieved_docs)):
                        with col:
                            # Handle both Document objects and plain strings
                            if isinstance(doc, str):
                                source = "Unknown"
                                snippet = doc[:150] + "..."
                            else:
                                source = doc.metadata.get("source_file", "Unknown")
                                snippet = doc.page_content[:150] + "..."

                            st.markdown(f"**Doc {i+1}:** {source}")
                            st.caption(snippet)
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