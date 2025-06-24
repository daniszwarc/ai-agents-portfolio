"""
Soccer Club Customer Support Assistant - Pinecone Version
Modified to use existing Pinecone vector database with soccer club information
"""

import os
import streamlit as st
from typing import List, Dict, Any
from dotenv import load_dotenv

# Import Pinecone correctly for v7.x
from pinecone import Pinecone

# LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document

# Try to import langchain-pinecone, fall back if not available
try:
    from langchain_pinecone import PineconeVectorStore
except ImportError:
    # Fallback for older versions
    from langchain_community.vectorstores import Pinecone as PineconeVectorStore

# Load environment variables
load_dotenv()

class SoccerClubSupportRAG:
    def __init__(self):
        """Initialize the RAG system with Pinecone vector database."""
        # Load environment variables
        load_dotenv()
        
        # API Keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")  
        self.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME") 
        self.pinecone_namespace = os.getenv("PINECONE_NAMESPACE")     
        
        # DEBUG: Print what we found (temporarily)
        print(f"OpenAI key exists: {bool(self.openai_api_key)}")
        print(f"Pinecone key exists: {bool(self.pinecone_api_key)}")
        print(f"Pinecone environment: {self.pinecone_environment}")
        print(f"Pinecone index name: {self.pinecone_index_name}")
        
        if not all([self.openai_api_key, self.pinecone_api_key, self.pinecone_environment, self.pinecone_index_name]):
            raise ValueError("Missing required environment variables. Check your .env file")
    
        
        # Initialize Pinecone
        pinecone.init(
            api_key=self.pinecone_api_key,
            environment=self.pinecone_environment
        )
        
        # Initialize embeddings and LLM
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            openai_api_key=self.openai_api_key
        )
        
        # Connect to your existing Pinecone index
        self.vector_store = None
        self.qa_chain = None
        
        self._connect_to_pinecone()
    
    def _connect_to_pinecone(self):
        """Connect to existing Pinecone vector database."""
        try:
            # Create Pinecone client
            pc = Pinecone(api_key=self.pinecone_api_key)
            
            # Connect to your existing index
            index = pc.Index(self.pinecone_index_name)
            
            # Create vector store
            self.vector_store = PineconeVectorStore(
                index=index,
                embedding=self.embeddings
            )
            
            # Create QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                ),
                return_source_documents=True
            )
            
            st.success("✅ Connected to Pinecone soccer club database!")
            
        except Exception as e:
            st.error(f"❌ Error connecting to Pinecone: {str(e)}")
            print(f"Detailed error: {e}")  # For debugging
            raise

    def get_response(self, question: str) -> Dict[str, Any]:
        """Get response for a soccer club related question."""
        if not self.qa_chain:
            return {
                "answer": "Please check the connection to the soccer club database.",
                "sources": []
            }
        
        try:
            # Create a soccer club context prompt
            soccer_prompt = f"""
            You are a helpful customer support assistant for a soccer club. 
            Answer the user's question based on the provided context about the club.
            Be friendly, knowledgeable, and passionate about soccer.
            
            Question: {question}
            """
            
            result = self.qa_chain({"query": soccer_prompt})
            
            # Extract source information
            sources = []
            for doc in result.get("source_documents", []):
                source_info = {
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata,
                    "relevance": "High"  # You could add scoring here
                }
                sources.append(source_info)
            
            return {
                "answer": result["result"],
                "sources": sources
            }
        except Exception as e:
            return {
                "answer": f"I apologize, but I encountered an error: {str(e)}",
                "sources": []
            }
    
    def search_soccer_info(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for specific soccer club information."""
        try:
            # Direct similarity search
            docs = self.vector_store.similarity_search(query, k=k)
            
            results = []
            for doc in docs:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "type": "search_result"
                })
            
            return results
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            return []

def main():
    """Main Streamlit application for Soccer Club Support."""
    st.set_page_config(
        page_title="⚽ Soccer Club Support Assistant",
        page_icon="⚽",
        layout="wide"
    )
    
    st.title("⚽ Soccer Club Customer Support Assistant")
    st.markdown("Ask me anything about our soccer club - players, matches, tickets, history, and more!")
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        try:
            with st.spinner("Connecting to soccer club database..."):
                st.session_state.rag_system = SoccerClubSupportRAG()
        except Exception as e:
            st.error(f"Failed to connect to database: {str(e)}")
            st.stop()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar with soccer club info
    with st.sidebar:
        st.header("⚽ Club Information")
        st.markdown("""
        **What you can ask about:**
        - 🏟️ Stadium information
        - 👥 Player profiles and stats
        - 📅 Match schedules and results
        - 🎫 Ticket information
        - 📈 Club history and achievements
        - 🛍️ Merchandise and shop
        - 🎯 Training and youth programs
        """)
        
        st.markdown("---")
        
        # Quick search feature
        st.subheader("🔍 Quick Search")
        search_query = st.text_input("Search club database:", placeholder="e.g., 'upcoming matches'")
        
        if st.button("Search") and search_query:
            with st.spinner("Searching..."):
                results = st.session_state.rag_system.search_soccer_info(search_query)
                if results:
                    st.write("**Search Results:**")
                    for i, result in enumerate(results[:3]):  # Show top 3
                        with st.expander(f"Result {i+1}"):
                            st.write(result["content"][:300] + "...")
    
    # Main chat interface
    st.header("💬 Ask About Our Club")
    
    # Example questions
    with st.expander("💡 Example Questions"):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🏟️ Stadium capacity?"):
                st.session_state.question_input = "What is the stadium capacity?"
            if st.button("👥 Star players?"):
                st.session_state.question_input = "Who are the star players?"
        with col2:
            if st.button("📅 Next match?"):
                st.session_state.question_input = "When is the next match?"
            if st.button("🎫 Ticket prices?"):
                st.session_state.question_input = "How much do tickets cost?"
    
    # Display chat history
    for i, (question, response) in enumerate(st.session_state.chat_history):
        with st.container():
            st.markdown(f"**🙋 You:** {question}")
            st.markdown(f"**⚽ Assistant:** {response['answer']}")
            
            if response['sources']:
                with st.expander(f"📖 Sources from club database ({len(response['sources'])})"):
                    for j, source in enumerate(response['sources']):
                        st.markdown(f"**Source {j+1}:**")
                        st.markdown(f"- **Content:** {source['content']}")
                        if source['metadata']:
                            st.markdown(f"- **Info:** {source['metadata']}")
            st.markdown("---")
    
    # Question input
    question = st.text_input(
        "Ask your question:",
        placeholder="e.g., When is the next home game?",
        key="question_input"
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        ask_button = st.button("Ask Question", type="primary")
    
    with col2:
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Process question
    if ask_button and question:
        with st.spinner("Searching club database..."):
            response = st.session_state.rag_system.get_response(question)
            st.session_state.chat_history.append((question, response))
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("⚽ **Powered by AI** | 🗄️ **Connected to Pinecone Database** | 🔍 **Real-time Club Information**")

if __name__ == "__main__":
    main()