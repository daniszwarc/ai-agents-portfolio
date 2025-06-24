"""
GenAI Customer Support Assistant - Main Implementation
A RAG-based customer support system using LangChain and ChromaDB
"""

import os
import streamlit as st
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class CustomerSupportRAG:
    def __init__(self):
        """Initialize the RAG system with necessary components."""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file")
        
        # Initialize embeddings and LLM
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            openai_api_key=self.openai_api_key
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True
        )
        
        # Initialize vector store
        self.vector_store = None
        self.qa_chain = None
        
        # Sample knowledge base - you can replace this with your own data
        self.sample_knowledge = [
            {
                "title": "Account Login Issues",
                "content": """
                If you're having trouble logging into your account:
                1. Check that you're using the correct email address
                2. Try resetting your password using the 'Forgot Password' link
                3. Clear your browser cache and cookies
                4. Try using an incognito/private browsing window
                5. If still having issues, contact our support team at support@company.com
                """
            },
            {
                "title": "Billing and Payments",
                "content": """
                For billing inquiries:
                - View your billing history in Account Settings > Billing
                - Update payment methods in Account Settings > Payment Methods
                - Billing occurs on the same date each month
                - We accept all major credit cards and PayPal
                - For billing disputes, please contact billing@company.com with your account details
                """
            },
            {
                "title": "Product Features and Usage",
                "content": """
                Our platform includes:
                - Dashboard for real-time analytics
                - API access for custom integrations
                - 24/7 customer support
                - Mobile app for iOS and Android
                - Advanced reporting and export capabilities
                - Multi-user collaboration tools
                For detailed feature documentation, visit our Help Center
                """
            },
            {
                "title": "Technical Support",
                "content": """
                For technical issues:
                1. Check our Status Page for known issues
                2. Try refreshing your browser or restarting the app
                3. Update to the latest version if using mobile app
                4. Check your internet connection
                5. Contact technical support at tech@company.com
                Include: browser version, operating system, and error messages
                """
            }
        ]
    
    def load_sample_knowledge(self):
        """Load sample knowledge base into the vector store."""
        documents = []
        for item in self.sample_knowledge:
            doc = Document(
                page_content=item["content"],
                metadata={"title": item["title"], "source": "knowledge_base"}
            )
            documents.append(doc)
        
        # Split documents
        split_docs = self.text_splitter.split_documents(documents)
        
        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
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
    
    def process_uploaded_file(self, uploaded_file) -> List[Document]:
        """Process uploaded file and return documents."""
        documents = []
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            if uploaded_file.name.endswith('.pdf'):
                loader = PyPDFLoader(tmp_file_path)
                documents = loader.load()
            elif uploaded_file.name.endswith('.txt'):
                loader = TextLoader(tmp_file_path)
                documents = loader.load()
            else:
                st.error("Unsupported file type. Please upload PDF or TXT files.")
                return []
            
            # Add filename to metadata
            for doc in documents:
                doc.metadata["filename"] = uploaded_file.name
                
        finally:
            os.unlink(tmp_file_path)
        
        return documents
    
    def add_documents_to_vectorstore(self, documents: List[Document]):
        """Add new documents to the existing vector store."""
        if not documents:
            return
        
        # Split documents
        split_docs = self.text_splitter.split_documents(documents)
        
        if self.vector_store is None:
            # Create new vector store
            self.vector_store = Chroma.from_documents(
                documents=split_docs,
                embedding=self.embeddings,
                persist_directory="./chroma_db"
            )
        else:
            # Add to existing vector store
            self.vector_store.add_documents(split_docs)
        
        # Update QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            ),
            return_source_documents=True
        )
    
    def get_response(self, question: str) -> Dict[str, Any]:
        """Get response for a customer question."""
        if not self.qa_chain:
            return {
                "answer": "Please load some documents first to provide assistance.",
                "sources": []
            }
        
        try:
            result = self.qa_chain({"query": question})
            
            # Extract source information
            sources = []
            for doc in result.get("source_documents", []):
                source_info = {
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata
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

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="GenAI Customer Support Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 GenAI Customer Support Assistant")
    st.markdown("Upload your knowledge base documents and start asking questions!")
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        try:
            st.session_state.rag_system = CustomerSupportRAG()
        except ValueError as e:
            st.error(str(e))
            st.stop()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📚 Knowledge Base Management")
        
        # Load sample knowledge base
        if st.button("Load Sample Knowledge Base"):
            with st.spinner("Loading sample knowledge..."):
                st.session_state.rag_system.load_sample_knowledge()
            st.success("Sample knowledge base loaded!")
        
        st.markdown("---")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            help="Upload PDF or TXT files to add to your knowledge base"
        )
        
        if uploaded_files:
            if st.button("Process Uploaded Files"):
                with st.spinner("Processing documents..."):
                    all_documents = []
                    for uploaded_file in uploaded_files:
                        documents = st.session_state.rag_system.process_uploaded_file(uploaded_file)
                        all_documents.extend(documents)
                    
                    if all_documents:
                        st.session_state.rag_system.add_documents_to_vectorstore(all_documents)
                        st.success(f"Successfully processed {len(uploaded_files)} file(s)!")
                    else:
                        st.error("No documents were processed successfully.")
    
    # Main chat interface
    st.header("💬 Ask a Question")
    
    # Display chat history
    for i, (question, response) in enumerate(st.session_state.chat_history):
        with st.container():
            st.markdown(f"**You:** {question}")
            st.markdown(f"**Assistant:** {response['answer']}")
            
            if response['sources']:
                with st.expander(f"📖 Sources ({len(response['sources'])})"):
                    for j, source in enumerate(response['sources']):
                        st.markdown(f"**Source {j+1}:**")
                        st.markdown(f"- **Content:** {source['content']}")
                        st.markdown(f"- **Metadata:** {source['metadata']}")
            st.markdown("---")
    
    # Question input
    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., How do I reset my password?",
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
        with st.spinner("Thinking..."):
            response = st.session_state.rag_system.get_response(question)
            st.session_state.chat_history.append((question, response))
        st.rerun()
    
    # Instructions
    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        1. **Load Knowledge Base:** Click "Load Sample Knowledge Base" or upload your own documents
        2. **Ask Questions:** Type your question in the input field and click "Ask Question"
        3. **Review Sources:** Expand the sources section to see which documents were used
        4. **Upload More Documents:** Use the sidebar to add more files to your knowledge base
        
        **Supported File Types:** PDF, TXT
        **Example Questions:**
        - "How do I reset my password?"
        - "What payment methods do you accept?"
        - "How do I contact technical support?"
        """)

if __name__ == "__main__":
    main()