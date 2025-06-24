"""
Soccer Club Customer Support Assistant - ChromaDB Version
Imports data from Pinecone using REST API, stores in local ChromaDB
"""

import os
import requests
import json
import streamlit as st
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SoccerClubSupportRAG:
    def __init__(self):
        """Initialize the RAG system with ChromaDB and Pinecone import capability."""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
        self.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
        self.pinecone_index_host = os.getenv("PINECONE_INDEX_HOST")

        if not all([self.openai_api_key, self.pinecone_api_key, self.pinecone_index_host]):
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file")
        
        # Initialize embeddings and LLM
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            openai_api_key=self.openai_api_key
        )
        
        # Initialize ChromaDB
        self.vector_store = None
        self.qa_chain = None
        self.chroma_persist_directory = "./soccer_club_chroma_db"
        
        # Try to load existing ChromaDB or offer to import from Pinecone
        self._initialize_chromadb()
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB, check if data exists, offer Pinecone import if empty."""
        # Skip loading existing ChromaDB due to permission issues
        st.warning("📥 Ready to import data from Pinecone.")
        self.vector_store = None
        self.qa_chain = None
    
    def _create_qa_chain(self):
        """Create the QA chain with the vector store."""
        print("DEBUG: _create_qa_chain called!")  # ADD THIS LINE
        if self.vector_store:
            print("DEBUG: Vector store exists, creating QA chain...")  # ADD THIS LINE
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 15}  # Get more context
                ),
                return_source_documents=True
            )
            print("DEBUG: QA chain created successfully!")  # ADD THIS LINE
        else:
            print("DEBUG: No vector store found!")

    def import_from_pinecone(self) -> Dict[str, Any]:
        """Import all data from Pinecone using REST API."""
        if not self.pinecone_api_key or not self.pinecone_index_name:
            return {"success": False, "error": "Pinecone credentials not found in .env file"}
        
        try:
            # Get the exact Pinecone host from environment
            pinecone_host = os.getenv("PINECONE_INDEX_HOST")
            if not pinecone_host:
                return {"success": False, "error": "PINECONE_INDEX_HOST not found in .env file"}
            
            # Clean the host URL (remove https:// if present)
            if pinecone_host.startswith("https://"):
                pinecone_host = pinecone_host.replace("https://", "")
            
            # Headers for Pinecone API
            headers = {
                "Api-Key": self.pinecone_api_key,
                "Content-Type": "application/json"
            }
            
            # Get index stats
            stats_url = f"https://{pinecone_host}/describe_index_stats"
            stats_response = requests.post(stats_url, headers=headers, timeout=10)
            
            if stats_response.status_code == 200:
                stats = stats_response.json()
                total_vectors = stats.get("totalVectorCount", 0)
                st.info(f"Found {total_vectors} vectors in Pinecone index")
            
            # Query to get all data using dummy vector
            query_url = f"https://{pinecone_host}/query"
            dummy_vector = [0.001] * 3072  # 3072 dimensions for text-embedding-3-large
            
            query_data = {
                "vector": dummy_vector,
                "topK": 1000,  # Get more documents (adjust based on your data size)
                "includeMetadata": True,
                "includeValues": False,
                "namespace": "test-web-scraper"
            }
            
            response = requests.post(query_url, headers=headers, json=query_data, timeout=30)
            
            if response.status_code == 200:
                results = response.json()
                matches = results.get("matches", [])
                
                # DEBUG: Let's see what we're getting
                st.write(f"🔍 **DEBUG: Found {len(matches)} matches from Pinecone**")
                
                # Convert Pinecone results to LangChain documents
                all_documents = []
                processed_count = 0
                empty_content_count = 0
                
                for match in matches:
                    metadata = match.get("metadata", {})
                    
                    # Extract text content 
                    text_content = metadata.get("text", "")
                    
                    # DEBUG: Show first 3 documents
                    
                    st.write(f"**Debug Document {processed_count + 1}:**")
                    st.write(f"- Metadata keys: {list(metadata.keys())}")
                    st.write(f"- Content exists: {bool(text_content)}")
                    st.write(f"- Content length: {len(text_content) if text_content else 0}")
                    if text_content:
                        st.write(f"- Content preview: {str(text_content)[:150]}...")

                    st.write(f"**Raw text (first 500 chars):** {repr(text_content[:500])}")    
                    # Specifically check for pricing
                    if "280" in text_content:
                        st.success(f"✅ Found '280' in raw text")
                    elif "28" in text_content:
                        st.warning(f"⚠️ Found '28' but not '280' in raw text")
                    
                    if "350" in text_content:
                        st.success(f"✅ Found '350' in raw text")
                    elif "35" in text_content:
                        st.warning(f"⚠️ Found '35' but not '350' in raw text")



                    if text_content and len(text_content.strip()) > 20:
                        import re
                        cleaned_text = text_content.strip()
                        
                        # FIX SPACING ISSUES FIRST (before other cleaning):
                        cleaned_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', cleaned_text)  # Add space between lowercase and uppercase
                        cleaned_text = re.sub(r'(\d)([A-Z][a-z])', r'\1 \2', cleaned_text)  # Add space between number and word
                        cleaned_text = re.sub(r'([a-z])(\d)', r'\1 \2', cleaned_text)  # Add space between letter and number
                        # SPECIFIC PRICING FIXES - add these to your text cleaning:
                        cleaned_text = re.sub(r'price:?\s*\$?(\d+)', r'price: CAD \1', cleaned_text)  # Fix price format
                        cleaned_text = re.sub(r'CAD\s*(\d+)([A-Z])', r'CAD \1 \2', cleaned_text)      # Space after prices
                        cleaned_text = re.sub(r'\$(\d+)([A-Z])', r'CAD \1 \2', cleaned_text)          # Convert $ to CAD with space
                        
                        # Clean the text content - FIXED VERSION
                        cleaned_text = re.sub(r'\*', '', cleaned_text)     # Remove asterisks
                        cleaned_text = re.sub(r'_', '', cleaned_text)      # Remove underscores
                        cleaned_text = re.sub(r'`', '', cleaned_text)      # Remove backticks
                        cleaned_text = re.sub(r'\$', 'CAD ', cleaned_text) # Replace $ with USD
                        cleaned_text = re.sub(r'[*_`~]', '', cleaned_text) # Remove other markdown chars

                        # ADD THESE NEW PATTERNS for the Day Camp document:
                        cleaned_text = re.sub(r'(\d+)(per)', r'\1 \2', cleaned_text)  # "28per" → "28 per"
                        cleaned_text = re.sub(r'(\d+)(days)', r'\1 \2', cleaned_text)  # "4days" → "4 days"
                        cleaned_text = re.sub(r'(\d+)(week)', r'\1 \2', cleaned_text)  # "28week" → "28 week"
                        cleaned_text = re.sub(r'(\))([A-Z])', r'\1 \2', cleaned_text)  # ") word" → ") word"

                        # DEBUG: Show what content we're actually storing
                        # ADD DEBUG HERE - RIGHT BEFORE CREATING DOCUMENT:
                        '''
                        if processed_count < 5:
                            st.write(f"**FIXED CONTENT {processed_count + 1}:**")
                            st.write(f"Before: `CAD 550Price` -> After: `{cleaned_text[cleaned_text.find('CAD 5'):cleaned_text.find('CAD 5')+20]}`")
                            
                            # Check specific price strings
                            if "CAD 280" in cleaned_text:
                                st.success("✅ CAD 280 in final content")
                            elif "CAD 28" in cleaned_text:
                                st.error("❌ Only CAD 28 in final content")
                                
                            # Check length
                            st.write(f"**Text length:** {len(cleaned_text)} characters")
                        '''

                        doc = Document(
                            page_content=cleaned_text,
                            metadata={
                                "source": "asmv_website",
                                "pinecone_id": match.get("id", ""),
                                "url": metadata.get("url", ""),
                                "imported_from": "pinecone"
                            }
                        )
                        all_documents.append(doc)
                    else:
                        empty_content_count += 1
                    
                    processed_count += 1
                
                # DEBUG: Show summary
                st.write(f"**📊 Processing Summary:**")
                st.write(f"- Total matches: {len(matches)}")
                st.write(f"- Documents with content: {len(all_documents)}")
                st.write(f"- Empty/short content: {empty_content_count}")
                
                if all_documents:
                    try:
                        # Create ChromaDB without persistence (in-memory)
                        self.vector_store = Chroma.from_documents(
                            documents=all_documents,
                            embedding=self.embeddings
                            # No persist_directory = in-memory only
                        )
                        
                        # Create QA chain
                        self._create_qa_chain()
                        
                        return {
                            "success": True,
                            "imported_count": len(all_documents),
                            "message": f"Successfully imported {len(all_documents)} documents from ASMV soccer club! (In-memory storage)"
                        }
                        
                    except Exception as e:
                        return {"success": False, "error": f"ChromaDB creation failed: {str(e)}"}
            
            else:
                return {"success": False, "error": f"Query failed: {response.status_code} - {response.text}"}
                
        except Exception as e:
            return {"success": False, "error": f"Import failed: {str(e)}"}

    def add_documents_to_vectorstore(self, documents: List[Document]):
        """Add new documents to ChromaDB."""
        if not documents:
            return
        
        if self.vector_store is None:
            # Create new ChromaDB
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.chroma_persist_directory
            )
        else:
            # Add to existing ChromaDB
            self.vector_store.add_documents(documents)
        
        # Update QA chain
        self._create_qa_chain()

    def get_response(self, question: str) -> Dict[str, Any]:
        """Get response for a soccer club related question."""
        if not self.qa_chain:
            return {
                "answer": "Please import data from Pinecone or load some documents first.",
                "sources": []
            }
        
        try:
            # Enhanced soccer club context prompt            

            if any(word in question.lower() for word in ["price", "pricing", "cost", "fee", "tarif", "prix", "coût"]):
                # SPECIALIZED PRICING PROMPT:
                soccer_prompt = f"""
                You are a knowledgeable customer support assistant for Soccer Verdun, a community soccer club. 
                
                The user is asking about pricing. Search through ALL the provided documents and list EVERY price, fee, and cost mentioned.
                
                Look for:
                - Registration fees for different age groups
                - Regular vs late pricing  
                - Special program costs
                - Any additional fees
                
                Organize the pricing by program/age group and include both regular and late prices where available.
                NEVER say "Based on the information provided" or similar.
                
                Question: {question}
                
                Provide a comprehensive pricing list based on ALL information in the context.
                """
            else:
                # YOUR EXISTING GENERAL PROMPT:
                soccer_prompt = f"""
                You are a knowledgeable customer support assistant for Soccer Verdun, a community soccer club. 
                
                Use the provided context to give detailed, specific answers about:
                - Registration processes and requirements
                - Program offerings for different age groups
                - Field locations and facilities
                - Schedules and timing
                - Fees and costs
                - Coaching staff and training programs
                - Club policies and procedures
                - Cancellation policies and timeframes
                
                Be specific with details like dates, times, locations, and contact information when available in the context.
                If the context doesn't contain enough information, say so and suggest they contact the club directly.

                ALWAYS respond in the language that the question was asked.
                NEVER say "Based on the information provided" or similar.
                
                Question: {question}
                
                Provide a comprehensive answer based on the club's information:
                """
            
            result = self.qa_chain({"query": soccer_prompt})

            '''
            # CAPTURE DEBUG INFO
            sources_debug = result.get("source_documents", [])
            debug_info = f"\n\n**🔍 DEBUG INFO:**\n"
            debug_info += f"- Found {len(sources_debug)} sources\n"
            
            
            if sources_debug:
                debug_info += f"- First source content: {sources_debug[0].page_content[:200]}...\n"
                if "280" in sources_debug[0].page_content:
                    debug_info += "- ✅ Found '280' in source!\n"
                elif "28" in sources_debug[0].page_content:
                    debug_info += "- ⚠️ Found only '28' in source!\n"
            '''

            # Extract source information
            sources = []
            for doc in result.get("source_documents", []):
                source_info = {
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata
                }
                sources.append(source_info)
            
            return {
                "answer": result["result"],  # ADD DEBUG TO ANSWER
                "sources": sources
            }
        except Exception as e:
            return {
                "answer": f"I apologize, but I encountered an error: {str(e)}",
                "sources": []
            }

    def search_soccer_info(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for specific soccer club information."""
        if not self.vector_store:
            return []
        
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            results = []
            for doc in docs:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            return results
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            return []

def main():
    """Main Streamlit application for Soccer Club Support."""
    st.set_page_config(
        page_title="⚽ Soccer Verdun Customer Support",
        page_icon="⚽",
        layout="wide"
    )
    
    st.title("⚽ Soccer Verdun Customer Support Assistant")
    st.markdown("Your friendly AI assistant for all things related to our club!")
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        try:
            with st.spinner("Initializing soccer club database..."):
                st.session_state.rag_system = SoccerClubSupportRAG()
        except Exception as e:
            st.error(f"Failed to initialize: {str(e)}")
            st.stop()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar for data management and club info
    with st.sidebar:
        st.header("⚽ Club Database Management")
        
        if st.button("📥 Import Data from Pinecone", type="primary"):
            with st.spinner("Importing soccer club data from Pinecone..."):
                result = st.session_state.rag_system.import_from_pinecone()
                
                if result["success"]:
                    st.success(result["message"])                    

                    if st.button("🔍 Test Stored Content"):
                        if st.session_state.rag_system.vector_store:
                            test_docs = st.session_state.rag_system.vector_store.similarity_search("280", k=3)
                            # st.write(f"**Search for '280' returned {len(test_docs)} documents:**")
                            for i, doc in enumerate(test_docs):
                                st.write(f"Doc {i+1}: {doc.page_content[:200]}...")
                    
                    # FORCE QA CHAIN CREATION:
                    try:
                        st.session_state.rag_system._create_qa_chain()
                        # st.write("✅ QA Chain created successfully!")
                        # st.write(f"QA chain exists: {st.session_state.rag_system.qa_chain is not None}")
                    except Exception as e:
                        st.error(f"Failed to create QA chain: {e}")
                    
                    # ENHANCED DEBUG:
                    # st.write(f"**🔍 Enhanced Debug Status:**")
                    # st.write(f"- Import result: {result}")
                    # st.write(f"- Vector store exists: {st.session_state.rag_system.vector_store is not None}")
                    # st.write(f"- QA chain exists: {st.session_state.rag_system.qa_chain is not None}")
                    
                    # Check vector store details
                    if st.session_state.rag_system.vector_store:
                        try:
                            collection = st.session_state.rag_system.vector_store._collection
                            count = collection.count()
                            # st.write(f"- Documents in ChromaDB: {count}")
                            
                            # Test a simple search
                            test_docs = st.session_state.rag_system.vector_store.similarity_search("soccer", k=1)
                            # st.write(f"- Test search returned: {len(test_docs)} documents")                            
                                
                        except Exception as e:
                            st.error(f"- Error checking vector store: {e}")
                    
                    # Check if _create_qa_chain was called
                    try:
                        st.session_state.rag_system._create_qa_chain()
                        st.write(f"- Manual QA chain creation: SUCCESS")
                        st.write(f"- QA chain after manual creation: {st.session_state.rag_system.qa_chain is not None}")
                    except Exception as e:
                        st.error(f"- Manual QA chain creation failed: {e}")
                        
                # else:
                    # st.error(f"Import failed: {result['error']}")
        
        if st.button("🗑️ Clear Local Database", type="secondary"):
            import shutil
            chroma_dir = "./soccer_club_chroma_db"
            if os.path.exists(chroma_dir):
                shutil.rmtree(chroma_dir)
                st.success("Local database cleared!")
                # Reset the RAG system
                st.session_state.rag_system.vector_store = None
                st.session_state.rag_system.qa_chain = None
                st.rerun()
            else:
                st.info("No local database to clear.")
              
                
        
        st.markdown("---")
        
        st.subheader("💡 What You Can Ask About")
        st.markdown("""
        - 🏟️ **Fields & Facilities:** Field conditions, locations, amenities
        - 👥 **Players & Teams:** Rosters, stats, team information  
        - 📅 **Schedules:** Game times, practice schedules, tournaments
        - 🎫 **Registration:** How to join, fees, requirements
        - 🏆 **Leagues & Competitions:** Standings, tournaments, results
        - 👨‍🏫 **Coaching:** Training programs, coaching staff
        - 🛍️ **Equipment:** Gear requirements, where to buy
        - 🚌 **Transportation:** Directions, parking, carpools
        - 📋 **Policies:** Rules, safety guidelines, weather policies
        """)
        
        st.markdown("---")
        
        # Quick search feature
        st.subheader("🔍 Quick Search")
        search_query = st.text_input("Search club database:", placeholder="e.g., 'field conditions'")
        
        if st.button("Search") and search_query:
            with st.spinner("Searching..."):
                results = st.session_state.rag_system.search_soccer_info(search_query)
                if results:
                    st.write("**Search Results:**")
                    for i, result in enumerate(results[:3]):
                        with st.expander(f"Result {i+1}"):
                            st.write(result["content"][:300] + "...")
    
    # Main chat interface
    st.header("💬 Ask About Our Soccer Club")
    
    # Example questions for community soccer
    with st.expander("💡 Example Questions"):
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🏟️ Field locations?"):
                st.session_state.question_input = "Where are the soccer fields located?"
            if st.button("👥 How to register?"):
                st.session_state.question_input = "How do I register my child for soccer?"
        with col2:
            if st.button("📅 Practice schedule?"):
                st.session_state.question_input = "What is the practice schedule?"
            if st.button("💰 What are the fees?"):
                st.session_state.question_input = "How much does it cost to join?"
        with col3:
            if st.button("🛍️ Equipment needed?"):
                st.session_state.question_input = "What equipment does my child need?"
            if st.button("☔ Weather policies?"):
                st.session_state.question_input = "What happens if it rains?"
    
    # Display chat history
    for i, (question, response) in enumerate(st.session_state.chat_history):
        with st.container():
            st.markdown(f"**🙋 You:** {question}")
            st.markdown(f"**⚽ Club Assistant:** {response['answer']}")
            
            if response['sources']:
                with st.expander(f"📖 Sources from club database ({len(response['sources'])})"):
                    for j, source in enumerate(response['sources']):
                        st.markdown(f"**Source {j+1}:**")
                        st.markdown(f"- **Content:** {source['content']}")
                        if source['metadata']:
                            st.markdown(f"- **Details:** {source['metadata']}")
            st.markdown("---")
    
    # Question input
    question = st.text_input(
        "Ask your question:",
        placeholder="e.g., How do I sign up my 8-year-old for soccer?",
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
    st.markdown("⚽ **Soccer Verdun AI Assistant** | 🗄️ **Powered by ChromaDB** | 📥 **Imported from Pinecone**")

if __name__ == "__main__":
    main()