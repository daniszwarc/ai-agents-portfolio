try:
    from pinecone import Pinecone
    print("✅ Pinecone import successful!")
    
    # Test client creation (replace with your real API key)
    pc = Pinecone(api_key="test")
    print("✅ Pinecone client creation successful!")
    
except Exception as e:
    print(f"❌ Error: {e}")