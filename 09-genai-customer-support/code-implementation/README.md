# GenAI Customer Support Assistant

A Retrieval-Augmented Generation (RAG) system that provides intelligent customer support by searching through internal documentation and knowledge bases to answer customer questions with accurate, contextual responses.

![Python](https://img.shields.io/badge/python-v3.11+-blue.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5--turbo-green.svg)
![LangChain](https://img.shields.io/badge/LangChain-🦜🔗-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20Apps-ffffff?logo=streamlit&style=flat&color=27e0c9&logoColor=FF4B4B)

## Project Overview

This customer support assistant demonstrates:
- **Retrieval-Augmented Generation (RAG)** implementation
- **Document processing** and vector storage
- **Semantic search** capabilities
- **Source attribution** for transparency
- **Interactive web interface** for easy testing

## Key Features

- 📄 **Multi-format document processing** (PDF, TXT)
- 🔍 **Intelligent semantic search** using vector embeddings
- 🤖 **Context-aware response generation** with GPT-3.5-turbo
- 📚 **Source attribution** showing which documents informed each answer
- 💬 **Interactive chat interface** with conversation history
- ⚡ **Real-time document upload** and processing
- 🎛️ **Configurable search parameters** and response settings

## Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-agents-portfolio.git
   cd ai-agents-portfolio/09-genai-customer-support
   ```

2. **Create and activate virtual environment**
   ```bash
   conda create -n ai-agents python=3.11
   conda activate ai-agents
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

5. **Run the application**
   ```bash
   streamlit run main.py
   ```


## How to Use

### Getting Started
1. **Load Sample Knowledge Base**: Click the button in the sidebar to load pre-configured FAQ data
2. **Or Upload Your Own Documents**: Use the file uploader to add PDF or TXT files
3. **Ask Questions**: Type questions in the chat interface
4. **Review Sources**: Expand source sections to see which documents were referenced

### Example Questions to Try
- "How do I reset my password?"
- "What payment methods do you accept?"
- "How do I contact technical support?"
- "What features are included in the platform?"



### Core Components

**Document Processing Pipeline**
- Text extraction from PDFs and text files
- Intelligent text chunking with overlap
- Metadata preservation for source attribution

**Vector Storage System**
- OpenAI embeddings for semantic representation
- ChromaDB for efficient similarity search
- Persistent storage for document retention

**RAG Chain**
- LangChain orchestration for retrieval and generation
- Context-aware prompt engineering
- Source document tracking

### Key Technologies

- **[LangChain](https://langchain.com/)**: Orchestration framework for LLM applications
- **[OpenAI API](https://openai.com/api/)**: GPT-3.5-turbo for response generation
- **[ChromaDB](https://www.trychroma.com/)**: Vector database for document storage
- **[Streamlit](https://streamlit.io/)**: Web interface framework

## Performance Metrics

- **Response Time**: < 2 seconds for most queries
- **Document Capacity**: Tested with 100+ documents
- **Search Accuracy**: Semantic similarity matching
- **Source Attribution**: 100% response traceability

## Configuration Options

### Environment Variables

```bash
# Required
OPENAI_API_KEY=your_api_key_here

# Optional Customization
OPENAI_MODEL=gpt-3.5-turbo          # or gpt-4
OPENAI_TEMPERATURE=0.1              # Response creativity (0-1)
CHROMA_PERSIST_DIRECTORY=./chroma_db # Vector storage location
```

### RAG Parameters

```python
# Text Splitting
chunk_size=1000          # Characters per chunk
chunk_overlap=200        # Overlap between chunks

# Retrieval
search_type="similarity" # Search algorithm
k=3                     # Number of sources to retrieve
```

## Testing

### Manual Testing
1. Load sample knowledge base
2. Test with provided example questions
3. Upload your own documents
4. Verify source attribution accuracy

### Automated Testing
```bash
# Run tests (when implemented)
pytest tests/
```

## Deployment Options

### Streamlit Cloud (Recommended for demos)
1. Push code to GitHub
2. Connect repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Add environment variables in dashboard
4. Deploy with one click

### Local Docker
```bash
# Build image
docker build -t customer-support-ai .

# Run container
docker run -p 8501:8501 customer-support-ai
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and commit: `git commit -m 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LangChain](https://langchain.com/) for the excellent RAG framework
- [OpenAI](https://openai.com/) for powerful language models
- [Streamlit](https://streamlit.io/) for rapid prototyping capabilities

---