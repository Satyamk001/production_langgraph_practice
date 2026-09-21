"""
Advanced RAG Patterns
Multi-query, self-query, compression, hybrid search
"""


import tempfile
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
import logging

load_dotenv()
TECH_DOCS = [
       Document(
        page_content="Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming. Python is widely used in web development, data science, artificial intelligence, and automation.",
        metadata={
            "topic": "programming",
            "language": "python",
            "difficulty": "beginner",
        },
    ),
    Document(
        page_content="JavaScript is the language of the web. It runs in browsers and on servers with Node.js. Modern frameworks like React, Vue, and Angular make building interactive web applications efficient. JavaScript supports asynchronous programming with Promises and async/await.",
        metadata={
            "topic": "programming",
            "language": "javascript",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="Machine learning is a subset of AI that enables systems to learn from data. Supervised learning uses labeled data, while unsupervised learning finds patterns in unlabeled data. Popular ML frameworks include TensorFlow, PyTorch, and scikit-learn.",
        metadata={
            "topic": "ai",
            "subtopic": "machine_learning",
            "difficulty": "advanced",
        },
    ),
    Document(
        page_content="LangChain is a framework for building LLM applications. It provides tools for prompts, chains, agents, and memory. LangChain supports multiple LLM providers including OpenAI, Anthropic, and local models.",
        metadata={
            "topic": "ai",
            "subtopic": "llm_frameworks",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs. Key features include state management, cycles and loops, human-in-the-loop workflows, and persistence. LangGraph extends LangChain for complex agent architectures.",
        metadata={
            "topic": "ai",
            "subtopic": "llm_frameworks",
            "difficulty": "advanced",
        },
    ),
    Document(
        page_content="Docker is a platform for containerizing applications. Containers package code and dependencies together for consistent deployment. Docker Compose orchestrates multi-container applications. Kubernetes scales Docker containers in production.",
        metadata={
            "topic": "devops",
            "subtopic": "containers",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="PostgreSQL is an advanced open-source relational database. It supports JSON data types, full-text search, and extensions like pgvector for vector similarity search. PostgreSQL is ACID compliant and highly extensible.",
        metadata={
            "topic": "database",
            "type": "relational",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="Vector databases like Pinecone, Chroma, and Qdrant are optimized for storing and searching embeddings. They enable semantic similarity search for RAG applications. Most support metadata filtering and hybrid search combining keywords with vectors.",
        metadata={"topic": "database", "type": "vector", "difficulty": "intermediate"},
    ),
]

llm = init_chat_model(model="gemini-3.6-flash",model_provider="google_genai", temperature=0)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")

# Create a base vector store from knowledge base.
vector_store = Chroma.from_documents(
        documents=TECH_DOCS, embedding=embedding_model
    )

def multi_query_rag():
        
    # Enable logging to see multi-query generation
    logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

    # Create multi-query retriever
    retriever = MultiQueryRetriever.from_llm( # number of queries to generate can be adjusted with prompt
        retriever=vector_store.as_retriever(search_kawargs={"k": 2}), llm=llm # can also add prompt= ChatPromptTemplate.from_template("Generate 3 queries to retrieve relevant documents for the question: {question}")
    )

    query = "What tools can I use to build AI applications?"

    # Retrieve documents
    docs = retriever.invoke(query)
    print(f"Retrieved {len(docs)} unique documents:")

    for i, doc in enumerate(docs):
        print(
            f"\n{i+1}. [{doc.metadata.get('topic', 'N/A')}] {doc.page_content[:100]}..."
        )



from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

def contextual_compression():
    
    compressor = LLMChainExtractor.from_llm(llm)
    
    compressor_retriever = ContextualCompressionRetriever(
        compressor=compressor,
        base_retriever=vector_store.as_retriever(search_kwargs={"k": 2})
    )
    
    query = "What frameworks exist for building LLM applications?"
    # base_docs = vector_store.as_retriever(search_kwargs={"k": 2}).invoke(query)
    compressed_docs = compressor_retriever.invoke(query)
    
    # print(f"Retrieved {len(base_docs)} base documents:")
    print(f"Retrieved {len(compressed_docs)} compressed documents:")
    

if __name__ == "__main__":
    # multi_query_rag()
    contextual_compression()