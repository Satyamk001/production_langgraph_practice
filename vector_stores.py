"""
    EXERCISE: Create a complete vector store setup that:
    1. Takes a list of text strings
    2. Splits them into chunks
    3. Embeds the chunks using Gemini embedding model
    4. Stores in Chroma
    
    Test with sample documents.
"""


import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Import the official LangChain wrapper for Google's Gemini embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

load_dotenv()

document_text = """
 "Python is a versatile programming language used in web development, "
        "data science, machine learning, and automation. It has a simple syntax "
        "that makes it easy to learn and read.",
        "JavaScript is the language of the web. It runs in browsers and on "
        "servers with Node.js. Modern frameworks like React and Vue make "
        "building web applications efficient.",
        "Rust is a systems programming language focused on safety and "
        "performance. It prevents common bugs like null pointer dereferences "
        "and data races at compile time.",
"""

persist_dir = "./chroma_db/"

class GeminiChromaRetriever:
    def __init__(self, collection_name: str = "udemy_collection"):
        self.collection_name = collection_name
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        
        # 1. Initialize the correct LangChain-compatible Gemini embedding model
        # It automatically looks for the GEMINI_API_KEY environment variable.
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
        
        # 2. Pass the embedding function directly into Chroma so it handles all vector operations internally
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_dir
        )
        original_count = self.vector_store._collection.count()
        print(f"Persisted vector store with {original_count} documents.")

    def split_docs(self, documents: list[str]) -> list[str]:
        # split_text takes a single string, or we use split_documents if passing a list of Document objects
        # For simplicity with your input, we combine the input list if needed or loop over them.
        full_text = " ".join(documents)
        return self.text_splitter.split_text(full_text)
        
    def save_data(self, documents: list[str]):
        # 1. Split text into string chunks
        chunks = self.split_docs(documents)
        
        # 2. Convert text chunks into LangChain Document objects
        langchain_docs = [Document(page_content=chunk) for chunk in chunks]
        
        # 3. Add to Chroma. Chroma will automatically invoke self.embeddings to vectorize them!
        self.vector_store.add_documents(langchain_docs)
    
    def retrieve(self, query: str):
        # Pass a raw text string. Chroma uses its internal embedding function automatically.
        results = self.vector_store.similarity_search(query, k=2)
        print(f"Retrieved {len(results)} results for query: '{query}'")
        print(f"Search result: {results[0].page_content[:50]}...")
        print(f"Search result: {results[1].page_content[:50]}...")
        return results
    
def main():
    retriever = GeminiChromaRetriever()
    retriever.save_data([document_text])
    
    results = retriever.retrieve("What prevent bugs?") 
    
    print("\n🎯 Retrieved Results:")
    for result in results:
        print(f"- {result.page_content}\n")
        
if __name__ == "__main__":
    main()
