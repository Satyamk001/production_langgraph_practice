from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def basic_chain():
    
    prompt = ChatPromptTemplate.from_template("Hello, I am {name}. Can you tell me about langgraph?")
    
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)
    
    parser = StrOutputParser()
    
    chain = prompt | llm | parser
    
    response = chain.invoke({"name": "Satyam"})
    print(response)
    
def batch_chain():
    prompt = ChatPromptTemplate.from_template("{text}, give answer in 20 words.")
    
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)
    
    parser = StrOutputParser()
    
    inputs = [{"text": "Hello, I am Satyam. Can you tell me about langgraph?"}, {"text": "Hello, I am Satyam. Can you tell me about langchain?"}]
    
    chain = prompt | llm | parser
    
    responses = chain.batch(inputs)
    
    for i, res in enumerate(responses):
        print(f"Response {i}: {res}")
    
def chain_streaming():
    prompt = ChatPromptTemplate.from_template("Hello, I am {name}. Can you tell me about langgraph?")
    
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0, streaming=True)
    
    parser = StrOutputParser()
    
    chain = prompt | llm | parser
    
    response = chain.stream({"name": "Satyam"})
    for chunk in response:
        print(chunk, end="", flush=False) # flush=True ensures that the output is printed immediately, end="" avoids adding a new line after each chunk
        
def my_first_chain():
    prompt = ChatPromptTemplate.from_template("write a poem about {topic} in 20 words.")
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)
    parser = StrOutputParser()
    
    chain = prompt | llm | parser
    
    response = chain.invoke({"nature"})
    print(response)

if __name__ == "__main__":
    # basic_chain()
    # batch_chain()
    # chain_streaming()
    my_first_chain()