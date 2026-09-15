from dotenv import load_dotenv
from langchain_core import __version__ as langchain_core_version
from langgraph import version as langgraph_version
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

load_dotenv()

print(f"langchain_core version: {langchain_core_version}")
print(f"langgraph version: {langgraph_version}")

def main():
    print("Hello from langgraph")
    
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)
    response = llm.invoke("hi , i am satyam, can you tell me about langgraph?")

    print(response.content)

if __name__ == "__main__":
    main()