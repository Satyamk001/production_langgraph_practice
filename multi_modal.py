"""
create a function that takes a question and list of model names, and returns a dictionary with model names as keys and their respective answers as values.   
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

def multi_model_response(question : str, models: list) -> dict:
    answer = {}
    
    def create_model(model_name):
        
        return init_chat_model(model=model_name,model_provider="google_genai", temperature=0)
    
    for model in models:
        llm = create_model(model)
        messages = [
            SystemMessage(content="You are a helpful assistant. You answer in 20 words."),
            HumanMessage(content=question)
        ]
        response = llm.invoke(messages)
        answer[model] = response.content
    return answer

def main():
    question = "What is LangGraph?"
    models = ["gemini-3.6-flash", "gemini-3.6-flash"]
    answers = multi_model_response(question, models)
    
    for model, answer in answers.items():
        print(f"Model: {model}, Answer: {answer}")
        
if __name__ == "__main__":
    main()