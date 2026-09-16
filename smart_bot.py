"""
Section 1 Project: Smart Q&A Bot
A production-ready question-answering bot with structured output
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
from langsmith import traceable, Client
import os

load_dotenv()

@traceable(project_name=os.getenv("LANGSMITH_PROJECT", "udemy"))
def main():
    
    # Define a Pydantic model for structured output
    class Answer(BaseModel):
        answer: str = Field(description="The answer to the question")
        sources: List[str] = Field(description="List of sources for the answer")
        confidence: float = Field(description="Confidence score of the answer", ge=0.0, le=1.0)
        reasoning: str = Field(description="Reasoning behind the answer")
        follow_up_questions: List[str] = Field(description="List of follow-up questions for further exploration")
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a knowledgeable Q&A assistant.

            Your guidelines:
            - Answer questions accurately and concisely
            - Be honest about uncertainty - set confidence to 'low' if unsure
            - Provide clear reasoning for your answers
            - Suggest relevant follow-up questions
            - Indicate if external sources would help

            Always respond with accurate, helpful information.""",), 
        ("human", "{question}")
    ])
    
    llm = init_chat_model(model="gemini-3.6-flash", model_provider="google_genai", temperature=0)
    llm_with_structured_output = llm.with_structured_output(Answer)
    
    question = "What is LangGraph?"
    
    chain = prompt | llm_with_structured_output
    response = chain.invoke({"question": question})
    print("="*50)
    print(response)
    print("="*50)
    
if __name__ == "__main__":
    main()