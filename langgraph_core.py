"""
EXERCISE: Create a LangGraph that:
1. Takes a topic as input
2. Node 1: Generates 3 questions about the topic
3. Node 2: Answers one of the questions
4. Returns both questions and answer
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
import operator

load_dotenv()

def main():
    llm = init_chat_model(model="gemini-3.6-flash",model_provider="google_genai", temperature=0)
    
    class QAState(TypedDict):
        topic: str
        questions: str
        answer: str
        
    graph = StateGraph(QAState)
    
    def generate_questions(state:QAState)-> dict:
        response = llm.invoke(
            f"Generate 3 interesting questions about {state['topic']} in 10 words\n"
            "Format: number list"
        )
        return {"questions": response.content}
    
    def generate_answers(state: QAState)-> dict:
        response = llm.invoke(
            f"Answer any one of the question from the list {state['questions']} in 50 words"
        )
        return {"answer": response.content}

    graph.add_node("generate_questions", generate_questions)
    graph.add_node("generate_answers", generate_answers)
    
    graph.add_edge(START, "generate_questions")
    graph.add_edge("generate_questions", "generate_answers")
    graph.add_edge("generate_answers", END)
    
    app = graph.compile()
    
    result = app.invoke({"topic": "The future of renewable energy"})
    
    print(f"topic : {result['topic']}")
    print(f"Questions : {result['questions']}")
    print(f"Answer : {result['answer']}")
    


if __name__ == "__main__":
    main()