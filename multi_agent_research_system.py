import os 
from dotenv import load_dotenv
import operator
import json

from typing import TypedDict, Annotated, Literal
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, BaseMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

llm = init_chat_model(model="gemini-3.6-flash",model_provider="google_genai", temperature=0)


class ResearchState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    topic: str
    search_queries: list[str]
    findings: Annotated[list[dict], operator.add]
    analysis: str
    report: str
    quality_score: float
    quality_feedback: str
    iteration: int

    
def superVisorAgent(state: ResearchState) -> dict:
    """Plans research by generating targeted search queries."""
    
    response = llm.invoke(
        [
            SystemMessage(
                 content=(
                    "You are a research supervisor. Given a topic, generate exactly 3 "
                    "specific search queries that will cover different angles of the topic. "
                    "Return ONLY a JSON array of strings. No markdown formatting."
                )
            ),
            HumanMessage(
                content=f"Research topic: {state['topic']}"
            )
        ]
    )
    
    return {
        response
    }