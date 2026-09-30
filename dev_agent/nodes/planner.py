from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import Command
from dev_agent.config import llm
from dev_agent.state import DevState

class FileTask(BaseModel):
    file: str = Field(description="File ka naam, jaise email_validator.py")
    spec: str = Field(description="Is file mein kya likhna hai, 1-2 line")

class Plan(BaseModel):
    files: list[FileTask]
    
planner_llm = llm.with_structured_output(Plan)
    
def planner(state: DevState) -> Command[Literal["supervisor"]]:
    plan = planner_llm.invoke([
        SystemMessage(content="Tum Planner ho. Task ko 1 se 4 chhoti Python files mein todo. Har file ka naam aur spec do."),
        HumanMessage(content=state["task"]),
    ])
    tasks = [t.model_dump() for t in plan.files]
    return Command(goto="supervisor", update={
        "plan": tasks,
        "messages": [AIMessage(content=f"[Planner] Files: {[t['file'] for t in tasks]}", name="planner")],
    })
