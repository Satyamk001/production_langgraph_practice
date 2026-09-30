from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import Command
from langgraph.graph import END
from dev_agent.config import llm
from dev_agent.state import DevState

MAX_ATTEMPTS = 3 
class Route(BaseModel):
    next: Literal["planner", "coder", "reviewer"]
    reason: str = Field(description="Ye agent kyun chuna")
    
router_llm = llm.with_structured_output(Route)

SUPERVISOR_PROMPT = """Tum DevAgent ke Supervisor ho. Status dekh ke next specialist chuno:
- Plan nahi bana: planner
- Naya code review ke intezaar mein hai (needs_review True): reviewer
- Plan hai par code abhi nahi likha, ya tests fail hue: coder"""

def status_summary(state: DevState) -> str:
    return (
        f"Task: {state['task']}\n"
        f"Plan bana hai: {bool(state.get('plan'))}\n"
        f"Code likha gaya: {bool(state.get('code_results'))}\n"
        f"needs_review: {state.get('needs_review', False)}\n"
        f"Reviewer kitni baar chala: {state.get('attempts', 0)}\n"
        f"Aakhri test report: {state.get('test_report', '(abhi tak nahi)')}"
    )

def supervisor(state: DevState) -> Command[Literal["planner", "coder", "reviewer", "__end__"]]:
    if state.get("tests_passed"):
        return Command(goto=END, update={"final_answer": "✅ Tests pass. " + state["test_report"]})
    if state.get("attempts", 0) >= MAX_ATTEMPTS:
        return Command(goto=END, update={"final_answer": "⚠️ Max attempts ho gaye. Aakhri report: " + state.get("test_report", "")})
    
    decision = router_llm.invoke([
        SystemMessage(content=SUPERVISOR_PROMPT),
        HumanMessage(content=status_summary(state)),
    ])
    return Command(goto=decision.next, update={
        "messages": [AIMessage(content=f"[Supervisor → {decision.next}] {decision.reason}", name="supervisor")],
    })
