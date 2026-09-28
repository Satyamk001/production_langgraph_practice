from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

class ApprovalState(TypedDict):
    recipient: str
    message: str
    action: Optional[str]
    approved: Optional[bool]
    result: Optional[str]


def draft_action(state: ApprovalState) -> dict:
    action = f"Send email to {state['recipient']}: '{state['message']}'"
    return {"action": action}


def request_approval(state: ApprovalState) -> dict:
    decision = interrupt({
        "question": "Approve this action before it runs?",
        "action": state["action"],
    })
    return {"approved": decision}


def execute_action(state: ApprovalState) -> dict:
    # In real life: actually send the email / make the API call here
    return {"result": f"✅ Executed: {state['action']}"}


def cancel_action(state: ApprovalState) -> dict:
    return {"result": "❌ Action cancelled by human reviewer."}


def route_after_approval(state: ApprovalState) -> str:
    return "execute" if state["approved"] else "cancel"


builder = StateGraph(ApprovalState)
builder.add_node("draft_action", draft_action)
builder.add_node("request_approval", request_approval)
builder.add_node("execute_action", execute_action)
builder.add_node("cancel_action", cancel_action)

builder.add_edge(START, "draft_action")
builder.add_edge("draft_action", "request_approval")
builder.add_conditional_edges(
    "request_approval",
    route_after_approval,
    {"execute": "execute_action", "cancel": "cancel_action"},
)
builder.add_edge("execute_action", END)
builder.add_edge("cancel_action", END)

checkpointer = MemorySaver()
app = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "approval-session-1"}}

# --- Step 1: Run until it pauses for approval ---
result = app.invoke(
    {"recipient": "boss@company.com", "message": "Q3 report is ready", "action": None, "approved": None, "result": None},
    config=config,
)
print("Paused for approval:", result["__interrupt__"])

# --- Step 2: Human reviews and approves (True) or rejects (False) ---
final_result = app.invoke(Command(resume=False), config=config)
print("Final result:", final_result["result"])