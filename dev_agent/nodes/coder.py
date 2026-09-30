"""
Coder Node Module

This module contains the logic for generating and syntax-checking Python code.
It demonstrates a Subgraph pattern in LangGraph, where a parent graph (`DevState`)
dispatches multiple parallel child subgraphs (`coder_graph`) to handle code generation for each file.
"""

from typing import Literal
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, Send
from dev_agent.config import llm
from dev_agent.tools import safe_path
from dev_agent.state import DevState, CodeTask

MAX_FIXES = 3

class CoderState(TypedDict):
    """
    State for the inner Coder Subgraph.
    
    IMPORTANT: This state is NOT shared with the main graph's `DevState`. 
    Each time the coder subgraph is invoked for a specific file, a fresh `CoderState` 
    is created. This allows multiple files to be coded in parallel without their 
    states mixing up.
    """
    file: str           # The name of the file to write
    spec: str           # Instructions/specifications for the code
    feedback: str       # Feedback from the reviewer (if any)
    code: str           # The generated python code
    error: str          # Any syntax errors found during the check phase
    tries: int          # Counter to limit the number of self-correction attempts

def strip_fences(text: str) -> str:
    """Removes markdown code block formatting (```python ... ```) from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rstrip()
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()

def generate(state: CoderState) -> dict:
    """
    Node: Generates code based on the spec, and optionally applies feedback or fixes syntax errors.
    Returns the updated code and increments the 'tries' counter.
    """
    extra = ""
    if state.get("error"):
        extra += f"\n\nPichla code:\n{state['code']}\n\nError aaya:\n{state['error']}\nIsse fix karo."
    if state.get("feedback"):
        extra += f"\n\nReviewer ka feedback:\n{state['feedback']}"
    reply = llm.invoke([
        SystemMessage(content="Tum Coder ho. Sirf Python code likho, koi explanation nahi."),
        HumanMessage(content=f"File: {state['file']}\nSpec: {state['spec']}{extra}"),
    ])
    return {"code": strip_fences(reply.content), "tries": state.get("tries", 0) + 1}

def check(state: CoderState) -> dict:
    """
    Node: Checks if the generated code is syntactically valid Python.
    Updates the 'error' field in the state.
    """
    try:
        compile(state["code"], state["file"], "exec")
        return {"error": ""}
    except SyntaxError as e:
        return {"error": f"SyntaxError: {e.msg} (line {e.lineno})"}
    
def after_check(state: CoderState) -> Literal["generate", "done"]:
    """
    Conditional Edge: Decides whether to try fixing the code again or stop.
    Stops if there's no error, or if the maximum number of attempts (MAX_FIXES) is reached.
    """
    if not state["error"] or state["tries"] >= MAX_FIXES:
        return "done"
    return "generate"

# Build the self-correcting Subgraph for a single file
cb = StateGraph(CoderState)
cb.add_node("generate", generate)
cb.add_node("check", check)
cb.add_edge(START, "generate")
cb.add_edge("generate", "check")
cb.add_conditional_edges("check", after_check, {"generate": "generate", "done": END})
coder_graph = cb.compile()

def code_file(task: CodeTask) -> dict:
    """
    Sub-graph wrapper node.
    
    This node acts as a bridge between the main DevState and the isolated CoderState.
    It takes a `CodeTask` from the main graph, initializes the `coder_graph` with it, 
    runs the generation loop, writes the file to disk, and then returns the result 
    back to the main graph's `code_results` reducer.
    """
    out = coder_graph.invoke({
        "file": task["file"], "spec": task["spec"], "feedback": task.get("feedback", ""),
        "code": "", "error": "", "tries": 0,
    })
    safe_path(task["file"]).write_text(out["code"])
    return {"code_results": [{
        "file": task["file"], "syntax_ok": not out["error"], "tries": out["tries"],
    }]}
    
def coder(state: DevState) -> Command[Literal["code_file"]]:
    """
    Main Graph Dispatcher Node (Map phase).
    
    This node iterates over the planned files in `state["plan"]`.
    For each file, it fires off a `Send` command targeting the "code_file" node.
    This creates multiple parallel branches, where each branch processes one file 
    using the isolated subgraph (`coder_graph`).
    """
    feedback = state.get("test_report", "") if state.get("attempts", 0) > 0 else ""
    sends = [
        Send("code_file", {"file": t["file"], "spec": t["spec"], "feedback": feedback})
        for t in state["plan"]
    ]
    return Command(goto=sends, update={"needs_review": True})
