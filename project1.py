"""
User task deta hai → 
Supervisor decide karta hai next kaun → 
Planner files ki list banata hai → 
Coder har file ke liye ek helper parallel (Send) bhejta hai, jo khud ko self-correct karta hai → 
Reviewer tools se pytest chalata hai → fail hua to kaam Coder ko wapas handoff, pass hua to khatam.
"""

from dotenv import load_dotenv
load_dotenv()

import json
import operator
import subprocess
import sys
from pathlib import Path

from typing_extensions import TypedDict, Annotated
from typing import Literal
from pydantic import BaseModel , Field
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage , HumanMessage
from langgraph.graph import StateGraph, START , END, MessagesState
from langgraph.graph.message import add_messages 
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command , Send

llm = init_chat_model(
    model="gemini-3.8-flash",
    model_provider="google_genai",
    temperature=0,
    max_retries=5,   # default is 2 — bump it up for a multi-agent loop like this
)
WORKSPACE = Path("workspace")
WORKSPACE.mkdir(exist_ok=True)

# 1️⃣ Step 1: Tools 

def safe_path(name : str) -> str:
    """Sirf workspace ke andar ki files allow karo"""
    root = WORKSPACE.resolve()
    p = (root / name).resolve()
    
    if(root not in p.parents):
        raise ValueError(f"{name} is outside allowd workspace")
    return p

@tool
def list_files() -> str:
    """List all the files under workspace directory"""
    names = sorted(p.name for p in WORKSPACE.iterdir())
    return "\n".join(names) or "Workspace is empty"

@tool
def read_file(name: str) -> str:
    """Read a txt file from workspace"""
    try:
        return safe_path(name).read_text()
    except FileNotFoundError:
        return f"Error: '{name}' does not exist in workspace. check name using list_files"
    except ValueError as e:
        return f"Error: {e}"
    
@tool 
def write_file(name: str, content: str) -> str:
    """Write text content to a file in the workspace (overwrite if it exists)"""
    try:
        safe_path(name).write_text(content)
        return f"OK: '{name}' written ({len(content)} chars)"
    except ValueError as e:
        return f"Error: {e}"
    
@tool 
def run_pytest(name : str="") -> str:
    """Run pytest inside the workspace (optionally one test file). Returns the pytest output."""
    cmd = [sys.executable, "-m", "pytest", "-q", "--tb=short"] + ([name] if name else [])
    try:
        r = subprocess.run(cmd, cwd=WORKSPACE, capture_output=True, text=True, timeout=60)
        return (r.stdout + r.stderr)[-3000:] or "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: pytest did not end in 60 sec (timeout)"
    
# 2️⃣ Step 2: Tool-calling agent factory

def make_tool_agent(tools):
    """ReAct Loop : agent node + ToolNode + loop-back edge"""
    bound = llm.bind_tools(tools)
    
    def agent_node(state: MessagesState) -> dict:
        return {"messages": [bound.invoke(state["messages"])]}
    
    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools",ToolNode(tools, handle_tool_errors=True))
    
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    
    return graph.compile()

# 3️⃣ Step 3: State

class DevState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]    # conversation log
    task: str                                                # user ka kaam
    plan: list[dict]                                         # [{"file", "spec"}] (Wed ke `files` jaisa)
    code_results: Annotated[list[dict], operator.add]        # Wed ke `reviews` jaisa (reducer!)
    needs_review: bool                                       # naya code review ka intezaar
    tests_passed: bool
    test_report: str
    attempts: int                                            # kitni baar reviewer chala
    final_answer: str

class CodeTask(TypedDict):                                   # ek parallel coder ka input
    file: str
    spec: str
    feedback: str                                            # reviewer ka feedback (pehli baar khali)
    
# 4️⃣ Step 4: Coder = self-correcting subgraph

MAX_FIXES = 3

class CoderState(TypedDict): # each invocation creates its own CoderState.
    file: str
    spec: str
    feedback: str
    code: str
    error: str
    tries: int
    
def strip_fences(text: str) -> str:
    """LLM kabhi ```python ... ``` mein code deta hai, wo hata do."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rstrip()
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()

def generate(state: CoderState) -> dict:
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
    """Syntax check. (Asli project mein yahan lint bhi laga sakte ho.)"""
    try:
        compile(state["code"], state["file"], "exec")
        return {"error": ""}
    except SyntaxError as e:
        return {"error": f"SyntaxError: {e.msg} (line {e.lineno})"}
    
def after_check(state: CoderState) -> Literal["generate", "done"]:
    if not state["error"] or state["tries"] >= MAX_FIXES:
        return "done"
    return "generate"

cb = StateGraph(CoderState)
cb.add_node("generate", generate)
cb.add_node("check", check)
cb.add_edge(START, "generate")
cb.add_edge("generate", "check")
cb.add_conditional_edges("check", after_check, {"generate": "generate", "done": END})
coder_graph = cb.compile()

def code_file(task: CodeTask) -> dict:
    """Ek file ka coder. Ye node parallel mein N baar chalta hai (Send)."""
    out = coder_graph.invoke({
        "file": task["file"], "spec": task["spec"], "feedback": task.get("feedback", ""),
        "code": "", "error": "", "tries": 0,
    })
    safe_path(task["file"]).write_text(out["code"])          # code workspace mein likh do
    return {"code_results": [{
        "file": task["file"], "syntax_ok": not out["error"], "tries": out["tries"],
    }]}
    
# 5️⃣ Step 5: Planner aur Supervisor 

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

# 💻 Supervisor: rules + LLM (hybrid)

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
    # Rule 1: tests pass ho gaye to khatam
    if state.get("tests_passed"):
        return Command(goto=END, update={"final_answer": "✅ Tests pass. " + state["test_report"]})
    # Rule 2: attempts khatam to khatam (safety valve)
    if state.get("attempts", 0) >= MAX_ATTEMPTS:
        return Command(goto=END, update={"final_answer": "⚠️ Max attempts ho gaye. Aakhri report: " + state.get("test_report", "")})
    # Baaki: LLM triage
    decision = router_llm.invoke([
        SystemMessage(content=SUPERVISOR_PROMPT),
        HumanMessage(content=status_summary(state)),
    ])
    return Command(goto=decision.next, update={
        "messages": [AIMessage(content=f"[Supervisor → {decision.next}] {decision.reason}", name="supervisor")],
    })
    
# 6️⃣ Step 6: Coder dispatch
def coder(state: DevState) -> Command[Literal["code_file"]]:
    """MAP : plan ki har file ke liye ek parallel code_file."""
    # Agar pehle tests fail hue the to reviewer ka feedback coder ko do
    feedback = state.get("test_report", "") if state.get("attempts", 0) > 0 else ""
    sends = [
        Send("code_file", {"file": t["file"], "spec": t["spec"], "feedback": feedback})
        for t in state["plan"]
    ]
    return Command(goto=sends, update={"needs_review": True})   # naya code aayega, review chahiye

# 💻 Reviewer: tools wala agent + structured verdict
class TestVerdict(BaseModel):
    passed: bool = Field(description="Kya saare tests pass hue")
    report: str = Field(description="Chhota summary: kya pass, kya fail, kyun")

reviewer_agent = make_tool_agent([list_files, read_file, write_file, run_pytest])
verdict_llm = llm.with_structured_output(TestVerdict)

def reviewer(state: DevState) -> Command[Literal["supervisor"]]:
    files = ", ".join(t["file"] for t in state["plan"])
    prompt = (
        f"Task: {state['task']}\nFiles: {files}\n"
        "1) read_file se code padho\n"
        "2) write_file se test_<naam>.py mein pytest tests likho\n"
        "3) run_pytest chalao\n"
        "4) End mein saaf batao kaunse tests pass hue aur kaunse fail (aur kyun)."
    )
    out = reviewer_agent.invoke(
        {"messages": [SystemMessage(content="Tum Reviewer aur Tester ho."), HumanMessage(content=prompt)]},
        config={"recursion_limit": 30},
    )
    final_text = out["messages"][-1].content
    # final_text = out["messages"][-1].text
    verdict = verdict_llm.invoke([
        SystemMessage(content="Neeche tester ka final message hai. Batao saare tests pass hue ya nahi."),
        HumanMessage(content=final_text),
    ])
    return Command(goto="supervisor", update={
        "tests_passed": verdict.passed,
        "test_report": verdict.report,
        "needs_review": False,
        "attempts": state.get("attempts", 0) + 1,
        "messages": [AIMessage(content=f"[Reviewer] {'PASS' if verdict.passed else 'FAIL'}: {verdict.report}", name="reviewer")],
    })

# 7️⃣ Step 7: Graph wiring aur run
b = StateGraph(DevState)
b.add_node("supervisor", supervisor)
b.add_node("planner", planner)
b.add_node("coder", coder)
b.add_node("code_file", code_file)      # Send ka target (subgraph wrapper)
b.add_node("reviewer", reviewer)

b.add_edge(START, "supervisor")
b.add_edge("code_file", "supervisor")   # FAN-IN: saare parallel coders ke baad supervisor
# baaki saari jumping Command ke goto se hoti hai
app = b.compile()

result = app.invoke(
    {
        "task": "write code which validates email , 10 digit phone number, date of birth (mm-dd-yyyy) format only. All validator in different .py files",
        "plan": [], "code_results": [], "messages": [],
        "needs_review": False, "attempts": 0,
    },
    config={"recursion_limit": 40},
)
print(result["final_answer"])
for m in result["messages"]:
    print(m.content)