from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import Command
from dev_agent.config import llm
from dev_agent.state import DevState
from dev_agent.agent_factory import make_tool_agent
from dev_agent.tools import list_files, read_file, write_file, run_pytest

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
