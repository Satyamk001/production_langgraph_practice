import operator
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages 

class DevState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]    # conversation log
    task: str                                                # user ka kaam
    plan: list[dict]                                         # [{"file", "spec"}]
    code_results: Annotated[list[dict], operator.add]        # code results reducer
    needs_review: bool                                       # naya code review ka intezaar
    tests_passed: bool
    test_report: str
    attempts: int                                            # kitni baar reviewer chala
    final_answer: str

class CodeTask(TypedDict):                                   # ek parallel coder ka input
    file: str
    spec: str
    feedback: str                                            # reviewer ka feedback
