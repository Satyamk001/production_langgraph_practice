from langgraph.graph import StateGraph, START
from dev_agent.state import DevState
from dev_agent.nodes.supervisor import supervisor
from dev_agent.nodes.planner import planner
from dev_agent.nodes.coder import coder, code_file
from dev_agent.nodes.reviewer import reviewer

def build_app():
    b = StateGraph(DevState)
    b.add_node("supervisor", supervisor)
    b.add_node("planner", planner)
    b.add_node("coder", coder)
    b.add_node("code_file", code_file)
    b.add_node("reviewer", reviewer)

    b.add_edge(START, "supervisor")
    b.add_edge("code_file", "supervisor")
    return b.compile()

if __name__ == "__main__":
    app = build_app()
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
