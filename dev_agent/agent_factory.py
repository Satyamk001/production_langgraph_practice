from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import MessagesState
from dev_agent.config import llm

def make_tool_agent(tools):
    """ReAct Loop : agent node + ToolNode + loop-back edge"""
    bound = llm.bind_tools(tools)
    
    def agent_node(state: MessagesState) -> dict:
        return {"messages": [bound.invoke(state["messages"])]}
    
    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    
    return graph.compile()
