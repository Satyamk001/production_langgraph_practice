from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model(model="gemini-3.6-flash",model_provider="google_genai", temperature=0)

def main():
    class QualityState(TypedDict):
        content: str
        quality_score: int
        feedback: str
        final_content: str
        iteration: int


        
    def evaluate_quality(state: QualityState) -> dict:
        response = llm.invoke(
            f"Rate this content quality from 1-10. Reply with just the number.\n\n"
            f"Content: {state['content']}"
            )
        try:
            score = int(response.content.strip())
        except:
            score = 5
        return {"quality_score": score}
    
    def improve_content(state: QualityState) -> dict:
        print(f"iteration : {state['iteration']}")
        response = llm.invoke(
            f"Improve this content to be more engaging and clear:\n\n{state['content']}"
        )
        return {"content": response.content, "iteration": state["iteration"] + 1}
    
    def finalize_content(state: QualityState) -> dict:
        return {
            "final_content": state["content"],
            "feedback": f"Approved after {state['iteration']} iterations with score {state['quality_score']}",
        }
        
    def should_continue(state: QualityState) -> Literal["improve", "finalize"]:
        if state["quality_score"] >= 7:
            return "finalize"
        elif state["iteration"] >= 3:
            return "finalize"  # Max iterations
        else:
            return "improve"
        
    graph = StateGraph(QualityState)
    
    graph.add_node("evaluate", evaluate_quality)
    graph.add_node("improve", improve_content)
    graph.add_node("finalize", finalize_content)

    
    graph.add_edge(START, "evaluate")
    graph.add_conditional_edges("evaluate", 
                                should_continue, 
                                {
                                    "improve": "improve",
                                    "finalize": "finalize"
                                }
                                )
    graph.add_edge("improve", "evaluate")
    graph.add_edge("finalize", END)
    
    app = graph.compile()
    result = app.invoke({
            "content": "AI is cool",
            "quality_score": 0,
            "feedback": "",
            "final_content": "",
            "iteration": 0,
        })
    print(f"content : {result['content']}")
    print(f"quality_score : {result['quality_score']}")
    print(f"feedback : {result['feedback']}")
    print(f"final_content : {result['final_content']}")
    print(f"iteration : {result['iteration']}")

if __name__ == "__main__":
  main()