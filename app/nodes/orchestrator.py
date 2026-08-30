from langchain_core.messages import SystemMessage, HumanMessage
from app.graphs.blog.states.state import State
from app.schemas.plan import Plan
from app.utils.prompt_loader import load_prompt

def orchestrator_node(llm):
    def orchestrator(state: State ) -> dict:
        system_prompt = load_prompt("../prompts/orchestrator.md")
        plan = llm.with_structured_output(Plan).invoke(
           [
               SystemMessage(content=system_prompt),
               HumanMessage(content=f"Topic is : ${state["topic"]}"),
           ]
        )
        return {"plan": plan}

    return orchestrator

