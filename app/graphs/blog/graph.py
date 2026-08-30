from __future__ import annotations

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

from app.graphs.blog.routing import fanout
from app.llm.groq_provider import GroqProvider
from app.nodes.orchestrator import orchestrator_node
from app.nodes.reducer import reducer_node
from app.nodes.worker import worker_node
from app.graphs.blog.states.state import State

load_dotenv()

llm = GroqProvider().create("openai/gpt-oss-120b")

workflow = StateGraph(State)

workflow.add_node("orchestrator", orchestrator_node(llm))
workflow.add_node("worker", worker_node(llm))
workflow.add_node("reducer", reducer_node(llm))

workflow.add_edge(START, "orchestrator")
workflow.add_conditional_edges("orchestrator", fanout, ["worker"])
workflow.add_edge("worker", "reducer")
workflow.add_edge("reducer", END)

app = workflow.compile()

out = app.invoke({
    "topic": "Write a blog on Self Attention",
    "sections": []
})

print(out)