from langgraph.types import Send

from app.graphs.blog.states.state import State


def fanout(state: State):
    return [
        Send(
            "worker",
            {
                "task": task,
                "topic": state["topic"],
                "plan": state["plan"],
            },
        )
        for task in state["plan"].tasks
    ]