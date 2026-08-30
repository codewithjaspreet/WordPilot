from typing import TypedDict, Annotated, List
import operator
from app.schemas.plan import Plan

class State(TypedDict):
    topic: str
    plan: Plan
    sections: Annotated[List[str], operator.add]
    final: str