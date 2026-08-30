from pydantic import BaseModel
from  typing import List
from .task import Task

class Plan(BaseModel):
    title: str
    tasks : List[Task]