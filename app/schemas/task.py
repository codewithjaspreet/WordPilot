from pydantic import BaseModel, Field


class Task(BaseModel):
    title: str
    id: str
    description: str = Field(description= 'Description of the task')
