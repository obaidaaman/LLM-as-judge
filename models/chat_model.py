from pydantic import BaseModel, Field


class ChatModel(BaseModel):
    query:str = Field(...,description="")
    thread_id:str = Field(...,description="")
    is_open_source:bool = False
    score:bool = Field(False)