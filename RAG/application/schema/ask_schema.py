from pydantic import BaseModel

class QuerySchema(BaseModel):
    query: str

class ResponseSchema(BaseModel):
    response: str
