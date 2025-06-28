from pydantic import BaseModel


class QuerySchema(BaseModel):
    query: str

class ResponseSchema(BaseModel):
    response: str


class MessageResponseSchema(BaseModel):
    text: str
    image_ids: list[str] | None = None