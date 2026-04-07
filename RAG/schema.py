from pydantic import BaseModel

class ResponseWithImages(BaseModel):
    text: str
    image_ids: list[str]