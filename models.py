from datetime import datetime

from pydantic import BaseModel, Field


class Query(BaseModel):
  query: str = Field(min_length=1, max_length=500)
  date: str = Field(min_length=1, max_length=100, default_factory=lambda: datetime.now().isoformat())


class Response(BaseModel):
  response: str = Field(min_length=1)
  status: int = Field(200)
  query: str = Field(min_length=1, max_length=600)
  date: str = Field(min_length=1, max_length=100, default_factory=lambda: datetime.now().isoformat())
