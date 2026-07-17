from typing import List, Optional

from pydantic import BaseModel, Field


class Movie(BaseModel):
    title: str = "NULL"
    release_year: Optional[int] = None
    genre: List[str] = Field(default_factory=list)
    director: Optional[str] = None
    cast: List[str] = Field(default_factory=list)
    rating: Optional[float] = None
    summary: str = "NULL"
