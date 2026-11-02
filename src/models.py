"""Pydantic models shared by all the pipeline stages."""

from pydantic import BaseModel, Field
import uuid
from typing import List


class MinimalSource(BaseModel):
    """Location of a chunk: file path and range."""

    file_path: str
    first_character_index: int
    last_character_index: int


class Chunk(MinimalSource):
    """Source with its titles and text."""

    titles: List[str]
    text: str


class UnansweredQuestion(BaseModel):
    """Question with an identifier."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Question with its expected sources and answer."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Dataset of questions."""

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Sources retrieved for a question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Search results with the generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Search results of all questions."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Search results and answers of all questions."""

    search_results: List[MinimalAnswer]
    k: int
