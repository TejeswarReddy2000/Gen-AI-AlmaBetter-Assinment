from pydantic import BaseModel
from typing import List

class PaperSection(BaseModel):
    section_name: str
    content: str

class ResearchPaper(BaseModel):
    paper_id: str
    title: str
    authors: List[str]
    abstract: str
    year: int
    venue: str
    keywords: List[str]
    sections: List[PaperSection]
