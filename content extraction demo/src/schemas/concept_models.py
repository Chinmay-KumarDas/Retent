from typing import List
from pydantic import BaseModel, Field

# Step 1 Schemas (Extractor)
class CoreConcept(BaseModel):
    title: str = Field(description="A short, distinct title for the concept")
    explanation: str = Field(description="A clear, academic explanation of the concept in Markdown")
    key_takeaways: List[str] = Field(description="1-3 bullet points summarizing the concept")

class ConceptExtraction(BaseModel):
    concepts: List[CoreConcept] = Field(description="List of extracted core concepts")

# Step 2 Schemas (MetaFilter Tracker)
class DeletedConcept(BaseModel):
    original_title: str = Field(description="The title of the concept that was deleted or merged")
    reason: str = Field(description="Why it was deleted (e.g., 'Activity', 'Fictional Character', 'Merged with Concept X')")

class MetaFilterOutput(BaseModel):
    clean_concepts: List[CoreConcept] = Field(description="The final, compressed list of high-quality concepts")
    deleted_concepts: List[DeletedConcept] = Field(description="Log of concepts that were removed and why")