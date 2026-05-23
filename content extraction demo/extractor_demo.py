import json
import logging
import fitz  # PyMuPDF
from typing import List
import re
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================
# 1. SETUP LOGGING
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==========================================
# 2. DEFINE PYDANTIC SCHEMAS (JSON Structure)
# ==========================================
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

# ==========================================
# 3. INITIALIZE LLM PIPELINES
# ==========================================
# Ensure Ollama is running in the background with Qwen 2.5!
llm = ChatOllama(model="qwen2.5:3b", temperature=0)

# We create two separate structured output binds for the two different tasks
structured_extractor = llm.with_structured_output(ConceptExtraction)
structured_filter = llm.with_structured_output(MetaFilterOutput)

# ==========================================
# 4. DEFINE PIPELINE FUNCTIONS (Linear Flow)
# ==========================================
def extract_concepts_from_chunk(chunk_text: str, chunk_index: int, total_chunks: int) -> List[dict]:
    logger.info(f"--- [EXTRACTOR] Processing Chunk {chunk_index}/{total_chunks} ---")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert curriculum designer. Extract atomic academic concepts from the provided text. "
                   "Ignore conversational dialogue, classroom activities, and chapter intro filler. "
                   "Extract ONLY factual, scientific, and academic concepts. "
                   "CRITICAL: You MUST provide 1 to 3 bullet points in the 'key_takeaways' array for EVERY concept."),
        ("human", "SOURCE TEXT:\n\n{raw_text}")
    ])
    
    chain = prompt | structured_extractor
    result = chain.invoke({"raw_text": chunk_text})
    
    # Use model_dump() instead of dict() to fix Pydantic V2 warnings
    extracted_list = result.model_dump().get("concepts", []) if result else []
    logger.info(f"Extracted {len(extracted_list)} concepts from chunk {chunk_index}.")
    
    return extracted_list

def meta_filter_node(all_raw_concepts: List[dict]) -> dict:
    logger.info("--- [META-FILTER] Purging noise and compressing concepts... ---")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a strict academic Lead Editor for a Spaced Repetition platform. 
Your job is to review a raw list of extracted concepts, purge the noise, merge redundancies, and output a final clean list.

CRITICAL PURGE RULES - You MUST DELETE any concept that:
1. Is an activity, crossword puzzle, diagram layout, or flow chart.
2. Contains fictional student characters (e.g., Boojho, Paheli, Sabiha).
3. Contains meta-commentary (e.g., "The text mentions...", "This chapter discusses...").
4. Asks a question but provides no scientific answer.

COMPRESSION RULES:
- Merge duplicate or highly similar concepts into a single, comprehensive concept.
- Prioritize high-level, atomic academic facts (e.g., definitions, processes).

You must return the 'clean_concepts' AND log any removed or merged items in 'deleted_concepts' with a brief reason."""),
        ("human", "RAW CONCEPTS:\n{raw_concepts}")
    ])
    
    chain = prompt | structured_filter
    
    # Convert the raw concepts to a formatted string for the prompt
    raw_concepts_str = json.dumps(all_raw_concepts, indent=2)
    result = chain.invoke({"raw_concepts": raw_concepts_str})
    
    return result.model_dump() if result else {"clean_concepts": [], "deleted_concepts": []}

# ==========================================
# 5. PDF PROCESSING & EXECUTION
# ==========================================
def process_pdf(pdf_path: str):
    logger.info(f"Loading PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    full_text = ""
    
    for page in doc:
        full_text += page.get_text()

    # ==========================================
    # LANDMINE AVOIDANCE
    # ==========================================
    cutoff_phrases = ["WHAT YOU HAVE LEARNT", "EXERCISES"]
    for phrase in cutoff_phrases:
        if phrase in full_text:
            logger.info(f"Found cutoff phrase '{phrase}'. Truncating document.")
            full_text = full_text.split(phrase)[0]
            break
        
    # ==========================================
    # HYBRID CHUNKING
    # ==========================================
    logger.info("Performing Hybrid Chunking...")
    heading_pattern = r'\n(?=\d+\.\d+\s+[A-Z])'
    raw_sections = re.split(heading_pattern, full_text)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". "]
    )
    
    final_chunks = []
    for section in raw_sections:
        cleaned_section = section.strip()
        if len(cleaned_section) > 100:
            if len(cleaned_section) > 1500:
                sub_chunks = text_splitter.split_text(cleaned_section)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(cleaned_section)

    logger.info(f"Created {len(final_chunks)} hybrid chunks.")
    
    final_clean_concepts = []
    all_deleted_concepts = []
    
    for i, chunk in enumerate(final_chunks):
        # Step 1: Extract from the single chunk (Yields ~3 to 8 concepts)
        raw_chunk_concepts = extract_concepts_from_chunk(chunk, i + 1, len(final_chunks))
        
        if not raw_chunk_concepts:
            continue
            
        # Step 2: Immediately filter those few concepts while the list is short
        logger.info(f"--- [META-FILTER] Cleaning Chunk {i + 1} ---")
        filter_result = meta_filter_node(raw_chunk_concepts)
        
        final_clean_concepts.extend(filter_result.get("clean_concepts", []))
        all_deleted_concepts.extend(filter_result.get("deleted_concepts", []))

    logger.info(f"Pipeline Complete. Extracted {len(final_clean_concepts)} clean concepts total.")

    # ==========================================
    # SAVE TO FILES
    # ==========================================
    with open("clean_concepts_output.json", "w", encoding="utf-8") as f:
        json.dump(final_clean_concepts, f, indent=4, ensure_ascii=False)
        
    with open("deleted_concepts_log.json", "w", encoding="utf-8") as f:
        json.dump(all_deleted_concepts, f, indent=4, ensure_ascii=False)
    
    logger.info(f"\nSUCCESS!")
    logger.info(f"Final Clean Concepts: {len(final_clean_concepts)} (Saved to clean_concepts_output.json)")
    logger.info(f"Deleted/Merged Concepts: {len(all_deleted_concepts)} (Saved to deleted_concepts_log.json)")

if __name__ == "__main__":
    process_pdf("hesc101.pdf")