from typing import List
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from src.schemas.concept_models import ConceptExtraction
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Initialize LLM and bind schema
llm = ChatOllama(model="qwen2.5:3b", temperature=0)
structured_extractor = llm.with_structured_output(ConceptExtraction)

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
    
    extracted_list = result.model_dump().get("concepts", []) if result else []
    logger.success(f"Extracted {len(extracted_list)} concepts from chunk {chunk_index}.")
    
    return extracted_list