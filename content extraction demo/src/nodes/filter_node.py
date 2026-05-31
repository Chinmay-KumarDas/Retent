import json
from typing import List
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from src.schemas.concept_models import MetaFilterOutput
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Initialize LLM and bind schema
llm = ChatOllama(model="qwen2.5:3b", temperature=0)
structured_filter = llm.with_structured_output(MetaFilterOutput)

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
    raw_concepts_str = json.dumps(all_raw_concepts, indent=2)
    result = chain.invoke({"raw_concepts": raw_concepts_str})
    
    return result.model_dump() if result else {"clean_concepts": [], "deleted_concepts": []}