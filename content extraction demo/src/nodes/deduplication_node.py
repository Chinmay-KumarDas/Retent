import json
import numpy as np
from typing import List
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from src.schemas.concept_models import CoreConcept
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Initialize the LLM for the merging task
llm = ChatOllama(model="qwen2.5:3b", temperature=0)
# We force the LLM to output a SINGLE CoreConcept this time, not a list!
structured_merger = llm.with_structured_output(CoreConcept)

# Initialize the lightweight local embedder
embedder = OllamaEmbeddings(model="nomic-embed-text")

def calculate_cosine_similarity(vec1, vec2):
    """Calculates the cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def merge_cluster_with_llm(cluster: List[dict]) -> dict:
    """Takes a list of similar concepts and uses the LLM to merge them into one master concept."""
    if len(cluster) == 1:
        return cluster[0] 
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert academic editor. I am giving you a cluster of highly similar, redundant concepts extracted from a textbook.
Your job is to merge them into ONE master concept. 
- Create a distinct, overarching title.
- Combine the best parts of their explanations.
- Consolidate all the bullet points into 2-4 comprehensive 'key_takeaways'.
Do not lose any critical scientific facts, but completely eliminate the redundancy."""),
        ("human", "REDUNDANT CONCEPTS TO MERGE:\n{cluster_data}")
    ])
    
    chain = prompt | structured_merger
    cluster_str = json.dumps(cluster, indent=2)
    
    result = chain.invoke({"cluster_data": cluster_str})
    
    return result.model_dump() if result else cluster[0]

def semantic_deduplication(concepts: List[dict], similarity_threshold: float = 0.82) -> List[dict]:
    # 🔵 Standard Info Log (Blue)
    logger.info(f"--- [DEDUPLICATION] Vectorizing {len(concepts)} concepts for semantic clustering ---")
    
    if not concepts:
        # 🟡 Warning Log (Yellow)
        logger.warning("No concepts were passed to the deduplication node!")
        return []

    texts_to_embed = [f"{c.get('title', '')}. {c.get('explanation', '')}" for c in concepts]
    embeddings = embedder.embed_documents(texts_to_embed)
    
    clusters = []
    visited = set()
    
    for i in range(len(concepts)):
        if i in visited:
            continue
            
        current_cluster = [concepts[i]]
        visited.add(i)
        
        for j in range(i + 1, len(concepts)):
            if j in visited:
                continue
                
            sim_score = calculate_cosine_similarity(embeddings[i], embeddings[j])
            
            if sim_score >= similarity_threshold:
                current_cluster.append(concepts[j])
                visited.add(j)
                
        clusters.append(current_cluster)

    # 🟢 Success Log (Green)
    logger.success(f"Math Complete: Clustered {len(concepts)} raw concepts into {len(clusters)} unique groups.")
    
    final_master_concepts = []
    for idx, cluster in enumerate(clusters):
        if len(cluster) > 1:
            # 🔵 Info Log (Blue) - So we can watch the LLM work through the duplicates
            logger.info(f"Merging Cluster {idx + 1}/{len(clusters)} (Contains {len(cluster)} redundant items)...")
        
        master_concept = merge_cluster_with_llm(cluster)
        final_master_concepts.append(master_concept)

    # 🟢 Success Log (Green)
    logger.success(f"Deduplication Complete! Returning {len(final_master_concepts)} master concepts.")
    return final_master_concepts