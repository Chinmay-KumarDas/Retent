import json
import os
from src.utils.logger import get_logger
from src.utils.text_chunker import get_chunks_from_pdf
from src.nodes.extraction_node import extract_concepts_from_chunk
from src.nodes.filter_node import meta_filter_node
from src.nodes.deduplication_node import semantic_deduplication

logger = get_logger(__name__)

def run_pipeline(pdf_path: str):
    # ℹ️ INFO: High-level state changes
    logger.info("Starting Extraction Pipeline...")
    
    try:
        # ==========================================
        # PHASE 1 & 2: EXTRACTION AND MICRO-FILTERING
        # ==========================================
        # ⚙️ EXECUTING: Granular steps
        logger.debug(f"Loading and chunking document: {pdf_path}")
        final_chunks = get_chunks_from_pdf(pdf_path)
        
        all_raw_concepts = []       
        final_clean_concepts = []   
        all_deleted_concepts = []   
        
        for i, chunk in enumerate(final_chunks):
            logger.debug(f"Processing Chunk {i + 1}/{len(final_chunks)}...")
            
            raw_chunk_concepts = extract_concepts_from_chunk(chunk, i + 1, len(final_chunks))
            
            if not raw_chunk_concepts:
                # ⚠️ WARNING: Something unexpected but not fatal happened
                logger.warning(f"Chunk {i + 1} yielded 0 concepts! Skipping filter.")
                continue
                
            all_raw_concepts.extend(raw_chunk_concepts)
                
            filter_result = meta_filter_node(raw_chunk_concepts)
            
            final_clean_concepts.extend(filter_result.get("clean_concepts", []))
            all_deleted_concepts.extend(filter_result.get("deleted_concepts", []))

        logger.info(f"Phase 1 & 2 Complete. Extracted {len(all_raw_concepts)} raw -> Purged down to {len(final_clean_concepts)} clean.")

        # ==========================================
        # PHASE 3: SEMANTIC DEDUPLICATION (Vector Math)
        # ==========================================
        logger.info("Starting Phase 3: Semantic Deduplication...")
        deduplicated_concepts = semantic_deduplication(final_clean_concepts)
        
        # ==========================================
        # SAVE TO FILES 
        # ==========================================
        logger.debug("Writing output files to disk...")
        os.makedirs("data/output", exist_ok=True)
        
        with open("data/output/1_raw_concepts.json", "w", encoding="utf-8") as f:
            json.dump(all_raw_concepts, f, indent=4, ensure_ascii=False)
            
        with open("data/output/2_filtered_concepts.json", "w", encoding="utf-8") as f:
            json.dump(final_clean_concepts, f, indent=4, ensure_ascii=False)
            
        with open("data/output/deleted_concepts_log.json", "w", encoding="utf-8") as f:
            json.dump(all_deleted_concepts, f, indent=4, ensure_ascii=False)

        with open("data/output/final_concepts_output.json", "w", encoding="utf-8") as f:
            json.dump(deduplicated_concepts, f, indent=4, ensure_ascii=False)
        
        # ==========================================
        # FINAL LOGS
        # ==========================================
        # ✅ SUCCESS: The ultimate payout
        logger.success("SUCCESS! Pipeline Complete.")
        logger.success(f"Saved {len(all_raw_concepts)} Node 1 concepts -> 1_raw_concepts.json")
        logger.success(f"Saved {len(final_clean_concepts)} Node 2 concepts -> 2_filtered_concepts.json")
        logger.success(f"Saved {len(deduplicated_concepts)} Final concepts -> final_concepts_output.json")

    except Exception as e:
        # ❌ ERROR: Total failure state
        logger.error(f"PIPELINE CRASHED: {str(e)}")
        logger.error("Please check if Ollama is running and the PDF file exists.")

if __name__ == "__main__":
    run_pipeline("data/raw/hesc101.pdf")