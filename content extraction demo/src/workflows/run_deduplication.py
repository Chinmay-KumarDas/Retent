import json
import os
from src.utils.logger import get_logger
from src.nodes.deduplication_node import semantic_deduplication

logger = get_logger(__name__)

def run_deduplication_only():
    input_file = "data/output/2_filtered_concepts.json"
    output_file = "data/output/final_concepts_output.json"
    
    # ⚙️ EXECUTING: The script is spinning up
    logger.debug("Starting Standalone Deduplication Test...")
    
    if not os.path.exists(input_file):
        # ❌ ERROR: Fatal failure
        logger.error(f"Could not find {input_file}. Make sure you renamed the file correctly!")
        return
        
    # ⚙️ EXECUTING: Granular mechanical step
    logger.debug(f"Loading JSON data from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        final_clean_concepts = json.load(f)
        
    # ℹ️ INFO: A major phase is beginning
    logger.info(f"Initiating Semantic Deduplication on {len(final_clean_concepts)} concepts...")
    deduplicated_concepts = semantic_deduplication(final_clean_concepts)
    
    # ⚙️ EXECUTING: Saving to disk
    logger.debug("Writing consolidated master concepts to disk...")
    os.makedirs("data/output", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(deduplicated_concepts, f, indent=4, ensure_ascii=False)
        
    # ==========================================
    # ✅ FINAL LOGS (SUCCESS)
    # ==========================================
    logger.success("Standalone Deduplication Complete.")
    logger.success(f"Processed {len(final_clean_concepts)} items down to {len(deduplicated_concepts)} master concepts.")
    logger.success(f"Saved to -> {output_file}")

if __name__ == "__main__":
    run_deduplication_only()