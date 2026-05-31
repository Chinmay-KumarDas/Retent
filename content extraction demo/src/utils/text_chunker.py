import fitz  # PyMuPDF
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.logger import get_logger

logger = get_logger(__name__)

def get_chunks_from_pdf(pdf_path: str) -> list:
    logger.info(f"Loading PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    full_text = ""
    
    for page in doc:
        full_text += page.get_text()

    # LANDMINE AVOIDANCE
    cutoff_phrases = ["WHAT YOU HAVE LEARNT", "EXERCISES"]
    for phrase in cutoff_phrases:
        if phrase in full_text:
            logger.info(f"Found cutoff phrase '{phrase}'. Truncating document.")
            full_text = full_text.split(phrase)[0]
            break
        
    # HYBRID CHUNKING
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
    return final_chunks