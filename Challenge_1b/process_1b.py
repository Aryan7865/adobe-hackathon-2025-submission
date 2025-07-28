import fitz  # PyMuPDF
import json
import os
import re
from sentence_transformers import SentenceTransformer, util
import torch
import datetime

# --- Configuration ---
# Point to the local model directory copied into our Docker container
MODEL_PATH = '/app/models/all-MiniLM-L6-v2'
INPUT_DIR = '/app/input'
OUTPUT_DIR = '/app/output'
DOCS_DIR = os.path.join(INPUT_DIR, 'docs')

def load_model():
    """Loads the SentenceTransformer model from the local path."""
    print("Loading model from local path...")
    # Ensure the model runs on CPU as per constraints
    device = 'cpu'
    model = SentenceTransformer(MODEL_PATH, device=device)
    print("Model loaded successfully.")
    return model

def parse_pdf_into_chunks(pdf_path):
    """
    Parses a PDF, extracts text, and splits it into meaningful chunks (paragraphs).
    Each chunk is stored with its source document and page number.
    """
    chunks = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            # Split page text into paragraphs (blocks in PyMuPDF)
            blocks = doc[page_num].get_text("blocks")
            for i, block in enumerate(blocks):
                # block[4] contains the text of the block
                text = block[4].strip()
                # Clean up text: remove excessive newlines and whitespace
                text = re.sub(r'\s+', ' ', text)
                if len(text) > 50: # Filter out very short, likely irrelevant blocks
                    chunks.append({
                        "doc_name": os.path.basename(pdf_path),
                        "page_num": page_num + 1,
                        "chunk_id": f"{os.path.basename(pdf_path)}_p{page_num + 1}_b{i}",
                        "text": text
                    })
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
    return chunks

def create_query_from_inputs():
    """
    Reads persona and job files and combines them into a single, rich query string.
    """
    persona_text = ""
    job_text = ""
    persona_data = {}
    try:
        with open(os.path.join(INPUT_DIR, 'persona.json'), 'r', encoding='utf-8') as f:
            persona_data = json.load(f)
        persona_text = f"As a {persona_data.get('role')} with expertise in {persona_data.get('expertise')}, my goals are to {', '.join(persona_data.get('goals', []))}."
    except Exception as e:
        print(f"Warning: Could not read persona.json. {e}")
        persona_text = "No persona description provided."

    try:
        with open(os.path.join(INPUT_DIR, 'job.txt'), 'r', encoding='utf-8') as f:
            job_text = f.read().strip()
    except Exception as e:
        print(f"Warning: Could not read job.txt. {e}")
        job_text = "No job description provided."

    query = f"User profile: {persona_text}\n\nTask: {job_text}"
    print(f"Generated Query:\n{query}")
    return query, persona_data, job_text

def main():
    """
    Main execution function for the persona-driven document intelligence task.
    """
    start_time = datetime.datetime.now()
    
    # 1. Load Model and create query
    model = load_model()
    query, persona_json, job_str = create_query_from_inputs()

    # 2. Process all documents
    all_chunks = []
    if not os.path.exists(DOCS_DIR):
        print(f"Error: Input directory for documents not found at {DOCS_DIR}")
        return
        
    doc_files = [f for f in os.listdir(DOCS_DIR) if f.lower().endswith('.pdf')]
    print(f"Found {len(doc_files)} PDF(s) to process.")
    for doc_name in doc_files:
        pdf_path = os.path.join(DOCS_DIR, doc_name)
        print(f"Parsing {doc_name}...")
        all_chunks.extend(parse_pdf_into_chunks(pdf_path))

    if not all_chunks:
        print("No text chunks could be extracted from the documents. Exiting.")
        return

    print(f"Extracted a total of {len(all_chunks)} text chunks.")

    # 3. Generate Embeddings
    print("Generating embeddings for all text chunks...")
    corpus_embeddings = model.encode([chunk['text'] for chunk in all_chunks], convert_to_tensor=True, show_progress_bar=True)
    
    print("Generating embedding for the query...")
    query_embedding = model.encode(query, convert_to_tensor=True)

    # 4. Perform Semantic Search
    print("Performing semantic search...")
    # Use cosine similarity to find the top N most similar chunks
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=50)
    hits = hits[0]  # Get the results for the first (and only) query

    # 5. Structure the output
    extracted_sections = []
    sub_section_analysis = []
    
    # Use a set to track sections we've already added to avoid duplicates
    added_sections = set()
    
    rank = 1
    for hit in hits:
        chunk_index = hit['corpus_id']
        chunk = all_chunks[chunk_index]
        
        # For sub-section analysis, we add the top relevant chunks directly
        if len(sub_section_analysis) < 20: # Limit to top 20 granular results
             sub_section_analysis.append({
                "document": chunk['doc_name'],
                "page_number": chunk['page_num'],
                "refined_text": chunk['text'],
                "importance_rank": len(sub_section_analysis) + 1,
                "score": round(hit['score'], 4)
            })

        # For extracted sections, we want to show the parent section (approximated by page)
        section_key = (chunk['doc_name'], chunk['page_num'])
        if section_key not in added_sections and len(extracted_sections) < 10: # Limit to top 10 unique sections
            extracted_sections.append({
                "document": chunk['doc_name'],
                "page_number": chunk['page_num'],
                "section_title": f"Content from page {chunk['page_num']}", # A more advanced solution would use the 1A output to find the actual heading
                "importance_rank": rank
            })
            added_sections.add(section_key)
            rank += 1

    # 6. Create the final JSON output
    end_time = datetime.datetime.now()
    
    output_data = {
        "metadata": {
            "input_documents": doc_files,
            "persona": persona_json,
            "job_to_be_done": job_str,
            "processing_timestamp": end_time.isoformat()
        },
        "extracted_sections": extracted_sections,
        "sub_section_analysis": sub_section_analysis
    }

    output_path = os.path.join(OUTPUT_DIR, 'results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"\nProcessing complete. Results saved to {output_path}")
    print(f"Total processing time: {round((end_time - start_time).total_seconds(), 2)} seconds.")

if __name__ == "__main__":
    main()

