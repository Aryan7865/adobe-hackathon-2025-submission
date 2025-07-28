import fitz  # PyMuPDF
import json
import os
import re
import statistics
from collections import defaultdict

def analyze_font_styles(doc):
    """
    Analyzes the document to find the most common font size.
    This helps establish a baseline for what is likely 'body' text vs. a 'heading'.
    """
    font_sizes = defaultdict(int)
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict").get("blocks", [])
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span["size"] > 7: # Ignore very small text
                        size = round(span["size"])
                        font_sizes[size] += len(span["text"])
    
    if not font_sizes:
        return 12 # Return a default if no text is found

    body_font_size = max(font_sizes, key=font_sizes.get)
    return body_font_size

def is_bold(span):
    """Checks if the font name suggests it is bold."""
    font_name = span["font"].lower()
    return "bold" in font_name or "black" in font_name or "heavy" in font_name

def classify_heading(text, span, body_font_size):
    """
    Classifies a line of text as a heading level (H1, H2, H3) or None based on refined heuristics.
    """
    size = round(span["size"])
    text = text.strip()
    words = text.split()

    # --- Final Stricter Filter for Noise ---
    # 1. Ignore empty or very short text
    if len(text) < 3:
        return None
    # 2. Ignore text that is clearly part of a sentence or paragraph
    if text.endswith((',', ';', '—', '.', '–', ':')) or (text.islower() and len(words) > 4):
        return None
    # 3. Ignore lines starting with special characters or are just a link
    if text.startswith(('[[', '(', 'http')) or text.endswith(')') or '.git' in text:
        return None
    # 4. Ignore lines that don't start with a capital letter or a number
    if not (text[0].isupper() or text[0].isdigit()):
        return None
    # 5. Ignore lines with more than 8 words, as headings are typically concise
    if len(words) > 8:
        return None
    # 6. Ignore lines that are likely list items or table entries (e.g., "No internet access")
    if len(words) in [2, 3] and " ".join(words[1:]).islower():
        return None


    is_heading_font = size > body_font_size
    is_bold_font = is_bold(span)

    if not (is_heading_font or is_bold_font):
        return None

    is_list_marker = re.match(r'^\s*(\d+(\.\d+)*|[A-Z]|[a-z]|[IVXLCDM]+)\.?\s+', text)
    
    size_ratio = size / body_font_size
    
    # Final classification logic
    if size_ratio > 1.8 and is_bold_font:
        return "H1"
    elif size_ratio > 1.35 or (is_bold_font and size_ratio > 1.15):
        return "H2"
    elif is_heading_font or is_bold_font or is_list_marker:
        return "H3"
        
    return None

def extract_structure_from_pdf(pdf_path):
    """
    Main logic to extract the structured outline (Title, H1, H2, H3) from a PDF.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening {pdf_path}: {e}")
        return None

    if len(doc) == 0:
        return {"title": "", "outline": []}

    body_font_size = analyze_font_styles(doc)
    outline = []
    title = ""
    
    # --- Final Title Detection ---
    first_page = doc[0]
    
    # Find the block with the largest average font size on the first page.
    max_avg_size = 0
    title_block = None
    blocks = first_page.get_text("dict").get("blocks", [])
    for block in blocks:
        if block['type'] == 0: # Text block
            # Get all non-empty text spans to calculate average size
            spans = [span for line in block['lines'] for span in line['spans'] if span['text'].strip()]
            if not spans: continue
            
            avg_size = statistics.mean([s['size'] for s in spans])
            
            # A title should be significantly larger than body text
            if avg_size > body_font_size * 1.5 and avg_size > max_avg_size:
                max_avg_size = avg_size
                title_block = block

    if title_block:
        title_lines = []
        for line in title_block['lines']:
            line_text = " ".join([span['text'] for span in line['spans']]).strip()
            if line_text:
                title_lines.append(line_text)
        title = " ".join(title_lines)
    
    if not title or len(title) < 5:
        title = os.path.basename(pdf_path) # Fallback

    # --- Heading Detection ---
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict").get("blocks", [])
        
        for block in blocks:
            for line in block.get("lines", []):
                if line["spans"]:
                    first_span = line["spans"][0]
                    line_text = " ".join([s["text"] for s in line["spans"]]).strip()

                    # Skip if it's the title or a part of it
                    if title and line_text in title:
                        continue
                    
                    level = classify_heading(line_text, first_span, body_font_size)
                    if level:
                        # Avoid adding duplicates
                        if not any(d['text'] == line_text for d in outline):
                            outline.append({
                                "level": level,
                                "text": line_text,
                                "page": page_num + 1
                            })
                            
    return {
        "title": title,
        "outline": outline
    }

def main():
    """
    Orchestrates the processing of all PDFs in the input directory
    and writes the JSON output to the output directory.
    """
    input_dir = "/app/input"
    output_dir = "/app/output"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            print(f"Processing {pdf_path}...")
            
            structure = extract_structure_from_pdf(pdf_path)
            
            if structure:
                output_filename = os.path.splitext(filename)[0] + ".json"
                output_path = os.path.join(output_dir, output_filename)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(structure, f, indent=4, ensure_ascii=False)
                
                print(f"Successfully created {output_path}")

if __name__ == "__main__":
    main()
