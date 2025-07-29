# adobe-hackathon-2025-submission
Team- A-dobyyy

Members- Aryan Shirbhate and MOHAMMAD FAISAL IQBAL

My Approach for Challenge 1A
My core approach for extracting the document structure is a heuristic-based parsing engine. I recognized that since PDFs are built for visual layout, relying on a single attribute like font size would be unreliable. Therefore, I built a multi-faceted system to make more intelligent decisions.

Baseline Analysis: First, my script performs a full-document analysis to find the most common font size, which I establish as the baseline for "body text." This is the most critical reference point for all other classifications.

Title Detection: I identify the title by looking for text on the first page that is both significantly larger than the body text and horizontally centered.

Heading Classification: I then evaluate every line of the PDF against a combination of rules:

Font Size & Weight: I check if the text is larger than the body text and if the font name indicates a bold weight (e.g., contains "Bold", "Black").

Text Patterns: I use regular expressions to look for common heading markers like "1. Introduction" or "A.1".

Noise Filtering: To increase accuracy, I added filters to ignore text that is unlikely to be a heading, such as very short lines or lines that don't start with a capital letter.

By combining these rules, my solution can robustly handle a wide variety of document formats.

My Approach for Challenge 1B
For this challenge, I used a semantic search approach to understand the meaning of the text, which is far more powerful than simple keyword matching. My solution is built on sentence embeddings.

Query Formulation: I start by combining the persona.json and job.txt files into a single, detailed query. This gives the AI model a rich, contextual understanding of the user's specific task and goals.

Document Parsing & Chunking: I parse each PDF using PyMuPDF and break the content down into smaller, paragraph-sized chunks. This allows for a more granular and accurate analysis of relevance.

Embedding Generation: I use the pre-trained all-MiniLM-L6-v2 transformer model to convert both the detailed query and every single text chunk into numerical vectors (embeddings). This model was chosen for its excellent balance of performance and size, fitting within the hackathon's offline and CPU-only constraints.

Similarity Search & Ranking: I then calculate the Cosine Similarity between the query's embedding and every chunk's embedding. This gives a score of how closely they match in meaning. The chunks are then ranked from highest to lowest score.

Output Generation: Finally, I format the top-ranked results into the required results.json file, providing both a high-level summary of the most important pages and a detailed list of the most relevant text snippets.
