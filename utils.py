from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv
from mistralai import Mistral
from mistralai.extra import response_format_from_pydantic_model
from pydantic import BaseModel
import json
from langchain_text_splitters import MarkdownHeaderTextSplitter
# Document Annotation response format
from pydantic import BaseModel, Field
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch


def perform_ocr(client, file_path: str, model: str = "mistral-ocr-latest"):
    """
    Uploads an image to the API, gets a signed URL, and performs OCR using the specified model.
    
    Args:
        client: Initialized API client.
        file_path: Local path to the image file.
        model: OCR model to use (default: "mistral-ocr-latest").
    
    Returns:
        OCR response object.
    """
    # Document Annotation response format
    class Document(BaseModel):
        language: str = Field(
            description="The primary language detected in the document, e.g., 'en' for English or 'de' for German."
        )
        chapter_titles: list[str] = Field(
            description="List of major section or chapter titles extracted from the document, preserving their original order."
        )
        tables: list[str] 
        number_of_tables: int = Field(
            description="The total number of tables present in the document."
        )
        tables_header_names: list[str] = Field(
            description="List of header names (column titles) found across all tables in the document."
        )
        tables_row_names: list[str] = Field(
            description="List of row labels or identifiers (first-column values) found across all tables in the document."
        )

    # Upload the image
    uploaded_file = client.files.upload(
        file={
            "file_name": file_path.split("/")[-1],
            "content": open(file_path, "rb")
        },
        purpose="ocr"
    )

    # Get signed URL
    file_signed_url = client.files.get_signed_url(file_id=uploaded_file.id)
    file_url = file_signed_url.url

    # Perform OCR
    response = client.ocr.process(
        model=model,
        document={
            "type": "image_url",
            "image_url": file_url
        },
        document_annotation_format=response_format_from_pydantic_model(Document),
        include_image_base64=True
    )

    return response


def get_markdown_from_ocr(response):
    # Convert response to dictionary if needed
    response_dict = response.model_dump()
    response_dict.get("pages")
    md = response_dict.get("pages")[0].get("markdown")
    return md


headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
def split_md(md):
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, return_each_line=True)
    md_header_splits = markdown_splitter.split_text(md)
    return md_header_splits
# list of dicts {"header": tuple, "text": str} 
def extract_headers_and_text(docs):
    """
    Converts Document list into a list of dicts:
    {'header': (Header1, Header2, ...), 'text': '...'}
    """
    chunks = []
    for doc in docs:
        header = tuple(doc.metadata.get(k) for k in sorted(doc.metadata.keys()))
        chunks.append({"header": header, "text": doc.page_content})
    return chunks

def extract_unique_headers(md_splits):
    """
    Extract unique (level, header_text) tuples from MarkdownHeaderTextSplitter output,
    preserving the original order.
    """
    seen = set()
    unique_headers = []
    for doc in md_splits:
        for key, value in doc.metadata.items():
            if value:
                level = int(key.split(" ")[1])
                header_tuple = (level, value)
                if header_tuple not in seen:
                    seen.add(header_tuple)
                    unique_headers.append(header_tuple)
    return unique_headers



tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def encode_texts(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings

def compute_content_similarity_with_header_penalty(ref_chunks, cand_chunks, window=4, header_penalty=1):
    """
    Compute a similarity score between two lists of chunks, preserving local structure
    and penalizing header mismatches.
    
    ref_chunks: list of dicts {"header": tuple, "text": str} for reference document
    cand_chunks: same format for candidate document
    window: number of neighboring chunks on either side to consider
    header_penalty: multiplier for similarity if headers do not match (0 < header_penalty <= 1)
    """
    ref_texts = [c["text"] for c in ref_chunks]
    cand_texts = [c["text"] for c in cand_chunks]

    ref_emb = encode_texts(ref_texts)
    cand_emb = encode_texts(cand_texts)

    sims = []
    for i, r_emb in enumerate(ref_emb):
        # define neighborhood in candidate document
        start = max(0, i - window)
        end = min(len(cand_emb), i + window + 1)
        local_cand_emb = cand_emb[start:end]

        # compute similarity with local neighborhood
        sim_scores = cosine_similarity([r_emb], local_cand_emb)[0]

        # find the best match in neighborhood
        best_idx = sim_scores.argmax()
        best_score = sim_scores[best_idx]

        # apply header penalty if headers do not match
        ref_header = ref_chunks[i]["header"]
        cand_header = cand_chunks[start + best_idx]["header"]
        if ref_header != cand_header:
            best_score *= header_penalty  # penalize mismatch

        sims.append(best_score)

    return float(np.mean(sims))


def compute_header_coverage(ref_headers, cand_headers):
    """
    Computes a score based on how many reference headers exist in the candidate.
    Returns a fraction [0,1].
    """
    ref_set = set(ref_headers)
    cand_set = set(cand_headers)
    covered = ref_set & cand_set
    return len(covered) / len(ref_set) if ref_set else 1.0


def compute_structural_similarity(doc_ann1, doc_ann2, weights=None):
    # Convert Pydantic models or JSON strings to dicts if needed
    if isinstance(doc_ann1, str):
        doc_ann1 = json.loads(doc_ann1)
    elif not isinstance(doc_ann1, dict):
        doc_ann1 = doc_ann1.dict()

    if isinstance(doc_ann2, str):
        doc_ann2 = json.loads(doc_ann2)
    elif not isinstance(doc_ann2, dict):
        doc_ann2 = doc_ann2.dict()
    
    if weights is None:
        weights = {"chapters": 0.3, "tables": 0.2, "headers": 0.3, "rows": 0.2}
    
    scores = {}
    
    # --- Chapter titles overlap ---
    set1, set2 = set(doc_ann1.get("chapter_titles", [])), set(doc_ann2.get("chapter_titles", []))
    if set1 or set2:
        scores["chapters"] = len(set1 & set2) / len(set1 | set2)
    else:
        scores["chapters"] = 1.0
    
    # --- Number of tables similarity ---
    n1, n2 = doc_ann1.get("number_of_tables", 0), doc_ann2.get("number_of_tables", 0)
    if max(n1, n2) > 0:
        scores["tables"] = 1 - abs(n1 - n2) / max(n1, n2)
    else:
        scores["tables"] = 1.0
    
    # --- Table header names overlap ---
    headers1, headers2 = set(doc_ann1.get("tables_header_names", [])), set(doc_ann2.get("tables_header_names", []))
    if headers1 or headers2:
        scores["headers"] = len(headers1 & headers2) / len(headers1 | headers2)
    else:
        scores["headers"] = 1.0
    
    # --- Table row names overlap ---
    rows1, rows2 = set(doc_ann1.get("tables_row_names", [])), set(doc_ann2.get("tables_row_names", []))
    if rows1 or rows2:
        scores["rows"] = len(rows1 & rows2) / len(rows1 | rows2)
    else:
        scores["rows"] = 1.0
    
    # --- Weighted combination ---
    structural_score = sum(scores[k] * weights[k] for k in weights)
    
    return structural_score, scores