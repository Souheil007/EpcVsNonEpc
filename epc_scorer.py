import argparse
import os
from dotenv import load_dotenv
from mistralai import Mistral
from utils import (
    perform_ocr,
    get_markdown_from_ocr,
    extract_headers_and_text,
    extract_unique_headers,
    compute_content_similarity_with_header_penalty,
    compute_structural_similarity,
    split_md
)

def main():
    # --- Argument parsing ---
    ap = argparse.ArgumentParser(description="Compare two documents using OCR, semantic, and structural similarity")
    ap.add_argument("--ref_file", required=True, help="Reference document file path")
    ap.add_argument("--cand_file", required=True, help="Candidate document file path to compare")
    ap.add_argument("--sem_weight", type=float, default=0.6, help="Weight for semantic similarity (default 0.6)")
    ap.add_argument("--struct_weight", type=float, default=0.4, help="Weight for structural similarity (default 0.4)")
    args = ap.parse_args()

    # --- Load API key ---
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("Missing MISTRAL_API_KEY in environment")

    client = Mistral(api_key=api_key)

    # --- OCR and processing for reference ---
    ocr_result1 = perform_ocr(client, args.ref_file)
    md1 = get_markdown_from_ocr(ocr_result1)
    md_header_splits1 = split_md(md1)
    extracted_chunks1 = extract_headers_and_text(md_header_splits1)
    unique_headers_list1 = extract_unique_headers(md_header_splits1)

    # --- OCR and processing for candidate ---
    ocr_result2 = perform_ocr(client, args.cand_file)
    md2 = get_markdown_from_ocr(ocr_result2)
    md_header_splits2 = split_md(md2)
    extracted_chunks2 = extract_headers_and_text(md_header_splits2)
    unique_headers_list2 = extract_unique_headers(md_header_splits2)

    # --- Semantic similarity ---
    sem_score = compute_content_similarity_with_header_penalty(extracted_chunks1, extracted_chunks2)
    print(f"Semantic similarity score: {sem_score:.4f}")

    # --- Structural similarity ---
    struct_score, sub_scores = compute_structural_similarity(
        ocr_result1.document_annotation,
        ocr_result2.document_annotation
    )
    print(f"Structural similarity score: {struct_score:.4f}")
    print("Structural breakdown:", sub_scores)

    # --- Overall weighted score ---
    overall_score = args.sem_weight * sem_score + args.struct_weight * struct_score
    print(f"Overall similarity score: {overall_score:.4f}")


if __name__ == "__main__":
    main()
