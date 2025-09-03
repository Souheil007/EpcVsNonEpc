
# Document Similarity Comparison Tool


This project allows you to **compare two documents (PDFs or images)** using OCR and compute both semantic and structural similarity scores. It leverages the **Mistral AI OCR API** for document text extraction, **transformers** for semantic embeddings, and **Pydantic models** for structured annotation.

## Approach

* In order to access our pdfs/images we have to acced to their content and i proceeded that by using ocr exactly mistral ocr because it excels at understanding complex layouts and free to use,
* After that we obtain a markdown file which gives us an insight about the two files structure and we havee to compare them , 
* when comparing i thought about chunking both documents using heading/tables and retrieving the chunks embedding and calculating smeentic similarity score of them within the same part of chunks
* and by that we obtain a sementic similarity score and also we have to calculate a structural score by comparing headings, number of tables , row names and header names

## Drawbacks

* mistral ocr eventhought it exceed in its field of ocring complex documents may fail in retrieving the layout of some documents
* by calculating sementic similarity score, some fields which are the filling of that epc may not reveal the actual layout and may give a bad score
* eventhough we are calculating structural score we are heavily relying on mistral output

## vision

* docling is a promising library that to get in depth of it in order to see what is capable of 
* we can try using vlms and see their results since gemini 2.5 pro is one of the leading vlms now but it is pay to use it 
* We use an llm and ask some set of questions to it and compare responses of both files in order to get the score
* We can use llm as a judge but it is a really weak solution

## Features

* Perform OCR on PDF or image documents.
* Extract structured information like chapter titles, table headers, and row names.
* Compute **semantic similarity** using sentence embeddings with local context and header-aware penalties.
* Compute **structural similarity** based on document annotation (chapter titles, tables, headers, rows).
* Calculate an overall weighted similarity score.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Souheil007/EpcVsNonEpc.git
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   source venv/Scripts/activate    # Windows

   pip install -r requirements.txt
   ```

3. Set up your environment variables (Mistral API key):

   ```bash
   echo MISTRAL_API_KEY=your_api_key_here > .env
   ```

## Usage

Run the main script with the reference and candidate files:

```bash
python epc_scorer.py --ref_file path/to/reference.pdf --cand_file path/to/candidate.pdf
```

Optional arguments:

* `--sem_weight`: Weight for semantic similarity in the overall score (default 0.6)
* `--struct_weight`: Weight for structural similarity in the overall score (default 0.4)

Example:

```bash
python epc_scorer.py --ref_file docs/reference.pdf --cand_file docs/candidate.pdf --sem_weight 0.7 --struct_weight 0.3
```

## Output

The script prints:

* **Semantic similarity score**: measures textual similarity of content.
* **Structural similarity score**: measures overlap of chapter titles, tables, headers, and row labels.
* **Structural breakdown**: detailed score per structural component.
* **Overall similarity score**: weighted combination of semantic and structural scores.

Example output:

```
Semantic similarity score: 0.82
Structural similarity score: 0.91
Structural breakdown: {'chapters': 1.0, 'tables': 0.8, 'headers': 0.9, 'rows': 0.85}
Overall similarity score: 0.86
```

## Requirements

* Python >= 3.9
* PyTorch
* Transformers
* scikit-learn
* langchain\_text\_splitters
* Mistral AI SDK
* python-dotenv

Install dependencies using:

```bash
pip install torch transformers scikit-learn langchain_text_splitters mistralai python-dotenv
```

## Notes

* The tool relies on the **Mistral AI OCR API**, so an active API key is required.
* Documents should preferably be in PDF or high-quality image formats.
* The semantic similarity uses a **sentence-transformers model** for embeddings.

## License

MIT License
