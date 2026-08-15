# Build Dataset with GPT

Turn any PDF into an **Alpaca-style instruction-tuning dataset** with GPT.

This project reads a book or document, splits it into overlapping chunks, sends each chunk to GPT, and writes a clean JSON file of `{instruction, input, output}` examples. The result is ready for supervised fine-tuning of instruction-following models.

---

## What you get

| Input | Process | Output |
| --- | --- | --- |
| `data/book.pdf` | Load → chunk → generate with GPT | `output/alpaca.json` |

Each source chunk produces **5 instruction / response pairs**, grounded in that passage instead of generic trivia.

- **PDF ingestion** with LangChain `PyPDFLoader`
- **Recursive chunking** (1000 characters, 200 overlap) so context is not lost at page boundaries
- **GPT-4o-mini** via the [Metis](https://api.metisai.ir) OpenAI-compatible API
- **Alpaca schema** used by Stanford Alpaca and most open instruction-tuning pipelines
- **Robust JSON cleanup** so markdown fences from the model do not break parsing
- **Progress bar** with `tqdm` for long documents

---

## Pipeline

```text
PDF pages
    │
    ▼
LangChain loader
    │
    ▼
RecursiveCharacterTextSplitter
  chunk_size=1000  overlap=200
    │
    ▼
GPT-4o-mini  (5 Alpaca examples / chunk)
    │
    ▼
JSON parse + merge
    │
    ▼
output/alpaca.json
```

---

## Dataset format

Every record follows the Alpaca instruction format:

```json
{
  "instruction": "Explain the main claim of this passage.",
  "input": "",
  "output": "A grounded answer derived from the source chunk."
}
```

| Field | Role |
| --- | --- |
| `instruction` | The task the model should perform |
| `input` | Extra context (often empty) |
| `output` | The target completion |

This matches the layout used by [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) and is compatible with common SFT loaders (Hugging Face `datasets`, Axolotl, LLaMA-Factory, Unsloth, and similar tools).

---

## Quick start

### 1. Clone

```bash
git clone https://github.com/EbiAraz/Build-Dataset-with-GPT.git
cd Build-Dataset-with-GPT
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the API key

Copy the example env file and add your Metis key:

```bash
copy .env.example .env   # Windows
cp .env.example .env     # macOS / Linux
```

```env
METIS_API_KEY=your_metis_api_key_here
```

The client talks to `https://api.metisai.ir/openai/v1`. Do not commit `.env`.

### 5. Add a PDF

Place your source document at:

```text
data/book.pdf
```

### 6. Generate the dataset

```bash
python generate_dataset.py
```

When it finishes you will see page count, chunk count, and dataset size. The file is written to:

```text
output/alpaca.json
```

---

## Configuration

Edit `generate_dataset.py` to match your document and budget:

| Setting | Default | Notes |
| --- | --- | --- |
| `pdf_path` | `data/book.pdf` | Source document |
| `chunk_size` | `1000` | Larger chunks = more context, higher token cost |
| `chunk_overlap` | `200` | Keeps sentences from being cut in half |
| `model` | `gpt-4o-mini` | Swap for another Metis-supported chat model |
| `temperature` | `0.3` | Lower = more faithful to the source text |
| examples per chunk | `5` | Change the prompt if you want more or fewer |

A 200-page book at these settings typically yields on the order of **thousands of instruction pairs**. Cost scales with page count × chunks × 5 examples.

---

## Project layout

```text
.
├── generate_dataset.py   # PDF → chunks → GPT → Alpaca JSON
├── requirements.txt
├── .env.example          # METIS_API_KEY placeholder
├── data/
│   └── book.pdf          # you provide this
└── output/
    └── alpaca.json       # generated dataset
```

---

## How generation works

1. **Load** every page of the PDF as LangChain documents.
2. **Split** into overlapping character chunks so each prompt stays inside a reasonable context window.
3. **Prompt** GPT as a dataset generator: return **only JSON**, no markdown, no commentary.
4. **Parse** the reply, strip accidental ` ```json ` fences, and `json.loads` it.
5. **Append** successful batches to one list. Failed chunks are logged and skipped so a single bad reply does not abort the run.
6. **Dump** the full list as UTF-8 JSON with `ensure_ascii=False`, so non-English text (Persian, Arabic, etc.) is preserved.

---

## Requirements

- Python 3.10+
- A Metis API key with access to `gpt-4o-mini` (or another chat model you set in the script)
- A readable PDF at `data/book.pdf`

Python packages:

```text
langchain
langchain-community
langchain-text-splitters
pypdf
openai
python-dotenv
tqdm
```

---

## Tips for a stronger dataset

- Prefer a **clean digital PDF** over a scanned image-only file (this pipeline does not OCR).
- Keep `temperature` low if you want answers tightly bound to the book.
- After generation, skim `output/alpaca.json` and drop empty, duplicated, or off-topic rows before training.
- For domain-specific fine-tuning, use a single coherent source (one textbook, one manual) rather than a mixed dump of unrelated PDFs.
- If a chunk fails JSON parsing, the script continues; re-run or lower temperature if too many chunks are skipped.

---

## License

Use this tooling for your own documents. Do not publish copyrighted books or generated datasets derived from them unless you have the right to do so.
