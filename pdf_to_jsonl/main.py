import fitz  # PyMuPDF
import json
import argparse
from tqdm import tqdm


def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF using PyMuPDF"""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def split_into_chunks(text, max_tokens=300):
    """Split text into chunks based on paragraph breaks and token limit"""
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")  # GPT-like tokenizer

    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 30]
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        temp = current_chunk + "\n" + para
        if len(enc.encode(temp)) > max_tokens:
            chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            current_chunk = temp

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def convert_to_jsonl(chunks, output_path):
    """Convert text chunks to JSONL format for LLM fine-tuning"""
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            item = {
                "prompt": "### Instruction: Summarize or understand this text\n\n"
                + chunk
                + "\n\n### Response:",
                "completion": " <your expected output here>",
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main(pdf_path, output_path):
    print(f"Reading: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)

    print("Splitting into chunks...")
    chunks = split_into_chunks(text)

    print(f"Generating JSONL with {len(chunks)} chunks...")
    convert_to_jsonl(chunks, output_path)

    print(f"Done! JSONL written to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert PDF to JSONL for LLM fine-tuning"
    )
    parser.add_argument("pdf_path", help="Path to input PDF")
    parser.add_argument("output_path", help="Path to output JSONL file")
    args = parser.parse_args()

    main(args.pdf_path, args.output_path)
