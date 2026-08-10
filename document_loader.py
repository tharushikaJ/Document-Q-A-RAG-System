"""
Loads raw text out of common document formats: .txt, .md, .pdf, .docx
Each loader returns a single string of extracted text.
"""
from pathlib import Path


def load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    text_parts = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            text_parts.append(f"[Page {page_num + 1}]\n{text}")
    return "\n\n".join(text_parts)


def load_docx(path: Path) -> str:
    import docx
    doc = docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


LOADERS = {
    ".txt": load_txt,
    ".md": load_txt,
    ".pdf": load_pdf,
    ".docx": load_docx,
}


def load_document(path: Path) -> str:
    """Dispatch to the right loader based on file extension."""
    suffix = path.suffix.lower()
    if suffix not in LOADERS:
        raise ValueError(
            f"Unsupported file type: {suffix}. Supported: {list(LOADERS.keys())}"
        )
    return LOADERS[suffix](path)


def load_documents_from_dir(dir_path: Path) -> list[dict]:
    """
    Load every supported file in a directory (non-recursive).
    Returns a list of {"source": filename, "text": content} dicts.
    """
    docs = []
    for path in sorted(dir_path.iterdir()):
        if path.is_file() and path.suffix.lower() in LOADERS:
            try:
                text = load_document(path)
                if text.strip():
                    docs.append({"source": path.name, "text": text})
            except Exception as e:
                print(f"  [!] Failed to load {path.name}: {e}")
    return docs
