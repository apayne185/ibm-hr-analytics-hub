"""RAG retrieval over this project's own docs for the "Ask the Data" chat agent.

Chunking is format-aware, not generic: docs/sql_findings.md and
docs/survival_model_findings.md are already hand-curated one-insight-per-section
prose, so they split on their "## " headers -- the header boundaries *are* the
semantic boundaries, no reason to re-derive them algorithmically. DECISIONS.md has
its own distinct "## YYYY-MM-DD — Title" entry format, where each entry is a
self-contained "why did you do X" answer, so it gets its own chunking function.
docs/ph_assumptions_check.txt is small plain text, not markdown, and stays a
single chunk.

Retrieval is TF-IDF + in-memory cosine similarity (scikit-learn), not dense
embeddings or a vector database -- a deliberate choice for a corpus this size
(a handful of files, on the order of tens of chunks), not a placeholder. See
DECISIONS.md for the full reasoning, including what would change if the corpus
grew. No disk persistence: the fit is sub-second, so it's rebuilt at process
start rather than cached to disk, same spirit as survival_model.py.main() only
regenerating what's actually missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOCS_DIR = Path("docs")
DECISIONS_PATH = Path("DECISIONS.md")

MAX_CHUNK_CHARS = 800
CHUNK_OVERLAP_CHARS = 100


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    section: str
    text: str
    index: int


def _split_oversized(doc_id: str, section: str, text: str, start_index: int) -> list[Chunk]:
    """Fallback fixed-window split with overlap, only used when a section
    exceeds MAX_CHUNK_CHARS -- primary splitting is header-based/semantic, so
    overlap only matters here, to avoid severing a paragraph mid-thought."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [Chunk(doc_id=doc_id, section=section, text=text, index=start_index)]

    chunks = []
    step = MAX_CHUNK_CHARS - CHUNK_OVERLAP_CHARS
    for i, pos in enumerate(range(0, len(text), step)):
        window = text[pos : pos + MAX_CHUNK_CHARS]
        if not window.strip():
            continue
        chunks.append(Chunk(doc_id=doc_id, section=section, text=window.strip(), index=start_index + i))
    return chunks


def chunk_markdown(doc_id: str, text: str) -> list[Chunk]:
    """Split on '## ' headers. Content before the first header (if any) is
    dropped as boilerplate (title/intro), matching how sql_findings.md and
    survival_model_findings.md are actually structured."""
    # Prepending "\n" ensures a "## " at position 0 (a doc with no title/intro
    # before its first header) still splits correctly -- without this, a plain
    # text.split("\n## ") never matches a header at the very start of the
    # string, so a doc starting directly with "## " would end up as a single
    # index-0 "preamble" segment and get silently dropped entirely.
    sections = ("\n" + text).split("\n## ")
    chunks: list[Chunk] = []
    index = 0
    for i, raw_section in enumerate(sections):
        if i == 0:
            continue  # content before the first "## " header (empty if the doc starts with "## ")
        header, _, body = raw_section.partition("\n")
        section_title = header.strip()
        section_text = body.strip()
        if not section_text:
            continue
        for chunk in _split_oversized(doc_id, section_title, section_text, index):
            chunks.append(chunk)
            index += 1
    return chunks


def chunk_decisions_log(text: str) -> list[Chunk]:
    """DECISIONS.md has its own header format: '## YYYY-MM-DD — Title'. Each
    entry is a self-contained decision narrative, answering "why did you do X"
    questions directly."""
    return chunk_markdown("DECISIONS.md", text)


def chunk_plain_text(doc_id: str, text: str) -> list[Chunk]:
    """For non-markdown files too small to bother splitting further."""
    stripped = text.strip()
    if not stripped:
        return []
    return _split_oversized(doc_id, "(whole file)", stripped, 0)


@dataclass
class RetrievalIndex:
    chunks: list[Chunk]
    vectorizer: TfidfVectorizer
    matrix: object  # scipy sparse matrix


def build_index(docs_dir: Path = DOCS_DIR, decisions_path: Path = DECISIONS_PATH) -> RetrievalIndex:
    chunks: list[Chunk] = []

    for name in ("sql_findings.md", "survival_model_findings.md"):
        path = docs_dir / name
        if path.exists():
            chunks.extend(chunk_markdown(name, path.read_text()))

    ph_check = docs_dir / "ph_assumptions_check.txt"
    if ph_check.exists():
        chunks.extend(chunk_plain_text("ph_assumptions_check.txt", ph_check.read_text()))

    if decisions_path.exists():
        chunks.extend(chunk_decisions_log(decisions_path.read_text()))

    return build_index_from_chunks(chunks)


def build_index_from_chunks(chunks: list[Chunk]) -> RetrievalIndex:
    """Split out from build_index() so tests can build an index over a small
    known fixture corpus without touching the filesystem."""
    if not chunks:
        vectorizer = TfidfVectorizer()
        return RetrievalIndex(chunks=[], vectorizer=vectorizer, matrix=None)

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(c.text for c in chunks)
    return RetrievalIndex(chunks=chunks, vectorizer=vectorizer, matrix=matrix)


def retrieve(index: RetrievalIndex, query: str, k: int = 4, min_score: float = 0.05) -> list[tuple[Chunk, float]]:
    if not index.chunks:
        return []
    query_vec = index.vectorizer.transform([query])
    scores = cosine_similarity(query_vec, index.matrix)[0]
    ranked = sorted(zip(index.chunks, scores), key=lambda pair: pair[1], reverse=True)
    return [(chunk, float(score)) for chunk, score in ranked[:k] if score >= min_score]
