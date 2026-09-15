"""Markdown-aware chunking.

Naive fixed-size chunking cuts sentences and code blocks in half and loses the
heading that says what the text is about. This splits on heading boundaries,
keeps fenced code blocks intact, and stamps every chunk with its heading path
so the retrieved fragment still knows where it came from.
"""
import re
from dataclasses import dataclass

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
FENCE = re.compile(r"^\s*```")

MAX_CHARS = 1800
MIN_CHARS = 220
OVERLAP_CHARS = 200


@dataclass
class Chunk:
    ordinal: int
    heading_path: str
    content: str

    @property
    def n_chars(self) -> int:
        return len(self.content)

    def embedding_text(self) -> str:
        """What we actually embed.

        The heading path goes in too: a chunk that reads 'Use AddScoped' is
        ambiguous on its own but unambiguous under 'Dependency injection >
        Service lifetimes'.
        """
        return f"{self.heading_path}\n\n{self.content}".strip()


def _sections(markdown: str) -> list[tuple[str, str]]:
    """Split into (heading_path, body), respecting fenced code blocks."""
    sections: list[tuple[str, str]] = []
    stack: list[str] = []
    body: list[str] = []
    in_fence = False

    def flush() -> None:
        text = "\n".join(body).strip()
        if text:
            sections.append((" > ".join(p for p in stack if p), text))
        body.clear()

    for line in markdown.splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
            body.append(line)
            continue
        m = None if in_fence else HEADING.match(line)
        if m:
            flush()
            level, title = len(m.group(1)), m.group(2).strip()
            del stack[level - 1 :]
            while len(stack) < level - 1:
                stack.append("")
            stack.append(title)
        else:
            body.append(line)
    flush()
    return sections


def _tail(text: str) -> str:
    """Last OVERLAP_CHARS of text, snapped forward to a word boundary.

    Slicing blindly produces chunks that open mid-word ("e UI" instead of
    "composite UI"), which is noise in the embedding and looks broken when the
    passage is shown to a user as a citation.
    """
    tail = text[-OVERLAP_CHARS:]
    space = tail.find(" ")
    return tail[space + 1 :] if 0 <= space < 40 else tail


def _hard_split(text: str) -> list[str]:
    """Last-resort split for a block with no paragraph breaks.

    Big markdown tables and long fenced code blocks are single paragraphs by
    definition, so paragraph splitting leaves them intact and they blow past
    MAX_CHARS. Falling back to line boundaries, then to a raw character cut,
    guarantees every chunk fits the context budget.
    """
    if len(text) <= MAX_CHARS:
        return [text]

    parts: list[str] = []
    buf = ""
    for line in text.splitlines(keepends=True):
        if len(buf) + len(line) > MAX_CHARS and buf:
            parts.append(buf.strip())
            buf = _tail(buf) + line
        else:
            buf += line
    if buf.strip():
        parts.append(buf.strip())

    # A single line longer than MAX_CHARS (minified content, a one-line table).
    out: list[str] = []
    for part in parts:
        while len(part) > MAX_CHARS:
            out.append(part[:MAX_CHARS])
            part = part[MAX_CHARS - OVERLAP_CHARS :]
        if part.strip():
            out.append(part)
    return out


def _split_long(text: str) -> list[str]:
    """Break an oversized section on paragraph boundaries, with overlap.

    Overlap matters: a fact stated at the end of one chunk and referenced at the
    start of the next survives in at least one retrievable piece.
    """
    if len(text) <= MAX_CHARS:
        return [text]

    parts: list[str] = []
    buf = ""
    for para in text.split("\n\n"):
        candidate = f"{buf}\n\n{para}".strip() if buf else para
        if len(candidate) > MAX_CHARS and buf:
            parts.append(buf)
            buf = f"{_tail(buf)}\n\n{para}".strip()
        else:
            buf = candidate
    if buf:
        parts.append(buf)

    # Paragraph splitting cannot help a paragraph that is itself too big.
    return [piece for part in parts for piece in _hard_split(part)]


def chunk_markdown(markdown: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for heading_path, body in _sections(markdown):
        for piece in _split_long(body):
            if len(piece.strip()) < MIN_CHARS:
                continue
            chunks.append(Chunk(len(chunks), heading_path, piece.strip()))
    return chunks
