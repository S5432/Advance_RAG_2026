import re

def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()

def _find_cut(text: str, start: int, end: int, min_pos: int) -> int:
    """Best place to end a chunk inside [min_pos, end]: paragraph > sentence end > line > space. Falls back to `end`."""
    seg = text[min_pos:end]
    i = seg.rfind("\n\n")
    if i != -1: return min_pos + i + 2
    m = None
    for m in re.finditer(r"[.!?]\s", seg): pass
    if m: return min_pos + m.end()
    i = seg.rfind("\n")
    if i != -1: return min_pos + i + 1
    i = seg.rfind(" ")
    if i != -1: return min_pos + i + 1
    return end                                           # one huge word: hard cut

def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> list[str]:
    """Split text into overlapping chunks of at most `chunk_size` characters.
    Cuts prefer paragraph/sentence/word boundaries (no mid-word cuts unless a single word exceeds chunk_size) and each
    chunk starts about `chunk_overlap` characters before the previous one ended, so context survives the boundary."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
    text = _normalize(text or "")
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            end = _find_cut(text, start, end, start + int(chunk_size * 0.5))
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        nxt = max(end - chunk_overlap, start + 1)
        while 0 < nxt < end and not text[nxt - 1].isspace():     # move forward to the start of a word
            nxt += 1
        start = nxt if nxt < end else end
    return chunks

def longest_overlap(a: str, b: str) -> str:
    """Longest suffix of `a` that is a prefix of `b` (the shared context between consecutive chunks)."""
    for k in range(min(len(a), len(b)), 0, -1):
        if a[-k:] == b[:k]:
            return a[-k:]
    return ""

if __name__ == "__main__":
    sample = " ".join(f"Policy clause {i} explains an important rule about leave, refunds or passwords." for i in range(40))
    parts = chunk_text(sample, 800, 150)
    print("text length:", len(sample), "| chunks:", len(parts))
    for i, c in enumerate(parts):
        ov = longest_overlap(parts[i - 1], c) if i else ""
        print(f"[{i}] len={len(c)} first100={c[:100]!r} overlap_with_prev={len(ov)} chars")
