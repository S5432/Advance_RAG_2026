import re
import pytest
from src.chunking import chunk_text, longest_overlap

LONG = " ".join(f"Sentence number {i} explains an important policy detail about leave." for i in range(60))   # ~4000 chars

def test_long_text_is_split_with_size_limit():
    chunks = chunk_text(LONG, 800, 150)
    assert len(LONG) > 2000 and len(chunks) >= 4
    assert all(0 < len(c) <= 800 for c in chunks)

def test_short_text_single_chunk_and_empty_text():
    assert chunk_text("Hello world.", 800, 150) == ["Hello world."]
    assert chunk_text("", 800, 150) == [] and chunk_text("  \n  ", 800, 150) == []

def test_no_empty_chunks_and_all_content_is_covered():
    chunks = chunk_text(LONG, 500, 100)
    assert all(c.strip() for c in chunks)
    assert LONG.split()[0] in chunks[0] and LONG.split()[-1] in chunks[-1]
    covered = " ".join(chunks)
    assert all(w in covered for w in set(LONG.split()))

def test_chunks_do_not_break_words():
    words = set(LONG.split())
    for c in chunk_text(LONG, 300, 60):
        assert set(c.split()) <= words

def test_consecutive_chunks_overlap_about_requested_amount():
    chunks = chunk_text(LONG, 800, 150)
    for a, b in zip(chunks, chunks[1:]):
        ov = longest_overlap(a, b)
        assert 100 <= len(ov) <= 150, len(ov)

def test_zero_overlap_chunks_share_no_text():
    chunks = chunk_text(LONG, 500, 0)
    assert all(longest_overlap(a, b) == "" for a, b in zip(chunks, chunks[1:]))

def test_prefers_sentence_boundaries():
    chunks = chunk_text(LONG, 800, 150)
    assert all(c.rstrip().endswith(".") for c in chunks)

def test_progress_with_one_giant_word():
    chunks = chunk_text("x" * 2500, 800, 150)
    assert len(chunks) >= 3 and all(len(c) <= 800 for c in chunks)

@pytest.mark.parametrize("size,overlap", [(0, 0), (-5, 0), (100, 100), (100, 150), (100, -1)])
def test_invalid_parameters(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("some text here", size, overlap)

def test_spec_example_two_overlapping_chunks():
    text = "Employees receive 12 paid leaves annually. Sick leave requires manager approval."
    chunks = chunk_text(text, 50, 10)
    assert len(chunks) >= 2 and all(len(c) <= 50 for c in chunks)
    assert "Sick leave requires manager approval." in " ".join(chunks)
