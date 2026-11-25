import pytest

from src.chunking import chunk_documents, split_text


def _doc(text, doc_id="doc_1", **meta):
    return {"text": text, "metadata": {"source": "a.pdf", "source_type": "pdf",
                                       "page_number": 1, "document_id": doc_id, **meta}}


def test_metadata_ids_and_extra_metadata_preserved():
    text = " ".join(f"Sentence number {i} is here." for i in range(80))
    chunks = chunk_documents([_doc(text)], 300, 60, extra_metadata={"tenant_id": "t1"})
    assert len(chunks) > 1
    for i, c in enumerate(chunks):
        m = c["metadata"]
        assert m["chunk_index"] == i
        assert m["chunk_id"] == f"doc_1_chunk_{i}"
        assert m["document_id"] == "doc_1" and m["page_number"] == 1 and m["tenant_id"] == "t1"


def test_chunk_size_respected_and_sentences_not_cut():
    text = " ".join(f"This is sentence {i}." for i in range(100))
    chunks = split_text(text, 200, 40)
    assert all(len(c) <= 200 for c in chunks)
    assert all(c.rstrip().endswith(".") for c in chunks)


def test_overlap_repeats_text_between_chunks():
    text = " ".join(f"Sentence {i} ends here." for i in range(60))
    chunks = split_text(text, 200, 60)
    assert chunks[0].split(". ")[-1] in chunks[1] or chunks[0][-20:] in chunks[1]


def test_long_sentence_falls_back_to_word_windows():
    chunks = split_text("word " * 500, 100, 20)
    assert all(len(c) <= 100 for c in chunks)


def test_chunk_index_continues_across_pages():
    docs = [_doc("Page one text. " * 5, page_number=1), _doc("Page two text. " * 5, page_number=2)]
    chunks = chunk_documents(docs, 800, 150)
    assert [c["metadata"]["chunk_index"] for c in chunks] == [0, 1]
    assert chunks[1]["metadata"]["page_number"] == 2


def test_section_from_loader_and_heading_detection():
    md = chunk_documents([_doc("Body text.", section="Leave Policy")])
    assert md[0]["metadata"]["section"] == "Leave Policy"
    pdf = chunk_documents([_doc("Leave Policy\nEmployees get 18 days of leave.")])
    assert pdf[0]["metadata"]["section"] == "Leave Policy"


def test_empty_text_skipped_and_bad_args_rejected():
    assert chunk_documents([_doc("   ")]) == []
    with pytest.raises(ValueError):
        split_text("abc", 100, 100)
    with pytest.raises(ValueError):
        chunk_documents([{"text": "x", "metadata": {}}])
