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


# ------------------------------------------------------------------ Day 11 / Phase 2
from src.chunking import (choose_best_chunks, choose_strategy, count_tokens, fixed_size_chunking,
                          heading_aware_chunking, parent_child_chunking, recursive_chunking, split_sections)

POLICY = ("Leave Policy\nEmployees are entitled to 18 days of paid annual leave per year.\n\n"
          "Sick Leave\nEmployees get 10 days of paid sick leave. A doctor note is needed after 3 days.\n\n"
          "Remote Work\nEmployees may work remotely up to 2 days per week with manager approval.")
MD = "# Leave Policy\nEighteen days of annual leave.\n\n## Sick Leave\nTen days of sick leave.\n"


def test_fixed_size_window_and_overlap():
    text = "a" * 2000
    chunks = fixed_size_chunking(text, 800, 150)
    assert all(len(c) <= 800 for c in chunks) and len(chunks) == 4  # windows start at 0, 650, 1300, 1950
    with pytest.raises(ValueError):
        fixed_size_chunking("abc", 100, 100)


def test_recursive_keeps_code_block_whole():
    code = "```python\n" + "\n".join(f"x{i} = {i}" for i in range(30)) + "\n```"
    chunks = recursive_chunking("Intro text here. " * 40 + "\n\n" + code + "\n\n" + "Outro. " * 40, 400, 50)
    assert sum(1 for c in chunks if "x0 = 0" in c and "x29 = 29" in c) == 1


def test_heading_aware_keeps_sections_together():
    chunks = heading_aware_chunking(POLICY, 1000)
    assert len(chunks) == 3
    assert chunks[0].startswith("Leave Policy") and "18 days" in chunks[0]
    assert chunks[1].startswith("Sick Leave") and "18 days" not in chunks[1]


def test_heading_aware_markdown_and_big_section_repeats_title():
    assert [t for t, _ in split_sections(MD)] == ["Leave Policy", "Sick Leave"]
    big = "# Big\n" + " ".join(f"Sentence number {i} is here." for i in range(120))
    chunks = heading_aware_chunking(big, 300)
    assert len(chunks) > 1 and all(c.startswith("Big\n") and len(c) <= 300 for c in chunks)


def test_headings_inside_code_fence_are_ignored():
    assert [t for t, _ in split_sections("# Real\ntext\n```\n# not a heading\n```\n")] == ["Real"]


def test_parent_child_ids_links_and_metadata():
    long = "Leave Policy\n" + " ".join(f"Rule {i} says employees must follow policy item {i}." for i in range(60))
    docs = [_doc(long, tenant_id="t1")]
    out = parent_child_chunking(docs, parent_size=500, child_size=200, child_overlap=30)
    parents = [c for c in out if c["metadata"]["chunk_type"] == "parent"]
    kids = [c for c in out if c["metadata"]["chunk_type"] == "child"]
    assert parents and len(kids) > len(parents)
    ids = [c["metadata"]["chunk_id"] for c in out]
    assert len(ids) == len(set(ids))
    assert [k["metadata"]["chunk_index"] for k in kids] == list(range(len(kids)))
    pids = {p["metadata"]["chunk_id"] for p in parents}
    assert pids == {f"doc_1_parent_{i}" for i in range(len(parents))}
    for k in kids:
        m = k["metadata"]
        assert m["parent_chunk_id"] in pids and m["tenant_id"] == "t1" and m["page_number"] == 1
        assert m["token_count"] > 0 and len(k["text"]) <= 200 + 1


def test_choose_best_chunks_drops_noise_and_duplicates():
    mk = lambda t: {"text": t, "metadata": {"document_id": "d", "chunk_type": "standalone"}}
    best = choose_best_chunks([mk("Hello world."), mk("  hello   WORLD. "), mk("-----"), mk(""), mk("Other text")])
    assert [c["text"] for c in best] == ["Hello world.", "Other text"]


def test_strategy_selection_rules():
    assert choose_strategy({"source_type": "pdf"}, "plain text only. " * 20) == "recursive"
    assert choose_strategy({"source_type": "markdown"}, "x") == "heading"
    assert choose_strategy({"source_type": "html"}, "x") == "heading"
    assert choose_strategy({"source_type": "pdf"}, POLICY) == "heading"      # structured PDF
    assert choose_strategy({"source_type": "markdown"}, "x", "fixed") == "fixed"


def test_chunk_documents_enriched_metadata_and_strategy():
    chunks = chunk_documents([_doc(POLICY)], 800, 150, extra_metadata={"tenant_id": "t1"})
    assert [c["metadata"]["section"] for c in chunks] == ["Leave Policy", "Sick Leave", "Remote Work"]
    m = chunks[0]["metadata"]
    assert m["chunking_strategy"] == "heading" and m["chunk_type"] == "standalone"
    assert m["language"] == "en" and m["access_level"] == "internal" and m["token_count"] == count_tokens(chunks[0]["text"])
    rec = chunk_documents([_doc("Plain sentence here. " * 100)], 300, 50)
    assert rec[0]["metadata"]["chunking_strategy"] == "recursive"


def test_chunk_documents_parent_child_strategy():
    long = "Leave Policy\n" + " ".join(f"Rule {i} says employees must follow policy item {i}." for i in range(80))
    out = chunk_documents([_doc(long)], 800, 150, strategy="parent_child")
    assert {c["metadata"]["chunk_type"] for c in out} == {"parent", "child"}
    assert all(c["metadata"]["access_level"] == "internal" for c in out)
