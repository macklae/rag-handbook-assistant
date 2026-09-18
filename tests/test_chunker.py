from app.core.chunker import chunk_pages, chunk_text
from app.core.pdf_reader import Page


def test_chunk_text_respects_size_and_overlap():
    text = "abcdefghij" * 10  # 100 chars
    chunks = chunk_text(text, chunk_size=30, overlap=5)
    assert all(len(c) <= 30 for c in chunks)
    assert len(chunks) == 4


def test_chunk_text_rejects_bad_overlap():
    for bad in (-1, 30, 40):
        try:
            chunk_text("abc", chunk_size=30, overlap=bad)
        except ValueError:
            continue
        raise AssertionError(f"overlap={bad} should have raised")


def test_chunk_pages_carries_page_numbers():
    pages = [Page(number=1, text="a" * 120), Page(number=7, text="b" * 60)]
    chunks = chunk_pages(pages, chunk_size=50, overlap=10)
    assert {c.page for c in chunks} == {1, 7}
    assert [c.id for c in chunks] == [f"chunk-{i}" for i in range(len(chunks))]


def test_empty_pages_produce_no_chunks():
    assert chunk_pages([], chunk_size=50, overlap=10) == []
