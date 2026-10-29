"""
DriveLegal – tests/test_rag.py
Unit tests for the RAG retriever and chain.

Run:  pytest tests/test_rag.py -v
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain.schema import Document


# ── Helpers ────────────────────────────────────────────────────────────────

def make_doc(text: str, region: str) -> Document:
    return Document(page_content=text, metadata={"region": region, "law_section": "Section 129"})


def _make_scored_point(doc_id: int, text: str, region: str):
    """Build a fake ScoredPoint matching qdrant_client's structure."""
    point = MagicMock()
    point.id = doc_id
    point.payload = {
        "text": text,
        "law_section": "Section 129",
        "category": "traffic",
        "region": region,
    }
    return point


# ── Retriever tests ────────────────────────────────────────────────────────

@patch("rag.retriever._client")
@patch("rag.retriever._embedder")
def test_retrieve_returns_docs(mock_embedder, mock_client):
    """retrieve() returns Document objects from Qdrant results."""
    import numpy as np
    from rag.retriever import retrieve

    mock_embedder.encode.return_value = np.zeros(384, dtype="float32")

    response = MagicMock()
    response.points = [
        _make_scored_point(1, "Helmet rule text", "Pune"),
        _make_scored_point(2, "National helmet law", "National"),
    ]
    mock_client.query_points.return_value = response

    docs, _ = retrieve("helmet fine", k=5, city="Pune", state="Maharashtra")

    assert len(docs) >= 1
    assert all(isinstance(d, Document) for d in docs)


@patch("rag.retriever._client")
@patch("rag.retriever._embedder")
def test_retrieve_deduplicates_results(mock_embedder, mock_client):
    """retrieve() does not return duplicate document IDs across cascade scopes."""
    import numpy as np
    from rag.retriever import retrieve

    mock_embedder.encode.return_value = np.zeros(384, dtype="float32")

    # Same point returned at both city and state scope
    shared_point = _make_scored_point(1, "Shared rule", "Pune")
    response = MagicMock()
    response.points = [shared_point]
    mock_client.query_points.return_value = response

    docs, _ = retrieve("fine", k=5, city="Pune", state="Maharashtra")
    ids = [r.page_content for r in docs]
    assert len(ids) == len(set(ids)), "Duplicate documents returned"


@patch("rag.retriever._client")
@patch("rag.retriever._embedder")
def test_retrieve_respects_k_limit(mock_embedder, mock_client):
    """retrieve() returns at most k results."""
    import numpy as np
    from rag.retriever import retrieve

    mock_embedder.encode.return_value = np.zeros(384, dtype="float32")

    response = MagicMock()
    response.points = [_make_scored_point(i, f"Rule {i}", "National") for i in range(20)]
    mock_client.query_points.return_value = response

    docs, _ = retrieve("speed limit", k=3)
    assert len(docs) <= 3


@patch("rag.retriever._build_region_filter")
@patch("rag.retriever._client")
@patch("rag.retriever._embedder")
def test_retrieve_uses_full_jurisdiction_fallback_order(mock_embedder, mock_client, mock_build_filter):
    import numpy as np
    from rag.retriever import retrieve

    mock_embedder.encode.return_value = np.zeros(384, dtype="float32")
    response = MagicMock()
    response.points = []
    mock_client.query_points.return_value = response
    mock_build_filter.side_effect = lambda region: region

    retrieve("fine", k=1, city="Pune", state="Maharashtra", country="India")

    expected_scopes = [
        "Pune",
        "Maharashtra",
        "India",
        "National",
        "Vienna Convention on Road Traffic",
        "UN Model Road Safety Legislation",
        "",
    ]
    actual_scopes = [call.args[0] for call in mock_build_filter.call_args_list]
    assert actual_scopes == expected_scopes


def test_retrieve_raises_without_loaded_index():
    """retrieve() raises RuntimeError if load_index() was never called."""
    import rag.retriever as r

    original_client, original_embedder = r._client, r._embedder
    r._client = None
    r._embedder = None

    with pytest.raises(RuntimeError, match="Qdrant index not loaded"):
        r.retrieve("test query")

    r._client = original_client
    r._embedder = original_embedder


# ── Chain tests ────────────────────────────────────────────────────────────

@patch("rag.chain.retrieve")
@patch("rag.chain.ChatOllama")
def test_get_answer_success(mock_llm_cls, mock_retrieve):
    """get_answer() returns answer dict with expected keys on happy path."""
    from rag.chain import get_answer

    mock_retrieve.return_value = ([make_doc("Red light fine is ₹1,000", "Pune")], 0.8)
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="The fine for jumping a red light is ₹1,000.")
    mock_llm_cls.return_value = mock_llm

    result = get_answer("red light fine", city="Pune", state="Maharashtra", country="India")

    assert "answer" in result
    assert "sources" in result
    assert result["offline_fallback"] is False
    assert len(result["answer"]) > 5


@patch("rag.chain.retrieve")
@patch("rag.chain.ChatOllama")
def test_get_answer_with_gps_context(mock_llm_cls, mock_retrieve):
    """get_answer() includes gps_context in the prompt without crashing."""
    from rag.chain import get_answer

    mock_retrieve.return_value = ([make_doc("Speed limit rule", "National")], 0.8)
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Speed limit is 50 km/h in cities.")
    mock_llm_cls.return_value = mock_llm

    result = get_answer(
        "speed limit",
        city="Pune", state="Maharashtra", country="India",
        gps_context="Pune, Maharashtra, India",
    )

    assert "answer" in result
    assert result["offline_fallback"] is False
    # Verify the prompt passed to LLM contained the GPS context
    call_args = mock_llm.invoke.call_args[0][0]
    assert "Pune, Maharashtra, India" in call_args


@patch("rag.chain.retrieve")
def test_get_answer_empty_docs_returns_not_found(mock_retrieve):
    """get_answer() returns a not-found message when retrieval returns nothing."""
    from rag.chain import get_answer

    mock_retrieve.return_value = ([], 0.0)

    result = get_answer("unknown query xyz", city="", state="", country="India")

    assert "answer" in result
    assert "could not find" in result["answer"].lower()


@patch("rag.chain.retrieve")
@patch("rag.chain.ChatOllama")
def test_get_answer_llm_error_uses_offline_cache(mock_llm_cls, mock_retrieve):
    """When Ollama/Mistral is unavailable, get_answer() returns offline_fallback=True."""
    from rag.chain import get_answer

    mock_retrieve.return_value = ([make_doc("some text", "National")], 0.0)
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("Ollama timeout")
    mock_llm_cls.return_value = mock_llm

    result = get_answer("helmet fine", city="", state="", country="India")

    assert "answer" in result
    assert result["offline_fallback"] is True
