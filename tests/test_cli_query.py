"""Unit tests for cli query command."""

import json
from unittest.mock import MagicMock, patch
from typer.testing import CliRunner

from adaptive_retriever.cli import app
from adaptive_retriever.models import RetrievalCandidate

runner = CliRunner()


@patch("adaptive_retriever.vectorstore.VectorStore.from_config")
def test_cli_query_json_output(mock_from_config):
    mock_vs = MagicMock()
    mock_candidate = RetrievalCandidate(
        chunk_id="chunk_1",
        rep_id="rep_1",
        fused_score=0.12345,
        final_rank=1,
        text="Sample text content for retrieval testing.",
        metadata={
            "kind": "summary",
            "doc_class": "digital_prose",
            "doc_id": "doc_100",
            "source_ranks": {"dense": 1, "sparse": 2},
        },
    )
    mock_vs.query_hybrid.return_value = [mock_candidate]
    mock_from_config.return_value = mock_vs

    result = runner.invoke(app, ["query", "what is retrieval", "--json"])
    assert result.exit_code == 0

    parsed = json.loads(result.stdout)
    assert parsed["query"] == "what is retrieval"
    assert len(parsed["results"]) == 1
    res = parsed["results"][0]
    assert res["rank"] == 1
    assert res["fused_score"] == 0.1235
    assert res["kind"] == "summary"
    assert res["doc_class"] == "digital_prose"
    assert res["doc_id"] == "doc_100"
    assert res["source_ranks"] == {"dense": 1, "sparse": 2}


@patch("adaptive_retriever.vectorstore.VectorStore.from_config")
def test_cli_query_passes_options_and_default_limit(mock_from_config):
    mock_vs = MagicMock()
    mock_vs.query_hybrid.return_value = []
    mock_from_config.return_value = mock_vs

    result = runner.invoke(
        app,
        [
            "query",
            "test query",
            "--doc-class",
            "digital_prose",
            "--query-class",
            "exact_lookup",
            "--limit",
            "3",
        ],
    )
    assert result.exit_code == 0
    assert "Querying: test query" in result.stdout

    mock_vs.query_hybrid.assert_called_once()
    kwargs = mock_vs.query_hybrid.call_args.kwargs
    assert kwargs["query_text"] == "test query"
    assert kwargs["limit"] == 3
    assert kwargs["doc_class_filter"] == "digital_prose"
    assert kwargs["query_class"] == "exact_lookup"
