"""Unit tests for VectorStore mode (local vs server), client construction, and payload index error handling."""

from unittest.mock import MagicMock, patch
import pytest
from adaptive_retriever.config import Config
from adaptive_retriever.vectorstore import VectorStore


def test_non_default_vectorstore_mode_reaches_client_construction(tmp_path):
    # 1. Test mode="local" reaches QdrantClient(path=...)
    with patch("adaptive_retriever.vectorstore.QdrantClient") as mock_qdrant:
        mock_client_instance = MagicMock()
        mock_qdrant.return_value = mock_client_instance
        mock_client_instance.get_collections.return_value.collections = []

        cfg_local = Config({
            "vectorstore": {
                "mode": "local",
                "storage_path": str(tmp_path / "local_db"),
            }
        })
        vs_local = VectorStore.from_config(cfg_local)
        assert vs_local.mode == "local"
        mock_qdrant.assert_called_once()
        _, kwargs = mock_qdrant.call_args
        assert "path" in kwargs
        assert kwargs["path"] == str(tmp_path / "local_db")

    # 2. Test mode="server" reaches QdrantClient(url=..., api_key=...)
    with patch("adaptive_retriever.vectorstore.QdrantClient") as mock_qdrant:
        mock_client_instance = MagicMock()
        mock_qdrant.return_value = mock_client_instance
        mock_client_instance.get_collections.return_value.collections = []

        cfg_server = Config({
            "vectorstore": {
                "mode": "server",
                "url": "http://custom-qdrant-server:6333",
            }
        })
        vs_server = VectorStore.from_config(cfg_server, api_key="secret-key")
        assert vs_server.mode == "server"
        mock_qdrant.assert_called_once()
        _, kwargs = mock_qdrant.call_args
        assert kwargs.get("url") == "http://custom-qdrant-server:6333"
        assert kwargs.get("api_key") == "secret-key"


def test_create_payload_index_raises_on_failure_in_server_mode():
    with patch("adaptive_retriever.vectorstore.QdrantClient") as mock_qdrant:
        mock_client_instance = MagicMock()
        mock_qdrant.return_value = mock_client_instance
        mock_client_instance.get_collections.return_value.collections = []
        mock_client_instance.create_payload_index.side_effect = Exception("Not supported in local mode")

        # In server mode, create_payload_index MUST raise even if message mentions "not supported"
        with pytest.raises(Exception, match="Not supported in local mode"):
            VectorStore(
                config_hash="test1234",
                mode="server",
                url="http://localhost:6333",
            )


def test_server_connection_failure_raises_runtime_error_with_url():
    with patch("adaptive_retriever.vectorstore.QdrantClient") as mock_qdrant:
        mock_client_instance = MagicMock()
        mock_qdrant.return_value = mock_client_instance
        mock_client_instance.get_collections.side_effect = Exception("Connection refused")

        with pytest.raises(RuntimeError) as exc_info:
            VectorStore(
                config_hash="test1234",
                mode="server",
                url="http://localhost:6333",
            )
        assert "http://localhost:6333" in str(exc_info.value)
