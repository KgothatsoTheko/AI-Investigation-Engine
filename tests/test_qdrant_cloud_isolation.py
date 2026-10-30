def test_temporary_database_path_never_uses_qdrant_cloud(
    tmp_path,
    monkeypatch,
):
    """Cloud credentials must not redirect an isolated test DB to production."""
    from src.storage.vector_store import QdrantVectorStore

    monkeypatch.setenv("QDRANT_URL", "https://example.invalid")
    monkeypatch.setenv("QDRANT_API_KEY", "test-key")

    store = QdrantVectorStore(
        tmp_path / "isolated_vectors",
        collection_name="investigation_chunks",
    )

    try:
        assert store.remote_url == ""
        store.recreate_collection(vector_size=3)
        assert store.count_points() == 0
    finally:
        store.close()


def test_application_data_path_uses_remote_qdrant_when_configured(
    monkeypatch,
):
    """The app's standard data directory should still select Qdrant Cloud."""
    from src.storage.vector_store import QdrantVectorStore, PROJECT_DATA_DIR

    monkeypatch.setenv("QDRANT_URL", "https://example.invalid")
    monkeypatch.setenv("QDRANT_API_KEY", "test-key")

    store = QdrantVectorStore(
        PROJECT_DATA_DIR / "qdrant_full",
        collection_name="investigation_chunks",
    )

    try:
        assert store.remote_url == "https://example.invalid"
    finally:
        store.close()
