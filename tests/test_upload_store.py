from src.storage.upload_store import SQLiteUploadStore


def make_document(
    document_id="upload_test123",
    text="Original document text.",
):
    return {
        "document_id": document_id,
        "title": "Example document",
        "text": text,
        "sentences": [text],
        "metadata": {
            "source_type": "uploaded_file",
            "file_format": "txt",
        },
        "provenance": [
            {"original_filename": "example.txt"}
        ],
    }


def test_save_and_reload_document(tmp_path):
    store = SQLiteUploadStore(tmp_path / "uploads.sqlite")
    document = make_document()

    saved_id = store.save_document(document)
    loaded = store.get_document(saved_id)

    assert saved_id == document["document_id"]
    assert loaded["text"] == document["text"]
    assert loaded["sentences"] == document["sentences"]
    assert loaded["metadata"]["file_format"] == "txt"
    assert loaded["provenance"] == document["provenance"]


def test_saved_documents_survive_store_reinitialization(tmp_path):
    db_path = tmp_path / "uploads.sqlite"

    SQLiteUploadStore(db_path).save_document(make_document())

    reopened_store = SQLiteUploadStore(db_path)

    assert reopened_store.count() == 1
    assert reopened_store.get_document("upload_test123") is not None


def test_saving_same_id_updates_existing_document(tmp_path):
    store = SQLiteUploadStore(tmp_path / "uploads.sqlite")

    store.save_document(make_document(text="Old version."))
    store.save_document(make_document(text="Updated version."))

    assert store.count() == 1
    assert (
        store.get_document("upload_test123")["text"]
        == "Updated version."
    )


def test_list_documents_returns_saved_document_metadata(tmp_path):
    store = SQLiteUploadStore(tmp_path / "uploads.sqlite")
    store.save_document(make_document())

    documents = store.list_documents()

    assert len(documents) == 1
    assert documents[0]["document_id"] == "upload_test123"
    assert documents[0]["title"] == "Example document"


def test_delete_document(tmp_path):
    store = SQLiteUploadStore(tmp_path / "uploads.sqlite")
    store.save_document(make_document())

    assert store.delete_document("upload_test123") is True
    assert store.get_document("upload_test123") is None
    assert store.count() == 0


def test_deleting_missing_document_returns_false(tmp_path):
    store = SQLiteUploadStore(tmp_path / "uploads.sqlite")

    assert store.delete_document("does-not-exist") is False