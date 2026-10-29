import importlib
import inspect


MODULES = [
    "src.ingestion.chunking",
    "src.embeddings.encoder",
    "src.storage.vector_store",
    "src.retrieval.sqlite_bm25",
    "src.retrieval.hybrid_search",
    "src.rag.retriever",
    "src.storage.upload_store",
]


def inspect_module(module_name: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"MODULE: {module_name}")
    print("=" * 72)

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        print(f"Import failed: {type(exc).__name__}: {exc}")
        return

    for name, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ != module.__name__:
            continue

        try:
            class_signature = inspect.signature(obj)
        except (TypeError, ValueError):
            class_signature = "(signature unavailable)"

        print(f"\nCLASS: {name}{class_signature}")

        for method_name, method in inspect.getmembers(obj, inspect.isfunction):
            if method_name.startswith("__") and method_name != "__init__":
                continue

            try:
                method_signature = inspect.signature(method)
            except (TypeError, ValueError):
                method_signature = "(signature unavailable)"

            print(f"  {method_name}{method_signature}")

    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if obj.__module__ != module.__name__:
            continue

        try:
            function_signature = inspect.signature(obj)
        except (TypeError, ValueError):
            function_signature = "(signature unavailable)"

        print(f"\nFUNCTION: {name}{function_signature}")


if __name__ == "__main__":
    for module_name in MODULES:
        inspect_module(module_name)