import importlib
import inspect
from pathlib import Path


def print_source(label, obj):
    print(f"\n{'=' * 76}")
    print(label)
    print("=" * 76)

    try:
        print(inspect.getsource(obj))
    except (OSError, TypeError) as exc:
        print(f"Source unavailable: {exc}")


def inspect_module(module_name):
    print(f"\nMODULE: {module_name}")

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        print(f"Import failed: {type(exc).__name__}: {exc}")
        return

    for _, cls in inspect.getmembers(module, inspect.isclass):
        if cls.__module__ != module.__name__:
            continue

        try:
            signature = inspect.signature(cls)
        except (TypeError, ValueError):
            signature = "(signature unavailable)"

        print_source(f"CLASS: {cls.__name__}{signature}", cls)

        for name, method in inspect.getmembers(cls, inspect.isfunction):
            if method.__module__ != module.__name__:
                continue

            if name.startswith("_") and name != "__init__":
                continue

            print_source(f"METHOD: {cls.__name__}.{name}", method)

    for name, function in inspect.getmembers(module, inspect.isfunction):
        if function.__module__ != module.__name__:
            continue

        if name.startswith("_"):
            continue

        print_source(f"FUNCTION: {name}", function)


if __name__ == "__main__":
    app_path = Path("app.py")

    print_source(
        "CURRENT STREAMLIT APP",
        compile(app_path.read_text(encoding="utf-8"), str(app_path), "exec"),
    ) if False else print(app_path.read_text(encoding="utf-8"))

    for module_name in [
        "src.rag.investigator",
        "src.rag.generator",
        "src.rag.retriever",
    ]:
        inspect_module(module_name)