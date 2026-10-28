
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class TextEncoder:
    """Generate vectors from documents and search queries."""

    def __init__(self, model_name: str = MODEL_NAME):
        print(f"Loading embedding model: {model_name}")

        self.model = SentenceTransformer(model_name)

        dimension = self.model.get_embedding_dimension()

        if dimension is None:
            raise RuntimeError(
                "The embedding model did not report its dimension."
            )

        self.dimension = int(dimension)

        print(f"Embedding dimension: {self.dimension}")

    def encode_documents(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ):
        """Convert multiple document chunks into vectors."""
        if not texts:
            return []

        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar,
        )

    def encode_query(self, query: str):
        """Convert one user query into a vector."""
        if not query.strip():
            raise ValueError("The search query cannot be empty.")

        vectors = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return vectors[0]
