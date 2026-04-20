"""
Skeleton Similarity Calculator

Implements similarity metrics for both question and SQL skeletons:
- Question skeletons: Cosine similarity on embeddings
- SQL skeletons: Edit distance on parse trees

Usage:
    from schema_linking.skeleton_similarity import SkeletonSimilarity

    sim = SkeletonSimilarity()

    # Question skeleton similarity (cosine)
    q_sim = sim.question_similarity(
        "How many [ENTITY] have [ATTRIBUTE] greater than [VALUE]?",
        "What is the average [ATTRIBUTE] of [ENTITY]?",
    )

    # SQL skeleton similarity (edit distance)
    sql_sim = sim.sql_similarity(
        "SELECT COUNT([COLUMN]) FROM [TABLE] WHERE [COLUMN] > [VALUE]",
        "SELECT AVG([COLUMN]) FROM [TABLE] WHERE [COLUMN] = [VALUE]",
    )
"""

from typing import List, Optional, Tuple, Dict
import numpy as np

from .sql_skeleton_extractor import (
    sql_skeleton_edit_distance,
    sql_skeleton_similarity,
)


class SkeletonSimilarity:
    """
    Calculates similarity between skeleton pairs using appropriate metrics:
    - Cosine similarity for question skeletons (via embeddings)
    - Edit distance for SQL skeletons (via parse tree tokens)
    """

    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the similarity calculator.

        Args:
            embedding_model: HuggingFace sentence transformer model for question embeddings
        """
        self.embedding_model_name = embedding_model
        self._embedding_model = None
        self._tokenizer = None

    def _load_embedding_model(self):
        """Lazy load the sentence transformer model."""
        if self._embedding_model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for question similarity. "
                "Install with: pip install sentence-transformers"
            )

        self._embedding_model = SentenceTransformer(self.embedding_model_name)
        # Use CPU for embeddings to avoid GPU memory conflicts with vLLM
        self._embedding_model = self._embedding_model.to("cpu")

    def question_similarity(self, skeleton1: str, skeleton2: str) -> float:
        """
        Calculate cosine similarity between two question skeletons.

        Args:
            skeleton1: First question skeleton (Q⋆)
            skeleton2: Second question skeleton (Q⋆)

        Returns:
            Cosine similarity in range [0, 1]
        """
        self._load_embedding_model()

        # Get embeddings
        embeddings = self._embedding_model.encode([skeleton1, skeleton2])

        # Calculate cosine similarity
        vec1 = embeddings[0]
        vec2 = embeddings[1]

        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

        # Clip to handle floating point errors
        return float(np.clip(similarity, 0.0, 1.0))

    def question_similarity_batch(
        self,
        query_skeleton: str,
        candidate_skeletons: List[str],
        batch_size: int = 32,
    ) -> List[float]:
        """
        Calculate cosine similarity between a query skeleton and multiple candidates.

        Args:
            query_skeleton: The query question skeleton (Q⋆)
            candidate_skeletons: List of candidate question skeletons
            batch_size: Batch size for encoding

        Returns:
            List of similarity scores
        """
        self._load_embedding_model()

        # Encode all skeletons
        all_skeletons = [query_skeleton] + candidate_skeletons
        embeddings = self._embedding_model.encode(
            all_skeletons,
            batch_size=batch_size,
            show_progress_bar=False,
        )

        query_embedding = embeddings[0]
        candidate_embeddings = embeddings[1:]

        # Calculate cosine similarities
        similarities = []
        for cand_emb in candidate_embeddings:
            sim = np.dot(query_embedding, cand_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(cand_emb)
            )
            similarities.append(float(np.clip(sim, 0.0, 1.0)))

        return similarities

    def sql_similarity(self, skeleton1: str, skeleton2: str) -> float:
        """
        Calculate similarity between two SQL skeletons using edit distance.

        Args:
            skeleton1: First SQL skeleton (S⋆)
            skeleton2: Second SQL skeleton (S⋆)

        Returns:
            Similarity score in range [0, 1] where 1 means identical
        """
        return sql_skeleton_similarity(skeleton1, skeleton2)

    def sql_similarity_batch(
        self,
        query_skeleton: str,
        candidate_skeletons: List[str],
    ) -> List[float]:
        """
        Calculate similarity between a query SQL skeleton and multiple candidates.

        Args:
            query_skeleton: The query SQL skeleton (S⋆)
            candidate_skeletons: List of candidate SQL skeletons

        Returns:
            List of similarity scores
        """
        similarities = []
        for cand_skeleton in candidate_skeletons:
            sim = self.sql_similarity(query_skeleton, cand_skeleton)
            similarities.append(sim)
        return similarities
