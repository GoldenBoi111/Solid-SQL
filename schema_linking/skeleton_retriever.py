"""
Skeleton-Based Example Retriever with FAISS Indexing

Retrieves the top-N most similar examples from a candidate library based on:
1. Question skeleton similarity (cosine similarity using FAISS)
2. SQL skeleton similarity (edit distance using FAISS)

Supports both retrieval modes:
- Question-based: Find examples with similar question structure
- SQL-based: Find examples with similar SQL structure (for round-2 refinement)

Usage:
    from schema_linking.skeleton_retriever import SkeletonRetriever

    # Initialize retriever
    retriever = SkeletonRetriever(candidate_examples=[
        {"question": "How many singers are older than 20?", "sql": "SELECT COUNT(*) FROM Singer WHERE Age > 20"},
        {"question": "What is the average salary?", "sql": "SELECT AVG(salary) FROM Employees"},
    ])

    # Extract skeletons for all examples and build FAISS index
    retriever.build_index()

    # Retrieve top-N similar examples for a new question
    results = retriever.retrieve_by_question(
        question="How many actors are younger than 30?",
        top_n=2,
    )

    # Retrieve top-N similar examples for a generated SQL
    results = retriever.retrieve_by_sql(
        sql="SELECT COUNT(*) FROM Actor WHERE age < 30",
        top_n=2,
    )
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not installed. Install with: pip install faiss-cpu")

from .question_skeleton_extractor import QuestionSkeletonExtractor
from .sql_skeleton_extractor import SQLSkeletonExtractor
from .skeleton_similarity import SkeletonSimilarity


class SkeletonRetriever:
    """
    Retrieves similar examples from a candidate library using skeleton-based similarity.
    
    Uses FAISS for efficient vector similarity search.
    
    Supports two retrieval modes:
    1. Question skeleton-based: Matches question structure (Q⋆)
    2. SQL skeleton-based: Matches SQL structure (S⋆)
    """

    def __init__(
        self,
        candidate_examples: List[Dict[str, str]],
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        sql_dialect: str = "sqlite",
        faiss_index_type: str = "flat",
    ):
        """
        Initialize the retriever.
        
        Args:
            candidate_examples: List of dicts with "question" and "sql" keys
            embedding_model: Sentence transformer model for question embeddings
            sql_dialect: SQL dialect for parsing
            faiss_index_type: FAISS index type ("flat", "ivf", "hnsw")
        """
        self.candidates = candidate_examples
        self.question_skeletons = []
        self.sql_skeletons = []
        self.question_embeddings = None
        self.sql_skeletons_list = []
        
        # FAISS indexes
        self.question_index = None
        self.sql_index = None
        
        # Initialize extractors and similarity calculator
        self.q_extractor = QuestionSkeletonExtractor()
        self.sql_extractor = SQLSkeletonExtractor(dialect=sql_dialect)
        self.similarity = SkeletonSimilarity(embedding_model=embedding_model)
        
        self.faiss_index_type = faiss_index_type

    def build_index(
        self,
        batch_size: int = 16,
        show_progress: bool = True,
        faiss_dim: int = 384,
    ) -> None:
        """
        Pre-compute skeletons for all candidate examples and build FAISS indexes.
        
        This should be called once during initialization to build the
        example index for fast retrieval.
        
        Args:
            batch_size: Batch size for skeleton extraction
            show_progress: Whether to show progress
            faiss_dim: Dimension for FAISS index (384 for MiniLM)
        """
        if not self.candidates:
            print("Warning: No candidate examples provided")
            return

        questions = [ex["question"] for ex in self.candidates]
        sqls = [ex["sql"] for ex in self.candidates]

        if show_progress:
            print(f"Extracting question skeletons for {len(questions)} examples...")
        self.question_skeletons = self.q_extractor.extract_batch(
            questions,
            batch_size=batch_size,
            show_progress=show_progress,
        )

        if show_progress:
            print(f"Extracting SQL skeletons for {len(sqls)} examples...")
        self.sql_skeletons = self.sql_extractor.extract_batch(
            sqls,
            show_progress=show_progress,
        )
        self.sql_skeletons_list = list(self.sql_skeletons)

        if show_progress:
            print(f"Building FAISS indexes...")
        
        # Build FAISS index for question skeletons
        if FAISS_AVAILABLE:
            self._build_faiss_index_for_questions(faiss_dim)
            self._build_faiss_index_for_sql(faiss_dim)
        else:
            # Fallback to in-memory similarity if FAISS not available
            if show_progress:
                print("  Using in-memory similarity (FAISS not available)")

        if show_progress:
            print(f"Index built with {len(self.question_skeletons)} question skeletons "
                  f"and {len(self.sql_skeletons)} SQL skeletons")

    def _build_faiss_index_for_questions(self, dim: int) -> None:
        """Build FAISS index for question skeleton embeddings."""
        if not FAISS_AVAILABLE:
            return
            
        if not self.question_skeletons:
            return
        
        # Extract embeddings for all question skeletons
        embeddings = self.similarity.get_question_embeddings(self.question_skeletons)
        self.question_embeddings = np.array(embeddings, dtype=np.float32)
        
        # Create FAISS index
        if self.faiss_index_type == "hnsw":
            self.question_index = faiss.IndexHNSWFlat(dim, 32)
            self.question_index.hsnw.efSearch = 64
        elif self.faiss_index_type == "ivf":
            nlist = 10
            quantizer = faiss.IndexFlatL2(dim)
            self.question_index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            self.question_index.train(self.question_embeddings)
        else:  # flat
            self.question_index = faiss.IndexFlatL2(dim)
        
        # Add embeddings to index
        self.question_index.add(self.question_embeddings)

    def _build_faiss_index_for_sql(self, dim: int) -> None:
        """Build FAISS index for SQL skeleton embeddings."""
        if not FAISS_AVAILABLE:
            return
            
        if not self.sql_skeletons_list:
            return
        
        # Extract embeddings for all SQL skeletons
        embeddings = self.similarity.get_sql_embeddings(self.sql_skeletons_list)
        self.sql_embeddings = np.array(embeddings, dtype=np.float32)
        
        # Create FAISS index
        if self.faiss_index_type == "hnsw":
            self.sql_index = faiss.IndexHNSWFlat(dim, 32)
            self.sql_index.hsnw.efSearch = 64
        elif self.faiss_index_type == "ivf":
            nlist = 10
            quantizer = faiss.IndexFlatL2(dim)
            self.sql_index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            self.sql_index.train(self.sql_embeddings)
        else:  # flat
            self.sql_index = faiss.IndexFlatL2(dim)
        
        # Add embeddings to index
        self.sql_index.add(self.sql_embeddings)

    def retrieve_by_question(
        self,
        question: str,
        top_n: int = 5,
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-N most similar examples based on question skeleton.
        
        Uses FAISS for efficient similarity search.
        
        Args:
            question: The target natural language question
            top_n: Number of examples to retrieve
            show_progress: Whether to show progress
            
        Returns:
            List of dicts with example, similarity score, and skeletons
        """
        if not self.question_skeletons:
            raise ValueError(
                "Skeleton index not built. Call build_index() first."
            )

        # Extract skeleton from target question
        if show_progress:
            print(f"Extracting question skeleton for target...")
        target_skeleton = self.q_extractor.extract(question)

        # Use FAISS if available
        if FAISS_AVAILABLE and self.question_index is not None:
            similarities = self._retrieve_with_faiss_question(target_skeleton, top_n)
        else:
            # Fallback to in-memory similarity
            similarities = self.similarity.question_similarity_batch(
                target_skeleton,
                self.question_skeletons,
            )

        # Rank and return top-N
        return self._rank_and_return(similarities, top_n, target_skeleton, "question")

    def _retrieve_with_faiss_question(
        self, 
        target_skeleton: str, 
        top_n: int
    ) -> List[float]:
        """Retrieve using FAISS index."""
        # Get embedding for target skeleton
        target_embedding = self.similarity.get_question_embeddings([target_skeleton])[0]
        target_embedding = np.array(target_embedding, dtype=np.float32).reshape(1, -1)
        
        # Search with FAISS
        if self.question_index is not None:
            distances, indices = self.question_index.search(target_embedding, top_n)
            
            # Convert distances to similarities (1 - normalized_distance)
            similarities = 1.0 / (1.0 + distances[0])
            return similarities.tolist()
        else:
            # Fallback
            return self.similarity.question_similarity_batch(
                target_skeleton,
                self.question_skeletons,
            )

    def retrieve_by_sql(
        self,
        sql: str,
        top_n: int = 5,
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-N most similar examples based on SQL skeleton.
        
        This is useful for round-2 SQL generation refinement.
        
        Args:
            sql: The generated SQL statement
            top_n: Number of examples to retrieve
            show_progress: Whether to show progress
            
        Returns:
            List of dicts with example, similarity score, and skeletons
        """
        if not self.sql_skeletons:
            raise ValueError(
                "Skeleton index not built. Call build_index() first."
            )

        # Extract skeleton from target SQL
        if show_progress:
            print(f"Extracting SQL skeleton for target...")
        target_skeleton = self.sql_extractor.extract(sql)

        # Use FAISS if available
        if FAISS_AVAILABLE and self.sql_index is not None:
            similarities = self._retrieve_with_faiss_sql(target_skeleton, top_n)
        else:
            # Fallback to in-memory similarity
            similarities = self.similarity.sql_similarity_batch(
                target_skeleton,
                self.sql_skeletons,
            )

        # Rank and return top-N
        return self._rank_and_return(similarities, top_n, target_skeleton, "sql")

    def _retrieve_with_faiss_sql(
        self, 
        target_skeleton: str, 
        top_n: int
    ) -> List[float]:
        """Retrieve using FAISS index."""
        # Get embedding for target skeleton
        target_embedding = self.similarity.get_sql_embeddings([target_skeleton])[0]
        target_embedding = np.array(target_embedding, dtype=np.float32).reshape(1, -1)
        
        # Search with FAISS
        if self.sql_index is not None:
            distances, indices = self.sql_index.search(target_embedding, top_n)
            
            # Convert distances to similarities (1 - normalized_distance)
            similarities = 1.0 / (1.0 + distances[0])
            return similarities.tolist()
        else:
            # Fallback
            return self.similarity.sql_similarity_batch(
                target_skeleton,
                self.sql_skeletons,
            )

    def _rank_and_return(
        self,
        similarities: List[float],
        top_n: int,
        target_skeleton: str,
        skeleton_type: str,
    ) -> List[Dict[str, Any]]:
        """
        Rank examples by similarity and return top-N.
        
        Args:
            similarities: List of similarity scores
            top_n: Number of examples to return
            target_skeleton: The target skeleton (for reference)
            skeleton_type: Either "question" or "sql"
            
        Returns:
            List of ranked examples with metadata
        """
        # Create (index, similarity) pairs
        indexed_sims = list(enumerate(similarities))

        # Sort by similarity (descending)
        indexed_sims.sort(key=lambda x: x[1], reverse=True)

        # Take top-N
        top_indices = indexed_sims[:top_n]

        # Build result list
        results = []
        for idx, sim_score in top_indices:
            example = self.candidates[idx]
            result = {
                "example": example,
                "similarity_score": sim_score,
                "candidate_question_skeleton": self.question_skeletons[idx],
                "candidate_sql_skeleton": self.sql_skeletons[idx],
            }
            results.append(result)

        return results

    def save_index(self, path: str, save_faiss: bool = True) -> None:
        """
        Save the built index to disk for reuse.
        
        Args:
            path: Path to save the index (JSON file)
            save_faiss: Whether to save FAISS indexes
        """
        if not self.question_skeletons or not self.sql_skeletons:
            raise ValueError("No index to save. Call build_index() first.")

        # Save FAISS indexes if available
        faiss_data = None
        if save_faiss and FAISS_AVAILABLE:
            faiss_data = {
                "question_index": self._save_faiss_index(self.question_index),
                "sql_index": self._save_faiss_index(self.sql_index),
            }

        index_data = {
            "candidates": self.candidates,
            "question_skeletons": self.question_skeletons,
            "sql_skeletons": self.sql_skeletons,
            "faiss_data": faiss_data,
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)

        print(f"Index saved to {path}")

    def _save_faiss_index(self, index) -> Optional[bytes]:
        """Save FAISS index to bytes."""
        if index is None:
            return None
        return faiss.serialize_index(index)

    def load_index(self, path: str, load_faiss: bool = True) -> None:
        """
        Load a previously saved index from disk.
        
        Args:
            path: Path to the saved index (JSON file)
            load_faiss: Whether to load FAISS indexes
        """
        with open(path, "r", encoding="utf-8") as f:
            index_data = json.load(f)

        self.candidates = index_data["candidates"]
        self.question_skeletons = index_data["question_skeletons"]
        self.sql_skeletons = index_data["sql_skeletons"]
        
        # Load FAISS indexes if available
        if load_faiss and FAISS_AVAILABLE and "faiss_data" in index_data:
            faiss_data = index_data["faiss_data"]
            if faiss_data:
                self.question_index = self._load_faiss_index(
                    faiss_data.get("question_index")
                )
                self.sql_index = self._load_faiss_index(
                    faiss_data.get("sql_index")
                )

        print(f"Index loaded from {path} with {len(self.candidates)} examples")

    def _load_faiss_index(self, data: Optional[bytes]) -> Optional[faiss.Index]:
        """Load FAISS index from bytes."""
        if data is None:
            return None
        return faiss.deserialize_index(data)

    def shutdown(self):
        """Shut down vLLM engine and free resources."""
        self.q_extractor.shutdown()

    def __del__(self):
        """Ensure resources are freed on deletion."""
        try:
            self.shutdown()
        except:
            pass


def load_candidate_library_from_json(path: str) -> List[Dict[str, str]]:
    """
    Load candidate examples from a JSON file.
    
    Expected format:
    [
        {"question": "...", "sql": "...", ...},
        {"question": "...", "sql": "...", ...},
    ]
    
    Args:
        path: Path to JSON file
        
    Returns:
        List of candidate examples
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate format
    for i, entry in enumerate(data):
        if "question" not in entry or "sql" not in entry:
            raise ValueError(
                f"Entry {i} missing 'question' or 'sql' field. "
                f"Found keys: {list(entry.keys())}"
            )

    return data


def load_candidate_library_from_spider(
    train_path: str = "./train.json",
    max_examples: Optional[int] = None,
) -> List[Dict[str, str]]:
    """
    Load candidate examples from Spider dataset format.
    
    Args:
        train_path: Path to Spider train.json
        max_examples: Maximum number of examples to load (None for all)
        
    Returns:
        List of candidate examples with "question" and "sql" keys
    """
    with open(train_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    candidates = []
    for entry in data:
        candidates.append({
            "question": entry.get("question", ""),
            "sql": entry.get("query", ""),
            "db_id": entry.get("db_id", ""),
        })
        if max_examples and len(candidates) >= max_examples:
            break

    return candidates