"""
Schema Linking Module for SolidSQL

Trains a model G(Q, SC) -> {T, C} that predicts relevant tables
and columns from a question and database schema.

Skeleton Similarity Components (Sections 3.4.1 and 3.4.2):
- QuestionSkeletonExtractor: Extracts Q⋆ from questions
- SQLSkeletonExtractor: Extracts S⋆ from SQL statements
- SkeletonSimilarity: Calculates similarity for both skeleton types
- SkeletonRetriever: Retrieves top-N similar examples from candidate library
"""

from .question_skeleton_extractor import QuestionSkeletonExtractor
from .sql_skeleton_extractor import SQLSkeletonExtractor, sql_skeleton_edit_distance, sql_skeleton_similarity
from .skeleton_similarity import SkeletonSimilarity
from .skeleton_retriever import SkeletonRetriever, load_candidate_library_from_json, load_candidate_library_from_spider

__all__ = [
    "QuestionSkeletonExtractor",
    "SQLSkeletonExtractor",
    "sql_skeleton_edit_distance",
    "sql_skeleton_similarity",
    "SkeletonSimilarity",
    "SkeletonRetriever",
    "load_candidate_library_from_json",
    "load_candidate_library_from_spider",
]
