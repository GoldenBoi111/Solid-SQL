"""
Tests for Skeleton Similarity Components

Tests for:
- Question skeleton extraction (Section 3.4.1)
- SQL skeleton extraction (Section 3.4.2)
- Skeleton similarity calculations
- Skeleton-based example retrieval

Run with:
    pytest test_skeleton_similarity.py -v
"""

import pytest
import json
from pathlib import Path
import tempfile


# ============================================================
# SQL Skeleton Extractor Tests
# ============================================================

class TestSQLSkeletonExtractor:
    """Tests for SQL skeleton extraction."""

    def setup_method(self):
        from schema_linking.sql_skeleton_extractor import SQLSkeletonExtractor
        self.extractor = SQLSkeletonExtractor(dialect="sqlite")

    def test_extract_simple_select(self):
        """Test skeleton extraction for simple SELECT."""
        sql = "SELECT name, age FROM Singer"
        skeleton = self.extractor.extract(sql)

        assert "[TABLE]" in skeleton
        assert "[COLUMN]" in skeleton
        # Should preserve SQL keywords
        assert "SELECT" in skeleton.upper()
        assert "FROM" in skeleton.upper()

    def test_extract_where_clause(self):
        """Test skeleton extraction with WHERE clause."""
        sql = "SELECT COUNT(*) FROM Singer WHERE Age > 20"
        skeleton = self.extractor.extract(sql)

        assert "[TABLE]" in skeleton
        assert "[COLUMN]" in skeleton
        assert "[VALUE]" in skeleton
        assert "WHERE" in skeleton.upper()

    def test_extract_join(self):
        """Test skeleton extraction with JOIN."""
        sql = """
            SELECT Singer.name, Concert.concert_name
            FROM Singer
            JOIN Singer_in_Concert ON Singer.Singer_ID = Singer_in_Concert.Singer_ID
            JOIN Concert ON Singer_in_Concert.concert_ID = Concert.concert_ID
        """
        skeleton = self.extractor.extract(sql)

        # Should have multiple table placeholders
        assert skeleton.count("[TABLE]") >= 2
        assert "[COLUMN]" in skeleton
        assert "JOIN" in skeleton.upper()

    def test_extract_aggregation(self):
        """Test skeleton extraction with aggregation functions."""
        sql = "SELECT AVG(Salary), COUNT(*) FROM Employee WHERE Department = 'Engineering'"
        skeleton = self.extractor.extract(sql)

        assert "[VALUE]" in skeleton
        # The string literal should be replaced
        assert "'Engineering'" not in skeleton

    def test_extract_batch(self):
        """Test batch skeleton extraction."""
        sqls = [
            "SELECT name FROM Singer",
            "SELECT COUNT(*) FROM Employee WHERE Age > 25",
        ]
        skeletons = self.extractor.extract_batch(sqls, show_progress=False)

        assert len(skeletons) == 2
        assert all("[TABLE]" in s for s in skeletons)


class TestSQLSkeletonEditDistance:
    """Tests for SQL skeleton edit distance calculation."""

    def test_identical_skeletons(self):
        """Identical skeletons should have distance 0."""
        from schema_linking.sql_skeleton_extractor import sql_skeleton_edit_distance

        s1 = "SELECT [COLUMN] FROM [TABLE] WHERE [COLUMN] > [VALUE]"
        s2 = "SELECT [COLUMN] FROM [TABLE] WHERE [COLUMN] > [VALUE]"

        distance = sql_skeleton_edit_distance(s1, s2)
        assert distance == 0.0

    def test_different_skeletons(self):
        """Different skeletons should have distance > 0."""
        from schema_linking.sql_skeleton_extractor import sql_skeleton_edit_distance

        s1 = "SELECT [COLUMN] FROM [TABLE]"
        s2 = "SELECT COUNT([COLUMN]) FROM [TABLE] WHERE [COLUMN] > [VALUE]"

        distance = sql_skeleton_edit_distance(s1, s2)
        assert distance > 0.0
        assert distance <= 1.0

    def test_similarity_score(self):
        """Test similarity score calculation."""
        from schema_linking.sql_skeleton_extractor import sql_skeleton_similarity

        s1 = "SELECT [COLUMN] FROM [TABLE] WHERE [COLUMN] > [VALUE]"
        s2 = "SELECT [COLUMN] FROM [TABLE] WHERE [COLUMN] = [VALUE]"

        similarity = sql_skeleton_similarity(s1, s2)
        assert 0.0 <= similarity <= 1.0
        # Should be high since they differ only in operator
        assert similarity > 0.7


# ============================================================
# Skeleton Similarity Tests
# ============================================================

class TestSkeletonSimilarity:
    """Tests for skeleton similarity calculations."""

    def setup_method(self):
        from schema_linking.skeleton_similarity import SkeletonSimilarity
        self.sim = SkeletonSimilarity()

    def test_question_similarity_identical(self):
        """Identical question skeletons should have high similarity."""
        s1 = "How many [ENTITY] have [ATTRIBUTE] greater than [VALUE]?"
        s2 = "How many [ENTITY] have [ATTRIBUTE] greater than [VALUE]?"

        similarity = self.sim.question_similarity(s1, s2)
        assert similarity > 0.99

    def test_question_similarity_different(self):
        """Different question skeletons should have lower similarity."""
        s1 = "How many [ENTITY] have [ATTRIBUTE] greater than [VALUE]?"
        s2 = "List all [ATTRIBUTE] of [ENTITY]"

        similarity = self.sim.question_similarity(s1, s2)
        assert similarity < 0.9

    def test_sql_similarity_identical(self):
        """Identical SQL skeletons should have similarity 1.0."""
        s1 = "SELECT [COLUMN] FROM [TABLE] WHERE [COLUMN] > [VALUE]"
        s2 = "SELECT [COLUMN] FROM [TABLE] WHERE [COLUMN] > [VALUE]"

        similarity = self.sim.sql_similarity(s1, s2)
        assert similarity == 1.0

    def test_question_similarity_batch(self):
        """Test batch question similarity calculation."""
        query = "How many [ENTITY] have [ATTRIBUTE] greater than [VALUE]?"
        candidates = [
            "What is the average [ATTRIBUTE] of [ENTITY]?",
            "How many [ENTITY] have [ATTRIBUTE] less than [VALUE]?",
            "List all [ATTRIBUTE] of [ENTITY]",
        ]

        similarities = self.sim.question_similarity_batch(query, candidates)
        assert len(similarities) == 3
        # Second should be most similar (same structure)
        assert similarities[1] > similarities[0]
        assert similarities[1] > similarities[2]


# ============================================================
# Skeleton Retriever Tests
# ============================================================

class TestSkeletonRetriever:
    """Tests for skeleton-based example retrieval."""

    def setup_method(self):
        """Set up test fixtures."""
        self.candidates = [
            {
                "question": "How many singers are older than 20?",
                "sql": "SELECT COUNT(*) FROM Singer WHERE Age > 20",
            },
            {
                "question": "What is the average salary of employees?",
                "sql": "SELECT AVG(Salary) FROM Employee",
            },
            {
                "question": "List the names of students who scored above 90",
                "sql": "SELECT Name FROM Student WHERE Score > 90",
            },
        ]

    def test_build_index(self):
        """Test index building."""
        from schema_linking.skeleton_retriever import SkeletonRetriever

        retriever = SkeletonRetriever(self.candidates)
        retriever.build_index(show_progress=False)

        assert len(retriever.question_skeletons) == 3
        assert len(retriever.sql_skeletons) == 3

    def test_retrieve_by_question(self):
        """Test retrieval by question skeleton."""
        from schema_linking.skeleton_retriever import SkeletonRetriever

        retriever = SkeletonRetriever(self.candidates)
        retriever.build_index(show_progress=False)

        results = retriever.retrieve_by_question(
            "How many actors are younger than 30?",
            top_n=2,
            show_progress=False,
        )

        assert len(results) == 2
        # First result should be most similar (counting with condition)
        assert all("similarity_score" in r for r in results)
        # Results should be sorted by similarity (descending)
        assert results[0]["similarity_score"] >= results[1]["similarity_score"]

    def test_retrieve_by_sql(self):
        """Test retrieval by SQL skeleton."""
        from schema_linking.skeleton_retriever import SkeletonRetriever

        retriever = SkeletonRetriever(self.candidates)
        retriever.build_index(show_progress=False)

        results = retriever.retrieve_by_sql(
            "SELECT COUNT(*) FROM Actor WHERE Age < 30",
            top_n=2,
            show_progress=False,
        )

        assert len(results) == 2
        assert all("similarity_score" in r for r in results)
        # Results should be sorted by similarity (descending)
        assert results[0]["similarity_score"] >= results[1]["similarity_score"]

    def test_save_and_load_index(self):
        """Test saving and loading index."""
        from schema_linking.skeleton_retriever import SkeletonRetriever

        retriever = SkeletonRetriever(self.candidates)
        retriever.build_index(show_progress=False)

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            retriever.save_index(temp_path)

            # Create new retriever and load
            new_retriever = SkeletonRetriever([])
            new_retriever.load_index(temp_path)

            assert len(new_retriever.candidates) == 3
            assert len(new_retriever.question_skeletons) == 3
            assert len(new_retriever.sql_skeletons) == 3
        finally:
            Path(temp_path).unlink()

    def test_retrieve_without_index_raises_error(self):
        """Test that retrieval without built index raises error."""
        from schema_linking.skeleton_retriever import SkeletonRetriever

        retriever = SkeletonRetriever(self.candidates)

        with pytest.raises(ValueError, match="build_index"):
            retriever.retrieve_by_question("How many singers?")

        with pytest.raises(ValueError, match="build_index"):
            retriever.retrieve_by_sql("SELECT COUNT(*) FROM Singer")


# ============================================================
# Integration Tests
# ============================================================

class TestSkeletonIntegration:
    """Integration tests for the full skeleton similarity pipeline."""

    def test_full_pipeline_question_based(self):
        """Test full pipeline: extract skeletons -> retrieve similar examples."""
        from schema_linking.skeleton_retriever import SkeletonRetriever

        candidates = [
            {
                "question": "How many singers are older than 20?",
                "sql": "SELECT COUNT(*) FROM Singer WHERE Age > 20",
            },
            {
                "question": "What is the name of the youngest student?",
                "sql": "SELECT Name FROM Student ORDER BY Age ASC LIMIT 1",
            },
        ]

        retriever = SkeletonRetriever(candidates)
        retriever.build_index(show_progress=False)

        # Retrieve similar examples for a new question
        results = retriever.retrieve_by_question(
            "How many employees have salary greater than 50000?",
            top_n=1,
            show_progress=False,
        )

        assert len(results) == 1
        assert "example" in results[0]
        assert "similarity_score" in results[0]

    def test_full_pipeline_sql_based(self):
        """Test full pipeline for SQL-based retrieval (round-2)."""
        from schema_linking.skeleton_retriever import SkeletonRetriever

        candidates = [
            {
                "question": "Count the number of singers",
                "sql": "SELECT COUNT(*) FROM Singer",
            },
            {
                "question": "Get average age of employees",
                "sql": "SELECT AVG(Age) FROM Employee",
            },
        ]

        retriever = SkeletonRetriever(candidates)
        retriever.build_index(show_progress=False)

        # Retrieve similar examples for generated SQL
        results = retriever.retrieve_by_sql(
            "SELECT COUNT(*) FROM Actor",
            top_n=1,
            show_progress=False,
        )

        assert len(results) == 1
        # Should retrieve the COUNT example
        assert results[0]["example"]["sql"].startswith("SELECT COUNT")


# ============================================================
# Run tests
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
