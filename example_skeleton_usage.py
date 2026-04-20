"""
Skeleton Similarity Usage Examples

Demonstrates how to use the skeleton similarity components for:
1. Question skeleton extraction (Section 3.4.1)
2. SQL skeleton extraction (Section 3.4.2)
3. Similarity calculation and example retrieval

Run with:
    python example_skeleton_usage.py
"""

from schema_linking import (
    QuestionSkeletonExtractor,
    SQLSkeletonExtractor,
    SkeletonSimilarity,
    SkeletonRetriever,
    load_candidate_library_from_spider,
)


def example_1_question_skeleton_extraction():
    """
    Example 1: Extract question skeletons (Section 3.4.1)

    This extracts the structural skeleton Q⋆ from questions by removing
    domain-specific details, enabling structure-based similarity matching.
    """
    print("\n" + "="*60)
    print("Example 1: Question Skeleton Extraction (Section 3.4.1)")
    print("="*60)

    # Initialize extractor
    extractor = QuestionSkeletonExtractor(
        model_name="openai/gpt-oss-20b",
        tensor_parallel_size=1,
    )

    # Extract skeleton from a single question
    question = "How many singers are older than 20?"
    print(f"\nOriginal question: {question}")

    skeleton = extractor.extract(question)
    print(f"Extracted skeleton (Q⋆): {skeleton}")

    # Batch extraction
    questions = [
        "What is the average age of all employees?",
        "List the names of students who scored above 90 in Math",
        "Find the department with the highest budget",
    ]

    print(f"\nBatch extraction for {len(questions)} questions:")
    skeletons = extractor.extract_batch(questions, show_progress=True)

    for q, s in zip(questions, skeletons):
        print(f"\n  Q: {q}")
        print(f"  Q⋆: {s}")

    # Clean up
    extractor.shutdown()


def example_2_sql_skeleton_extraction():
    """
    Example 2: Extract SQL skeletons (Section 3.4.2)

    This extracts the structural skeleton S⋆ from SQL statements by
    replacing table/column names and values with placeholders.
    """
    print("\n" + "="*60)
    print("Example 2: SQL Skeleton Extraction (Section 3.4.2)")
    print("="*60)

    # Initialize extractor
    extractor = SQLSkeletonExtractor(dialect="sqlite")

    # Extract skeleton from a single SQL statement
    sql = "SELECT COUNT(*) FROM Singer WHERE Age > 20"
    print(f"\nOriginal SQL: {sql}")

    skeleton = extractor.extract(sql)
    print(f"Extracted skeleton (S⋆): {skeleton}")

    # More complex example with JOIN
    complex_sql = """
        SELECT Singer.name, Concert.concert_name
        FROM Singer
        JOIN Singer_in_Concert ON Singer.Singer_ID = Singer_in_Concert.Singer_ID
        JOIN Concert ON Singer_in_Concert.concert_ID = Concert.concert_ID
        WHERE Concert.Year > 2019
    """

    print(f"\nComplex SQL with JOIN:")
    print(f"  Original: {complex_sql[:100]}...")
    skeleton = extractor.extract(complex_sql)
    print(f"  Skeleton: {skeleton[:100]}...")

    # Batch extraction
    sqls = [
        "SELECT name FROM Employee WHERE Salary > 50000",
        "SELECT AVG(Age) FROM Student",
        "SELECT COUNT(*) FROM Products WHERE Price < 100",
    ]

    print(f"\nBatch extraction for {len(sqls)} SQL statements:")
    skeletons = extractor.extract_batch(sqls, show_progress=True)

    for sql, s in zip(sqls, skeletons):
        print(f"\n  SQL: {sql}")
        print(f"  S⋆: {s}")


def example_3_skeleton_similarity():
    """
    Example 3: Calculate skeleton similarity

    Demonstrates similarity calculation for both question and SQL skeletons.
    """
    print("\n" + "="*60)
    print("Example 3: Skeleton Similarity Calculation")
    print("="*60)

    sim = SkeletonSimilarity()

    # Question skeleton similarity
    print("\n--- Question Skeleton Similarity (Cosine) ---")
    q1 = "How many [ENTITY] have [ATTRIBUTE] greater than [VALUE]?"
    q2 = "What is the average [ATTRIBUTE] of [ENTITY]?"
    q3 = "How many [ENTITY] have [ATTRIBUTE] less than [VALUE]?"

    sim_12 = sim.question_similarity(q1, q2)
    sim_13 = sim.question_similarity(q1, q3)

    print(f"Q1⋆: {q1}")
    print(f"Q2⋆: {q2}")
    print(f"Q3⋆: {q3}")
    print(f"\nSimilarity(Q1⋆, Q2⋆): {sim_12:.3f}")
    print(f"Similarity(Q1⋆, Q3⋆): {sim_13:.3f}")
    print("(Q1 and Q3 should be more similar - same counting structure)")

    # SQL skeleton similarity
    print("\n--- SQL Skeleton Similarity (Edit Distance) ---")
    s1 = "SELECT COUNT([COLUMN]) FROM [TABLE] WHERE [COLUMN] > [VALUE]"
    s2 = "SELECT AVG([COLUMN]) FROM [TABLE]"
    s3 = "SELECT COUNT([COLUMN]) FROM [TABLE] WHERE [COLUMN] < [VALUE]"

    sim_s12 = sim.sql_similarity(s1, s2)
    sim_s13 = sim.sql_similarity(s1, s3)

    print(f"S1⋆: {s1}")
    print(f"S2⋆: {s2}")
    print(f"S3⋆: {s3}")
    print(f"\nSimilarity(S1⋆, S2⋆): {sim_s12:.3f}")
    print(f"Similarity(S1⋆, S3⋆): {sim_s13:.3f}")
    print("(S1 and S3 should be more similar - same COUNT structure with WHERE)")


def example_4_skeleton_retrieval():
    """
    Example 4: Retrieve similar examples using skeletons

    Demonstrates the full retrieval pipeline for both question-based
    and SQL-based example retrieval.
    """
    print("\n" + "="*60)
    print("Example 4: Skeleton-Based Example Retrieval")
    print("="*60)

    # Candidate library
    candidates = [
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
        {
            "question": "Find the department with the most employees",
            "sql": "SELECT Department FROM Employee GROUP BY Department ORDER BY COUNT(*) DESC LIMIT 1",
        },
    ]

    # Initialize retriever
    print("\nInitializing retriever...")
    retriever = SkeletonRetriever(candidates)

    # Build index (pre-computes skeletons)
    print("Building skeleton index...")
    retriever.build_index(show_progress=True)

    # Question-based retrieval
    print("\n--- Question-Based Retrieval (Section 3.4.1) ---")
    query_question = "How many actors are younger than 30?"
    print(f"\nQuery: {query_question}")

    results = retriever.retrieve_by_question(
        query_question,
        top_n=3,
        show_progress=False,
    )

    print(f"\nTop 3 most similar examples:")
    for i, result in enumerate(results, 1):
        print(f"\n  {i}. Similarity: {result['similarity_score']:.3f}")
        print(f"     Question: {result['example']['question']}")
        print(f"     SQL: {result['example']['sql']}")

    # SQL-based retrieval (for round-2 refinement)
    print("\n--- SQL-Based Retrieval (Section 3.4.2) ---")
    query_sql = "SELECT COUNT(*) FROM Actor WHERE Age < 30"
    print(f"\nQuery SQL: {query_sql}")

    results = retriever.retrieve_by_sql(
        query_sql,
        top_n=3,
        show_progress=False,
    )

    print(f"\nTop 3 most similar SQL examples:")
    for i, result in enumerate(results, 1):
        print(f"\n  {i}. Similarity: {result['similarity_score']:.3f}")
        print(f"     Question: {result['example']['question']}")
        print(f"     SQL: {result['example']['sql']}")

    # Save and load index
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        index_path = f.name

    print(f"\nSaving index to {index_path}...")
    retriever.save_index(index_path)

    print("Loading index...")
    new_retriever = SkeletonRetriever([])
    new_retriever.load_index(index_path)

    # Clean up
    retriever.shutdown()
    import os
    os.unlink(index_path)


def example_5_spider_dataset():
    """
    Example 5: Load candidate library from Spider dataset

    Demonstrates loading examples from the Spider dataset format.
    """
    print("\n" + "="*60)
    print("Example 5: Loading from Spider Dataset")
    print("="*60)

    # This would load from a real Spider dataset file
    # For demonstration, we'll show the expected format
    print("""
Expected Spider dataset format (train.json):
[
    {
        "db_id": "concert_singer",
        "question": "How many singers are there?",
        "query": "SELECT count(*) FROM Singer",
        "query_toks": [...],
        "query_toks_no_value": [...],
        "question_toks": [...],
        "question_toks_no_value": [...]
    },
    ...
]

Usage:
    candidates = load_candidate_library_from_spider(
        train_path="./data/train.json",
        max_examples=1000,  # Optional: limit for memory
    )

    retriever = SkeletonRetriever(candidates)
    retriever.build_index()
    """)


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("Skeleton Similarity Usage Examples")
    print("Sections 3.4.1 and 3.4.2")
    print("="*60)

    # Run examples (some require vLLM and actual models)
    print("\nNote: Examples 1, 2, and 4 require vLLM and a local model.")
    print("Examples 3 demonstrates the similarity calculation without model dependency.")

    try:
        # Example 3 works without vLLM
        example_3_skeleton_similarity()

        # Uncomment these if you have vLLM and a model installed:
        # example_1_question_skeleton_extraction()
        # example_2_sql_skeleton_extraction()
        # example_4_skeleton_retrieval()

        example_5_spider_dataset()

    except ImportError as e:
        print(f"\nSkipping vLLM-dependent examples: {e}")
        print("Install vLLM with: pip install vllm")

    print("\n" + "="*60)
    print("Examples complete!")
    print("="*60)


if __name__ == "__main__":
    main()
