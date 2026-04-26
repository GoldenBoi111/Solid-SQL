#!/usr/bin/env python3
"""
Build and test the SolidSQL retrieval system.

This script demonstrates how to:
1. Load candidate examples from Spider dataset or create manually
2. Build FAISS indexes for efficient retrieval
3. Test question-based and SQL-based retrieval
4. Save and load the built index

Usage:
    python build_retrieval.py
"""

import sys
from pathlib import Path

# Add schema_linking to path
sys.path.insert(0, str(Path(__file__).parent / "schema_linking"))

from skeleton_retriever import SkeletonRetriever, load_candidate_library_from_spider, load_candidate_library_from_json


def main():
    """Build and test the retrieval system."""
    
    print("=" * 60)
    print("SolidSQL Retrieval System Builder")
    print("=" * 60)
    
    # Option 1: Load from Spider dataset
    # Uncomment and modify these lines if you have Spider data
    """
    print("\n1. Loading candidate examples from Spider dataset...")
    candidates = load_candidate_library_from_spider(
        train_path="data/train.json",  # Path to your Spider train.json
        max_examples=1000  # Adjust as needed
    )
    print(f"   Loaded {len(candidates)} examples")
    """
    
    # Option 2: Create sample examples manually
    # Commented out to use JSON file instead
    """
    print("\n1. Creating sample candidate examples...")
    candidates = [
        {
            "question": "How many singers are older than 20?",
            "sql": "SELECT COUNT(*) FROM Singer WHERE Age > 20",
            "db_id": "concert_singer"
        },
        {
            "question": "What is the average salary of employees?",
            "sql": "SELECT AVG(Salary) FROM Employee",
            "db_id": "employee_dept"
        },
        {
            "question": "List the names of students who scored above 90",
            "sql": "SELECT Name FROM Student WHERE Score > 90",
            "db_id": "student_records"
        },
        {
            "question": "Find the department with the highest budget",
            "sql": "SELECT Department FROM Employee GROUP BY Department ORDER BY SUM(Budget) DESC LIMIT 1",
            "db_id": "employee_dept"
        },
        {
            "question": "Show all albums by a specific singer",
            "sql": "SELECT Album.title FROM Album JOIN Singer ON Album.singer_id = Singer.id WHERE Singer.name = ?",
            "db_id": "concert_singer"
        },
        {
            "question": "How many movies are in the database?",
            "sql": "SELECT COUNT(*) FROM Movie",
            "db_id": "movie_db"
        },
        {
            "question": "What is the total revenue?",
            "sql": "SELECT SUM(Revenue) FROM Sales",
            "db_id": "sales_db"
        },
        {
            "question": "List all customers who made a purchase",
            "sql": "SELECT DISTINCT Customer.name FROM Customer JOIN Purchase ON Customer.id = Purchase.customer_id",
            "db_id": "ecommerce"
        },
        {
            "question": "What is the youngest employee's age?",
            "sql": "SELECT MIN(Age) FROM Employee",
            "db_id": "employee_dept"
        },
        {
            "question": "Which department has the most employees?",
            "sql": "SELECT Department FROM Employee GROUP BY Department ORDER BY COUNT(*) DESC LIMIT 1",
            "db_id": "employee_dept"
        },
    ]
    print(f"   Created {len(candidates)} sample examples")
    """
    
    # Option 3: Load from JSON file
    # Uncomment and modify these lines if you have a JSON file
    print("\n1. Loading candidate examples from JSON file...")
    candidates = load_candidate_library_from_json("my_examples.json")  # <-- Change this path to your JSON file
    print(f"   Loaded {len(candidates)} examples")
    
    # Build retriever
    print("\n2. Building retrieval system...")
    retriever = SkeletonRetriever(
        candidate_examples=candidates,
        faiss_index_type="hnsw",  # Options: "flat", "ivf", "hnsw"
    )
    
    # Build FAISS index
    print("\n3. Building FAISS indexes...")
    retriever.build_index(show_progress=True)
    
    # Test question-based retrieval
    print("\n4. Testing question-based retrieval...")
    test_question = "How many actors are younger than 30?"
    results = retriever.retrieve_by_question(
        question=test_question,
        top_n=3,
        show_progress=True,
    )
    
    print(f"\nTop 3 similar examples for: '{test_question}'")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Similarity: {result['similarity_score']:.3f}")
        print(f"   Question: {result['example']['question']}")
        print(f"   SQL: {result['example']['sql']}")
        print(f"   DB ID: {result['example'].get('db_id', 'N/A')}")
    
    # Test SQL-based retrieval
    print("\n5. Testing SQL-based retrieval...")
    test_sql = "SELECT COUNT(*) FROM Actor WHERE age < 30"
    results = retriever.retrieve_by_sql(
        sql=test_sql,
        top_n=3,
        show_progress=True,
    )
    
    print(f"\nTop 3 similar examples for: '{test_sql}'")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Similarity: {result['similarity_score']:.3f}")
        print(f"   Question: {result['example']['question']}")
        print(f"   SQL: {result['example']['sql']}")
    
    # Save index
    print("\n6. Saving retrieval index...")
    index_path = "retrieval_index.json"
    retriever.save_index(index_path, save_faiss=True)
    
    # Load and verify index
    print("\n7. Loading and verifying saved index...")
    new_retriever = SkeletonRetriever([])
    new_retriever.load_index(index_path, load_faiss=True)
    
    # Test loaded index
    results = new_retriever.retrieve_by_question(
        question=test_question,
        top_n=2,
        show_progress=False,
    )
    print(f"   Loaded index can retrieve: {len(results)} results")
    
    # Cleanup
    retriever.shutdown()
    new_retriever.shutdown()
    
    print("\n" + "=" * 60)
    print("✓ Retrieval system built and tested successfully!")
    print(f"  Index saved to: {index_path}")
    print("=" * 60)
    
    print("\nNext steps:")
    print("  1. Load the index in your application:")
    print("     retriever = SkeletonRetriever([])")
    print("     retriever.load_index('retrieval_index.json')")
    print()
    print("  2. Use for retrieval:")
    print("     results = retriever.retrieve_by_question(question='...', top_n=5)")


if __name__ == "__main__":
    main()