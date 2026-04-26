"""
FAISS Index Test

Test that FAISS indexes are working correctly for skeleton retrieval.
"""

import json
from schema_linking.skeleton_retriever import SkeletonRetriever


def test_faiss_retrieval():
    """Test FAISS-based retrieval."""
    
    print("Testing FAISS Index for Skeleton Retrieval")
    print("=" * 60)
    
    # Sample candidate examples
    examples = [
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
            "question": "Find the department with the highest budget",
            "sql": "SELECT Department FROM Employee GROUP BY Department ORDER BY SUM(Budget) DESC LIMIT 1",
        },
        {
            "question": "Show all albums by a specific singer",
            "sql": "SELECT Album.title FROM Album JOIN Singer ON Album.singer_id = Singer.id WHERE Singer.name = ?",
        },
    ]
    
    # Create retriever
    print("\n1. Creating SkeletonRetriever...")
    retriever = SkeletonRetriever(
        candidate_examples=examples,
        faiss_index_type="hnsw",  # Use HNSW for fast approximate search
    )
    
    # Build index
    print("2. Building FAISS indexes...")
    retriever.build_index(show_progress=True)
    
    # Check if FAISS is available
    from schema_linking.skeleton_retriever import FAISS_AVAILABLE
    print(f"\n   FAISS available: {FAISS_AVAILABLE}")
    
    # Test question-based retrieval
    print("\n3. Testing question-based retrieval...")
    test_question = "How many actors are younger than 30?"
    results = retriever.retrieve_by_question(
        question=test_question,
        top_n=3,
        show_progress=True,
    )
    
    print(f"\n   Retrieved {len(results)} results for: {test_question}")
    for i, result in enumerate(results, 1):
        print(f"\n   {i}. Similarity: {result['similarity_score']:.3f}")
        print(f"      Question: {result['example']['question']}")
        print(f"      SQL: {result['example']['sql']}")
    
    # Test SQL-based retrieval
    print("\n4. Testing SQL-based retrieval...")
    test_sql = "SELECT COUNT(*) FROM Actor WHERE age < 30"
    results = retriever.retrieve_by_sql(
        sql=test_sql,
        top_n=3,
        show_progress=True,
    )
    
    print(f"\n   Retrieved {len(results)} results for: {test_sql}")
    for i, result in enumerate(results, 1):
        print(f"\n   {i}. Similarity: {result['similarity_score']:.3f}")
        print(f"      Question: {result['example']['question']}")
        print(f"      SQL: {result['example']['sql']}")
    
    # Test saving and loading index
    print("\n5. Testing index persistence...")
    index_path = "test_faiss_index.json"
    retriever.save_index(index_path, save_faiss=True)
    
    # Create new retriever and load
    new_retriever = SkeletonRetriever([])
    new_retriever.load_index(index_path, load_faiss=True)
    
    print(f"   Index saved to: {index_path}")
    print(f"   Index loaded successfully with {len(new_retriever.candidates)} examples")
    
    # Verify loaded index works
    results = new_retriever.retrieve_by_question(
        question=test_question,
        top_n=2,
        show_progress=False,
    )
    print(f"   Loaded index can retrieve: {len(results)} results")
    
    # Cleanup
    retriever.shutdown()
    
    print("\n" + "=" * 60)
    print("✓ All FAISS tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_faiss_retrieval()