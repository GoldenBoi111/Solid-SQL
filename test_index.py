#!/usr/bin/env python3
"""
Test Retrieval with Built Index

This script tests the built FAISS index by loading it and performing
retrieval queries.

Usage:
    python test_index.py --index retrieval_index.json

Options:
    --index, -i     Path to the built index file (default: retrieval_index.json)
    --question, -q  Test question for retrieval
    --sql, -s       Test SQL for retrieval
    --top-n         Number of results to retrieve (default: 3)
"""

import argparse
import sys
from pathlib import Path

# Add schema_linking to path
sys.path.insert(0, str(Path(__file__).parent / "schema_linking"))

from skeleton_retriever import SkeletonRetriever


def main():
    """Main function."""
    
    parser = argparse.ArgumentParser(
        description="Test retrieval with built FAISS index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with default index
  python test_index.py

  # Test with specific index
  python test_index.py --index my_index.json

  # Test with specific question
  python test_index.py --question "How many actors are younger than 30?"

  # Test with specific SQL
  python test_index.py --sql "SELECT COUNT(*) FROM Actor WHERE age < 30"
        """
    )
    
    parser.add_argument(
        "--index", "-i",
        default="retrieval_index.json",
        help="Path to the built index file (default: retrieval_index.json)",
    )
    
    parser.add_argument(
        "--question", "-q",
        default=None,
        help="Test question for retrieval",
    )
    
    parser.add_argument(
        "--sql", "-s",
        default=None,
        help="Test SQL for retrieval",
    )
    
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of results to retrieve (default: 3)",
    )
    
    args = parser.parse_args()
    
    # Check if index file exists
    index_path = Path(args.index)
    if not index_path.exists():
        print(f"Error: Index file not found: {args.index}")
        print("\nBuild an index first:")
        print(f"  python build_index_from_json.py --input examples.json --output {args.index}")
        sys.exit(1)
    
    print("=" * 60)
    print("Testing FAISS Index")
    print("=" * 60)
    print(f"Index file: {args.index}")
    
    # Load index
    print("\n1. Loading index...")
    retriever = SkeletonRetriever([])
    retriever.load_index(args.index, load_faiss=True)
    
    print(f"   Loaded {len(retriever.candidates)} examples")
    
    # Test retrieval
    print("\n2. Testing retrieval...")
    
    if args.question:
        print(f"\nQuestion: {args.question}")
        results = retriever.retrieve_by_question(
            question=args.question,
            top_n=args.top_n,
            show_progress=True,
        )
        
        print(f"\nTop {len(results)} similar examples:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Similarity: {result['similarity_score']:.3f}")
            print(f"   Question: {result['example']['question']}")
            print(f"   SQL: {result['example']['sql']}")
            print(f"   DB ID: {result['example'].get('db_id', 'N/A')}")
    
    if args.sql:
        print(f"\nSQL: {args.sql}")
        results = retriever.retrieve_by_sql(
            sql=args.sql,
            top_n=args.top_n,
            show_progress=True,
        )
        
        print(f"\nTop {len(results)} similar examples:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Similarity: {result['similarity_score']:.3f}")
            print(f"   Question: {result['example']['question']}")
            print(f"   SQL: {result['example']['sql']}")
            print(f"   DB ID: {result['example'].get('db_id', 'N/A')}")
    
    # If no test specified, use first example
    if not args.question and not args.sql and retriever.candidates:
        print("\n3. Using first example as test...")
        test_example = retriever.candidates[0]
        
        print(f"\nQuestion: {test_example['question']}")
        results = retriever.retrieve_by_question(
            question=test_example['question'],
            top_n=args.top_n,
            show_progress=True,
        )
        
        print(f"\nTop {len(results)} similar examples:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Similarity: {result['similarity_score']:.3f}")
            print(f"   Question: {result['example']['question']}")
            print(f"   SQL: {result['example']['sql']}")
    
    # Cleanup
    retriever.shutdown()
    
    print("\n" + "=" * 60)
    print("✓ Test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()