#!/usr/bin/env python3
"""
Build FAISS Indexes from JSON Input

This script builds FAISS indexes for the SolidSQL retrieval system from a JSON file
containing question-SQL pairs. It creates indexes for:
1. Question skeletons (Q⋆)
2. SQL skeletons (S⋆)
3. Question embeddings
4. SQL embeddings

Usage:
    python build_index_from_json.py --input examples.json --output index.json

Options:
    --input, -i     Input JSON file path (required)
    --output, -o    Output index file path (default: retrieval_index.json)
    --type, -t      FAISS index type: flat, ivf, hnsw (default: hnsw)
    --top-n         Number of results to retrieve (default: 5)
    --max-examples  Maximum number of examples to load (default: all)

JSON Input Format:
    [
        {
            "question": "How many singers are older than 20?",
            "sql": "SELECT COUNT(*) FROM Singer WHERE Age > 20",
            "db_id": "concert_singer"  // Optional
        },
        ...
    ]

Also supports Spider-style records with `SQL` or `query` fields.
"""

import argparse
import json
import sys
from pathlib import Path

# Add schema_linking to path
sys.path.insert(0, str(Path(__file__).parent / "schema_linking"))

from skeleton_retriever import SkeletonRetriever, load_candidate_library_from_json


def load_examples(input_path: str, max_examples: int = None) -> list:
    """Load examples from JSON file."""
    print(f"Loading examples from: {input_path}")
    
    if not Path(input_path).exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    
    try:
        examples = load_candidate_library_from_json(input_path)
        print(f"Loaded {len(examples)} examples")
        
        if max_examples and len(examples) > max_examples:
            print(f"Truncating to {max_examples} examples")
            examples = examples[:max_examples]
        
        return examples
    except Exception as e:
        print(f"Error loading JSON: {e}")
        sys.exit(1)


def build_and_save_index(
    examples: list,
    output_path: str,
    index_type: str = "hnsw",
    top_n: int = 5,
):
    """Build FAISS index and save it."""
    
    print("\n" + "=" * 60)
    print("Building FAISS Index")
    print("=" * 60)
    
    # Create retriever
    print(f"\n1. Creating SkeletonRetriever with {len(examples)} examples")
    print(f"   Index type: {index_type}")
    
    retriever = SkeletonRetriever(
        candidate_examples=examples,
        faiss_index_type=index_type,
    )
    
    # Build index
    print("\n2. Building FAISS indexes...")
    retriever.build_index(show_progress=True)
    
    # Save index
    print(f"\n3. Saving index to: {output_path}")
    retriever.save_index(output_path, save_faiss=True)
    
    # Test retrieval
    print("\n4. Testing retrieval...")
    if examples:
        test_question = examples[0]["question"]
        results = retriever.retrieve_by_question(
            question=test_question,
            top_n=top_n,
            show_progress=True,
        )
        
        print(f"\nTest results for: '{test_question}'")
        for i, result in enumerate(results, 1):
            print(f"  {i}. Similarity: {result['similarity_score']:.3f}")
            print(f"     Question: {result['example']['question'][:60]}...")
    
    # Cleanup
    retriever.shutdown()
    
    print("\n" + "=" * 60)
    print("✓ Index built and saved successfully!")
    print("=" * 60)


def main():
    """Main function."""
    
    parser = argparse.ArgumentParser(
        description="Build FAISS indexes for SolidSQL retrieval system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build index with default settings
  python build_index_from_json.py --input examples.json

  # Build index with specific settings
  python build_index_from_json.py \\
      --input examples.json \\
      --output my_index.json \\
      --type hnsw \\
      --max-examples 1000

  # Test retrieved examples
  python build_index_from_json.py --input examples.json --output index.json
  python test_retrieval.py --index index.json
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input JSON file path",
    )
    
    parser.add_argument(
        "--output", "-o",
        default="retrieval_index.json",
        help="Output index file path (default: retrieval_index.json)",
    )
    
    parser.add_argument(
        "--type", "-t",
        choices=["flat", "ivf", "hnsw"],
        default="hnsw",
        help="FAISS index type (default: hnsw)",
    )
    
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of results to retrieve (default: 5)",
    )
    
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to load (default: all)",
    )
    
    args = parser.parse_args()
    
    # Load examples
    examples = load_examples(args.input, args.max_examples)
    
    if not examples:
        print("Error: No examples found in input file")
        sys.exit(1)
    
    # Build and save index
    build_and_save_index(examples, args.output, args.type, args.top_n)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Index Summary")
    print("=" * 60)
    print(f"  Input file: {args.input}")
    print(f"  Output file: {args.output}")
    print(f"  Index type: {args.type}")
    print(f"  Examples: {len(examples)}")
    print(f"  Top-N: {args.top_n}")
    print("=" * 60)
    
    print("\nNext steps:")
    print("  1. Load the index in your application:")
    print("     retriever = SkeletonRetriever([])")
    print("     retriever.load_index('retrieval_index.json')")
    print()
    print("  2. Use for retrieval:")
    print("     results = retriever.retrieve_by_question(question='...', top_n=5)")
    print()
    print("  3. Or use with SolidSQL:")
    print("     solidsql = SolidSQL(candidate_examples=examples)")
    print("     result = solidsql.generate_sql(question='...', schema='...', top_n=5)")


if __name__ == "__main__":
    main()
