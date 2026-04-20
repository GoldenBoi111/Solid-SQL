"""
SolidSQL Full Pipeline Example

Demonstrates the complete SolidSQL system with:
1. Schema linking (original task)
2. Skeleton-based retrieval (Section 3.4.1 and 3.4.2)
3. Round-2 refinement pipeline

This shows how the system works end-to-end following the SolidSQL methodology.
"""

import json
from solidsql import SolidSQL


def main():
    print("=== SolidSQL Full Pipeline Demo ===")
    print()

    # Sample candidate examples (would normally come from Spider or other datasets)
    candidate_examples = [
        {
            "question": "How many singers are older than 20?",
            "sql": "SELECT COUNT(*) FROM Singer WHERE Age > 20",
            "db_id": "concert_singer",
        },
        {
            "question": "What is the average salary of employees?",
            "sql": "SELECT AVG(Salary) FROM Employee",
            "db_id": "employee_dept",
        },
        {
            "question": "List the names of students who scored above 90",
            "sql": "SELECT Name FROM Student WHERE Score > 90",
            "db_id": "student_records",
        },
        {
            "question": "Find the department with the highest budget",
            "sql": "SELECT Department FROM Employee GROUP BY Department ORDER BY SUM(Budget) DESC LIMIT 1",
            "db_id": "employee_dept",
        },
        {
            "question": "Show all albums by a specific singer",
            "sql": "SELECT Album.title FROM Album JOIN Singer ON Album.singer_id = Singer.id WHERE Singer.name = ?",
            "db_id": "concert_singer",
        },
    ]

    print("1. Initializing SolidSQL system with candidate examples...")
    solidsql = SolidSQL(candidate_examples=candidate_examples, build_index=True)

    print("   ✓ System initialized with", len(candidate_examples), "candidate examples")
    print()

    # Example question to process
    question = "How many actors are younger than 30?"
    schema_text = "Actor(id, name, age)\nMovie(id, actor_id, title)"

    print("2. Processing question:", question)
    print("   Schema:", schema_text)
    print()

    # Full end-to-end pipeline
    print("3. Running complete SolidSQL pipeline...")
    result = solidsql.generate_sql(
        question=question, schema_text=schema_text, top_n=3, round_2_refinement=True
    )

    print("   ✓ Generation complete")
    print()

    # Display results
    print("4. Results:")
    print("-" * 50)
    print("Original question:", result["question"])
    print("Schema:", result["schema"])
    print()

    print("Schema Linking Result:")
    print("  Tables:", result["schema_linking_result"].get("tables", []))
    print("  Columns:", result["schema_linking_result"].get("columns", []))
    print()

    print("Question Skeleton:", result["question_skeleton"])
    print()

    if result["retrieval_results"]:
        print("Retrieval Results (top 3):")
        for i, item in enumerate(result["retrieval_results"][:3], 1):
            print(f"  {i}. Similarity: {item['similarity_score']:.3f}")
            print(f"     Question: {item['example']['question']}")
            print(f"     SQL: {item['example']['sql']}")
            print(f"     Q⋆: {item['candidate_question_skeleton']}")
            print()
    else:
        print("No retrieval results (index not built)")
        print()

    if result["refined_sql"]:
        print("Round-2 Refined SQL:")
        print("  ", result["refined_sql"])
        print()
    else:
        print("Round-2 refinement disabled or not applicable")
        print()

    print("5. Full Generation Details:")
    print(json.dumps(result["full_generation"], indent=2, ensure_ascii=False))
    print()

    print("=== Pipeline Complete ===")
    print()
    print("This demonstrates the complete SolidSQL workflow:")
    print("- Schema linking identifies relevant tables/columns")
    print("- Skeleton-based retrieval finds similar examples")
    print("- Round-2 refinement improves SQL quality")
    print("- All components work together end-to-end")

    # Clean up
    solidsql.shutdown()


if __name__ == "__main__":
    main()
