"""
SolidSQL System Test

Test that the complete SolidSQL system works correctly with all components.
"""

import json
from solidsql import SolidSQL


def test_solidsql_system():
    """Test the complete SolidSQL system functionality."""

    print("Testing SolidSQL System")
    print("=" * 30)

    # Sample candidate examples
    examples = [
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
    ]

    # Test 1: Initialize system
    print("1. Testing system initialization...")
    solidsql = SolidSQL(candidate_examples=examples, build_index=True)
    print("   ✓ System initialized successfully")

    # Test 2: Generate SQL
    print("2. Testing SQL generation...")
    result = solidsql.generate_sql(
        question="How many actors are younger than 30?",
        schema_text="Actor(id, name, age)\nMovie(id, actor_id, title)",
        top_n=2,
    )
    print("   ✓ SQL generation completed")

    # Test 3: Check result structure
    print("3. Validating result structure...")
    assert "question" in result
    assert "schema" in result
    assert "schema_linking_result" in result
    assert "question_skeleton" in result
    assert "retrieval_results" in result
    print("   ✓ All expected result fields present")

    # Test 4: Check schema linking result
    schema_result = result["schema_linking_result"]
    assert "tables" in schema_result
    assert "columns" in schema_result
    print("   ✓ Schema linking result structure valid")

    # Test 5: Check retrieval results
    retrieval_results = result["retrieval_results"]
    print(f"   ✓ Found {len(retrieval_results)} retrieval results")

    # Test 6: Test adding more examples
    print("4. Testing candidate addition...")
    new_examples = [
        {
            "question": "List the names of students who scored above 90",
            "sql": "SELECT Name FROM Student WHERE Score > 90",
            "db_id": "student_records",
        }
    ]

    solidsql.add_candidate_examples(new_examples)
    print("   ✓ Added new candidate examples")

    # Test 7: Test rebuild index
    print("5. Testing index rebuilding...")
    solidsql.build_retrieval_index(show_progress=False)
    print("   ✓ Index rebuilt successfully")

    # Test 8: Test with context examples
    print("6. Testing context-based generation...")
    context_result = solidsql.generate_sql_with_context(
        question="How many movies are there?",
        schema_text="Movie(id, title, year)\nActor(id, name)",
        context_examples=new_examples,
    )
    print("   ✓ Context-based generation completed")

    # Cleanup
    solidsql.shutdown()

    print("\n✓ All tests passed! SolidSQL system is working correctly.")
    return True


if __name__ == "__main__":
    test_solidsql_system()
