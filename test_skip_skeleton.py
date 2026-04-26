#!/usr/bin/env python3
"""
Test script to verify that SolidSQL can skip question skeleton extraction.
"""

from solidsql import SolidSQL

def test_skip_skeleton_extraction():
    """Test that SolidSQL can be initialized with skeleton extraction skipped."""
    print("Testing SolidSQL with skip_skeleton_extraction=True...")
    
    # Initialize with skeleton extraction skipped
    solidsql = SolidSQL(
        candidate_examples=[],  # Empty since we'll load from indices
        build_index=False,
        skip_skeleton_extraction=True
    )
    
    # Check that q_extractor is None
    assert solidsql.q_extractor is None, "q_extractor should be None when skip_skeleton_extraction=True"
    print("✓ q_extractor is None as expected")
    
    # Test shutdown
    solidsql.shutdown()
    print("✓ shutdown completed without errors")
    
    print("Test passed!")

if __name__ == "__main__":
    test_skip_skeleton_extraction()