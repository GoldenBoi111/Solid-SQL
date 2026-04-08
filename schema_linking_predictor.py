"""
Schema Linking Predictor

Thin wrapper around the schema_linking.inference module for convenient
top-level imports.

Usage:
    from schema_linking_predictor import SchemaLinkingPredictor

    predictor = SchemaLinkingPredictor(
        base_model="openai/gpt-oss-20b",
        adapter_path="./schema_linking_output/lora_adapter",
    )

    # Single prediction
    result = predictor.predict(
        question="How many singers are older than 20?",
        schema_text="Singer(id, name, age)\nAlbum(id, singer_id, title)",
    )

    # With automatic schema loading from SQLite
    result = predictor.predict_from_db(
        question="How many singers are older than 20?",
        db_id="concert_singer",
        db_root="./databases",
    )
"""

import sys
from pathlib import Path

# Ensure schema_linking package is importable
sys.path.insert(0, str(Path(__file__).parent / "schema_linking"))

from inference import SchemaLinker as SchemaLinkingPredictor

__all__ = ["SchemaLinkingPredictor"]
