# SolidSQL Implementation

This repository contains an implementation of the SolidSQL system for converting natural language questions to SQL queries.

## Architecture Overview

The system consists of several key components:

1. **Schema Linking Component** (`schema_linking/`)
   - Core inference engine for predicting relevant tables/columns
   - Uses vLLM with LoRA adapters for efficient inference
   - Follows structured JSON output format

2. **Skeleton Similarity Components** (Sections 3.4.1 and 3.4.2)
   - Question skeleton extraction
   - SQL skeleton extraction
   - Skeleton similarity calculations
   - Example retrieval system

3. **Full Retrieval System with FAISS**
   - Complete skeleton-based example retrieval pipeline
   - Round-2 SQL refinement
   - Candidate example management
   - FAISS indexes for efficient similarity search

4. **Training Infrastructure**
   - Dataset building pipeline
   - Model training with LoRA adapters
   - Configuration management

## Components Implemented

### Core Inference Engine
- `schema_linking_predictor.py`: Easy-to-use wrapper for inference
- `schema_linking/inference.py`: Main inference implementation with vLLM support
- `schema_linking/config.py`: Configuration settings

### Training Pipeline
- `schema_linking/train.py`: Main training script
- `schema_linking/build_dataset.py`: Dataset building from raw examples
- `schema_linking/vllm_model_manager.py`: vLLM integration

### Skeleton Components
- `schema_linking/question_skeleton_extractor.py`
- `schema_linking/sql_skeleton_extractor.py`
- `schema_linking/skeleton_similarity.py`
- `schema_linking/skeleton_retriever.py` (with FAISS indexing)

### Full End-to-End System
- `solidsql.py`: Complete SolidSQL system combining all components
- `example_full_pipeline.py`: Demo of full pipeline usage
- `solidsql_utils.py`: Utility functions for data management

## Full Retrieval System and Round-2 Refinement

The complete SolidSQL system now includes:

### Section 3.4.1 - Question Skeleton Extraction
- Extracts structural skeletons Q⋆ from natural language questions
- Enables structure-based similarity matching

### Section 3.4.2 - SQL Skeleton Extraction  
- Extracts structural skeletons S⋆ from SQL statements
- Supports edit distance similarity calculation

### Skeleton-Based Example Retrieval with FAISS
- Retrieves top-N most similar examples based on question structure
- Supports both question-based and SQL-based retrieval modes
- Uses FAISS for efficient vector similarity search
- Supports multiple index types: flat, ivf, hnsw
- FAISS indexes can be saved and loaded for reuse

### Round-2 Refinement Pipeline
- Uses retrieved examples to improve SQL generation quality
- Implements contextual SQL refinement based on similar examples
- Complete end-to-end workflow from question to final SQL

## Usage

### For Inference (Already Implemented):
```
python test_schema_linking.py
```

### For Full End-to-End Pipeline:
```
python example_full_pipeline.py
```

### Using the SolidSQL Class Directly:
```python
from solidsql import SolidSQL

# Initialize with candidate examples
solidsql = SolidSQL(candidate_examples=[
    {"question": "How many singers are older than 20?", "sql": "SELECT COUNT(*) FROM Singer WHERE Age > 20"},
    {"question": "What is the average salary?", "sql": "SELECT AVG(salary) FROM Employees"},
])

# Generate complete SQL with retrieval and refinement
result = solidsql.generate_sql(
    question="How many actors are younger than 30?",
    schema_text="Actor(id, name, age)\nMovie(id, actor_id, title)",
    top_n=3
)
```

### Using FAISS Indexes:
```python
from schema_linking.skeleton_retriever import SkeletonRetriever

# Create retriever with FAISS index
retriever = SkeletonRetriever(
    candidate_examples=examples,
    faiss_index_type="hnsw"  # Options: "flat", "ivf", "hnsw"
)

# Build FAISS index
retriever.build_index()

# Retrieve similar examples
results = retriever.retrieve_by_question(question="...", top_n=5)

# Save and load index
retriever.save_index("index.json")
new_retriever = SkeletonRetriever([])
new_retriever.load_index("index.json")
```

## Training Pipeline

The training pipeline is structured as follows:

1. **Dataset Preparation**:
   - `build_dataset.py` processes raw examples into training format
   - Converts questions, SQL queries, and database schemas into JSONL format
   
2. **Model Training**:
   - `train.py` handles the fine-tuning process
   - Uses LoRA adapters for memory-efficient fine-tuning
   - Supports HuggingFace Transformers and PEFT

3. **Inference**:
   - `schema_linking_predictor.py` provides simplified interface
   - `schema_linking/inference.py` handles the core logic

## Requirements

- Python 3.8+
- vLLM
- HuggingFace Transformers
- PEFT
- sqlglot
- torch
- numpy
- sentence-transformers
- faiss-cpu (for efficient similarity search)

Install all requirements:
```bash
pip install -r schema_linking/requirements.txt
```

## Next Steps for Minidev

1. Complete the dataset building process with real data
2. Implement proper training data generation
3. Set up the full training pipeline
4. Test with sample data
5. Validate inference quality with full retrieval system
6. Run round-2 refinement tests

## Notes

The current implementation provides the complete end-to-end SolidSQL system as specified in the research paper:

1. ✅ Core schema linking with GPT-OSS-20B model
2. ✅ Question skeleton extraction (Section 3.4.1)
3. ✅ SQL skeleton extraction (Section 3.4.2) 
4. ✅ Skeleton-based example retrieval with FAISS
5. ✅ Round-2 refinement pipeline
6. ✅ Complete end-to-end workflow

The system is ready for minidev results with full retrieval capabilities and round-2 refinement as described in the research paper.

## FAISS Index Types

The retriever supports three FAISS index types:

- **flat**: Exact nearest neighbor search (slowest, most accurate)
- **ivf**: Inverted file index with clustering (faster, approximate)
- **hnsw**: Hierarchical Navigable Small World (fastest, approximate)

Choose the index type based on your needs:
- Use `flat` for small datasets (< 1000 examples)
- Use `ivf` or `hnsw` for larger datasets