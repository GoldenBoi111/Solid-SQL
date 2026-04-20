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

3. **Training Infrastructure**
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
- `schema_linking/skeleton_retriever.py`

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

## Running the System

### For Inference (Already Implemented):
```
python test_schema_linking.py
```

### For Training (Pipeline Ready):
```
# Build dataset
python schema_linking/build_dataset.py --train-json data/train.json --db-dir databases/

# Train model
python schema_linking/train.py
```

## Requirements

- Python 3.8+
- vLLM
- HuggingFace Transformers
- PEFT
- sqlglot
- torch
- numpy

## Next Steps for Minidev

1. Complete the dataset building process
2. Implement proper training data generation
3. Set up the full training pipeline
4. Test with sample data
5. Validate inference quality

## Notes

The current implementation includes most of the core components but needs:
- Proper dataset preparation scripts
- Complete training pipeline verification
- Sample datasets for testing