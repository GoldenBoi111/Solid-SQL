"""
Centralized configuration for schema linking training pipeline.
"""

# ============================================================
# Model Configuration
# ============================================================
MODEL_NAME = "openai/gpt-oss-20b"  # Hugging Face model to fine-tune
BASE_MODEL_PATH = ""  # Optional: override with local path
OUTPUT_DIR = "./schema_linking_output"
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03
LR_SCHEDULER_TYPE = "cosine"
FP16 = False
BF16 = True
MAX_SEQ_LENGTH = 2048
GRADIENT_CHECKPOINTING = True

# LoRA Configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]

# ============================================================
# Data Configuration
# ============================================================
RAW_DATA_DIR = "../data"  # Directory containing raw dataset JSON files
DB_ROOT = "../databases"  # Root directory containing SQLite databases
OUTPUT_TRAIN_PATH = "./train.jsonl"
OUTPUT_VAL_PATH = "./val.jsonl"
VAL_SPLIT_RATIO = 0.1  # Fraction of data to use for validation

# ============================================================
# SQL Parser Configuration
# ============================================================
SQL_DIALECT = "sqlite"

# ============================================================
# Prompt Templates
# ============================================================
INSTRUCTION_TEMPLATE = (
    "Given a natural language question and a database schema, "
    "identify which tables and columns are relevant to answering the question.\n\n"
    "Return your answer as a JSON object with this exact structure:\n"
    "{{\n"
    '    "tables": [\n'
    '        {{"name": "table_name", "reason": "why this table is relevant"}}\n'
    "    ],\n"
    '    "columns": [\n'
    '        {{"name": "TableName.column_name", "reason": "why this column is relevant"}}\n'
    "    ]\n"
    "}}\n\n"
    "Rules:\n"
    "- Include every table/column used in SELECT, FROM, JOIN, WHERE, GROUP BY, ORDER BY, HAVING\n"
    "- Only include tables/columns that are actually needed\n"
    "- Each reason should be a brief explanation referencing the question\n"
    "- Do NOT include any text outside of the JSON object\n\n"
    "Question:\n{question}\n\n"
    "Database Schema:\n{schema_text}"
)

# Output schema for structured generation
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "tables": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["name", "reason"],
            },
        },
        "columns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["name", "reason"],
            },
        },
    },
    "required": ["tables", "columns"],
}
