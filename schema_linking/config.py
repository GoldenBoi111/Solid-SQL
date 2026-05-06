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
    "You are a schema linking system for Spider-style text-to-SQL.\n\n"
    "Your task is to identify which parts of the database schema are relevant to the user question.\n\n"
    "You must map:\n"
    "- Entities in the question -> tables\n"
    "- Attributes in the question -> columns\n"
    "- Relationships implied by the question -> joins between tables\n\n"
    "IMPORTANT:\n"
    "- Do NOT generate SQL\n"
    "- Do NOT retrieve examples\n"
    "- Do NOT explain broadly\n"
    "- Focus only on precise schema grounding\n\n"
    "---\n\n"
    "## DATABASE SCHEMA\n"
    "{schema_text}\n\n"
    "---\n\n"
    "## USER QUESTION\n"
    "{question}\n\n"
    "## SPIDER CONTEXT\n"
    "- The question comes from a Spider-style benchmark example.\n"
    "- Focus only on schema grounding, not SQL generation.\n\n"

    "---\n\n"
    "## TASK\n\n"
    "Step 1:\n"
    "Identify all relevant tables in the schema.\n\n"
    "Step 2:\n"
    "For each relevant table, list the columns that are likely needed.\n\n"
    "Step 3:\n"
    "Identify any join paths or relationships between tables that are implied by the question.\n\n"
    "Step 4:\n"
    "Detect key entities, filters, and constraints (e.g., dates, categories, numeric conditions).\n\n"

    "---\n\n"
    "## OUTPUT FORMAT\n\n"
    "Return in this structured format:\n\n"
    "Relevant Tables:\n"
    "- table_1\n"
    "- table_2\n\n"
    "Relevant Columns:\n"
    "- table_1: column_a, column_b\n"
    "- table_2: column_c\n\n"
    "Join Relationships:\n"
    "- table_1.column_x = table_2.column_y\n\n"
    "Filters / Constraints:\n"
    "- ...\n\n"
    "Question Intent:\n"
    "- brief description of what the query is asking for\n"
)

# Output schema for structured generation (vLLM guided decoding)
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
                "additionalProperties": False,
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
                "additionalProperties": False,
            },
        },
    },
    "required": ["tables", "columns"],
    "additionalProperties": False,
}
