"""
Test Schema Linking After Training

Loads the fine-tuned LoRA adapter and runs predictions on validation examples
to inspect model performance qualitatively.
"""

import json
from pathlib import Path
from schema_linking.inference import SchemaLinker

# ============================================================
# Configuration
# ============================================================
BASE_MODEL = "openai/gpt-oss-20b"
ADAPTER_PATH = "./schema_linking_output/lora_adapter"
VAL_JSONL = "./val.jsonl"
NUM_SAMPLES = 10


def extract_question_schema(input_text: str) -> dict:
    """
    Extract question and schema_text from the formatted input string.
    Expected format:
    "Given a natural language question...\n\nQuestion:\n{question}\n\nDatabase Schema:\n{schema_text}"
    """
    # Split on "Database Schema:\n" first
    schema_split = input_text.split("Database Schema:\n")
    schema_text = schema_split[1] if len(schema_split) > 1 else ""

    # Split the first part on "Question:\n"
    question_split = schema_split[0].split("Question:\n")
    question = question_split[1].strip() if len(question_split) > 1 else ""

    return {"question": question, "schema_text": schema_text}


def main():
    print("=" * 60)
    print("Testing Schema Linking Model")
    print("=" * 60)

    # Load model
    linker = SchemaLinker(
        base_model=BASE_MODEL,
        adapter_path=ADAPTER_PATH,
    )

    # Load validation set
    val_path = Path(VAL_JSONL)
    if not val_path.exists():
        print(f"Error: {VAL_JSONL} not found. Run build_dataset.py first.")
        return

    print(f"\nLoading validation data from {VAL_JSONL}...")
    val_data = []
    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                val_data.append(json.loads(line))

    print(f"Loaded {len(val_data)} validation examples")

    # Prepare inputs
    limit = min(NUM_SAMPLES, len(val_data))
    inputs = [extract_question_schema(e["input"]) for e in val_data[:limit]]
    ground_truths = [e["output"] for e in val_data[:limit]]

    print(f"\nRunning predictions on first {limit} examples...")
    print("=" * 60)

    results = linker.predict_batch(inputs, show_progress=True)

    # Display results
    for i, (inp, prediction, gt_str) in enumerate(zip(inputs, results, ground_truths)):
        print(f"\n{'='*60}")
        print(f"Example {i + 1}")
        print(f"{'='*60}")
        print(f"Question: {inp['question']}")
        print(f"\nSchema: {inp['schema_text'][:200]}...")

        try:
            gt = json.loads(gt_str)
        except json.JSONDecodeError:
            gt = {"error": "Failed to parse ground truth"}

        print(f"\nPredicted tables: {json.dumps(prediction.get('tables', []), indent=2)}")
        print(f"Ground truth tables: {json.dumps(gt.get('tables', []), indent=2)}")

        print(f"\nPredicted columns: {json.dumps(prediction.get('columns', []), indent=2)}")
        print(f"Ground truth columns: {json.dumps(gt.get('columns', []), indent=2)}")

        if "error" in prediction:
            print(f"\n⚠ Model error: {prediction.get('error')}")

    print(f"\n{'='*60}")
    print(f"Done. Tested {limit} examples.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
