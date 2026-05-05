"""
Compatibility entrypoint for schema-linking training.

Use the stable single-GPU trainer implementation from `train_stable.py`.
"""

from train_stable import main


if __name__ == "__main__":
    main()
