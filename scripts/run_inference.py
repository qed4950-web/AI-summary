"""Compatibility wrapper for `scripts.pipeline.run_inference`.

This keeps the historical entry point available while the actual logic lives
under `scripts/pipeline/`. See that module for full documentation.
"""

from scripts.pipeline.run_inference import main


if __name__ == "__main__":
    main()
