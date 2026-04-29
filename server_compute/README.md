# Server compute

Contains only the language-model side of the split:

- `language_model.py`: LLaVA-OneVision text generation for operator-facing summaries.
- YOLO metrics remain ground truth. The language model summarizes; it does not count.
- There is no local HTTP server layer anymore. The edge module calls this file directly for local testing.
