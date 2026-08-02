"""Provider callers: one class per LLM provider in `llms`, one per embeddings provider in `embeddings`.

Importing either module constructs nothing, but does import that provider's SDK — build callers through
`builder.factory` (or `agent_builder`) to keep those imports lazy.
"""
