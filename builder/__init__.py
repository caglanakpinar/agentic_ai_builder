"""Builds what a pipeline runs: agents, their prompts, their tools, and the factory behind all of it.

Nothing is re-exported here on purpose — `builder.agents` reaches every provider SDK through
`models.llms`, so importing it eagerly would make `import builder.tools` pay for it. Import the module
you want (`from builder.factory import build_agent`), or use the `agent_builder` facade, which exports
the same names lazily.
"""
