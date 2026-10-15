"""Per-agent graph nodes.

Each module defines a `<node>_node(state) -> dict` that `otl.graph` imports by name. Until a node is
implemented, `otl.graph` falls back to an inline pass-through stub, so the graph always compiles.
"""
