"""IDEAgent: an agentic quality-diversity search system for research idea generation.

A fixed-budget scheduled search over free generation, repair, and refinement, backed by
a bounded active archive of accepted ideas plus persistent discovery/rejection memory.
See generation_loop.run_search for the entry point (invoked by
scripts/run_ideagent.py).
"""
