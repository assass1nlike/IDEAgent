"""IDEAgent: an agentic quality-diversity search system for research idea generation.

A fixed-budget scheduled search over free generation, repair, and refinement, backed by
a bounded active archive of accepted ideas plus persistent discovery/rejection memory.
See yield_loop.run_yield_search for the entry point (invoked by
scripts/run_yield_archive.py).
"""
