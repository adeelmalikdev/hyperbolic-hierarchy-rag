# Hyperbolic Hierarchy-Aware Embeddings for Version-Conflicted Retrieval in Support Documentation

## Research Question

Enterprise support documents form a natural hierarchy:
product > version > category > issue/article > passage.

When a user asks about the current version, normal semantic retrieval often pulls passages from older versions of the same product. Those passages look almost identical but give different, older instructions, so the final answer mixes two versions and contradicts itself.

Question: do hierarchy aware hyperbolic embeddings reduce these contradictory retrievals compared with normal Euclidean retrieval, and compared with simply filtering by version metadata?

## Status

- Phase 0: Repository skeleton created. Active.
- Phase 1: Dataset collection and processing (pending).
- Phase 2: Baselines implementation (pending).
- Phase 3: Hyperbolic model implementation and training (pending).
- Phase 4: Evaluation and metrics computation (pending).
- Phase 5: Final report and presentation inputs (pending).
