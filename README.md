# Hyperbolic Hierarchy-Aware Embeddings for Version-Conflicted Retrieval in Support Documentation

## Research Question

Enterprise support documents form a natural hierarchy:
product > version > category > issue/article > passage.

When a user asks about the current version, normal semantic retrieval often pulls passages from older versions of the same product. Those passages look almost identical but give different, older instructions, so the final answer mixes two versions and contradicts itself.

Question: do hierarchy aware hyperbolic embeddings reduce these contradictory retrievals compared with normal Euclidean retrieval, and compared with simply filtering by version metadata?

## Status

- Phase 0: Repository skeleton created. Completed.
- Phase 1: Dataset collection and processing. Completed. (13,877 passages, 314 benchmark queries, 666 conflict links).
- Phase 2: Baselines implementation (BM25, Euclidean, Filter, RRF). Completed.
- Phase 3: Hyperbolic model implementation and training (Lorentz model, 384d and 32d). Completed.
- Phase 4: Evaluation, metrics computation, and publication figures. Completed.
- Phase 5: Final report (`docs/REPORT.md`) and presentation inputs (`docs/SLIDE_INPUTS.md`). Completed.
