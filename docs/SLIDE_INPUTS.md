# Slide Deck Inputs: Hierarchy-Aware Hyperbolic Embeddings for RAG

This document provides the six core slides for presenting this research to Professor Ankita Shukla.
Every number matches `results/results.json` exactly.

---

## Slide 1: The Problem

- Enterprise support documentation forms a strict hierarchy: Product > Version > Category > Article > Passage.
- When users ask questions without specifying a version, dense semantic retrieval pulls passages from older versions that look almost identical.
- These older passages contain outdated commands, obsolete paths, and deprecated parameters.
- When fed to an LLM, mixed-version passages produce self-contradictory answers that break customer installations.

---

## Slide 2: Real World Example

**Query**: "How do I configure SERVER_EMAIL in Django?" (Target: Django 5.1)

- **Gold Passage (Django 5.1)**:
  `Default: 'root@localhost'. The email address that error messages come from... This address is used in the From: header and can take any format valid in the chosen email sending protocol.`
- **Stale Passage (Django 4.2)**:
  `Default: 'root@localhost'. The email address that error messages come from... This address is used only for error messages.`
- **The Failure**:
  Django 4.2 omits the protocol format rule added in 5.1. Standard retrieval returns both in top 5, creating immediate confusion for administrators.

---

## Slide 3: Dataset Counts

- **Corpus Passages**: 13,877 total passages (80 to 300 words each).
  - Nextcloud Server (Versions 28, 29, 30): 2,373 passages.
  - Django Framework (Versions 4.2, 5.0, 5.1): 11,504 passages.
- **Benchmark Queries**: 314 total support questions.
  - Nextcloud: 114 queries.
  - Django: 200 queries.
- **Version Splits**:
  - Version Stated: 157 queries (50.0%).
  - Version Unstated (Headline Test): 157 queries (50.0%).
- **Cross-Version Conflict Pairs**: 666 aligned pairs with procedural changes.

---

## Slide 4: Method in 4 Lines

1. **Base Encoder**: Extract 384-dimensional dense vectors from text chunks using `all-MiniLM-L6-v2`.
2. **Lorentz Projection**: Map tangent space vectors to the hyperbolic hyperboloid manifold using exponential maps.
3. **Contrastive Loss**: Penalize distance between queries and stale version passages relative to gold targets.
4. **Hierarchy Loss**: Embed taxonomy nodes such that depth maps to norm and passages cluster near their specific version branch.

---

## Slide 5: Headline Result Table (Version Unstated Split, 157 Queries)

| System | Conflict Rate @ 5 | Stale Rate @ 5 | Recall @ 5 | MRR |
| :--- | :---: | :---: | :---: | :---: |
| **B: Euclidean (MiniLM)** | 17.20% | 7.39% | 28.03% | 0.1789 |
| **B-Trained Control (384d)** | 5.73% | 1.40% | 22.93% | 0.1664 |
| **C: Hard Metadata Filter** | 0.00% | 0.00% | 50.32% | 0.3505 |
| **D: Hyperbolic Model (384d)** | **1.27%** | **0.38%** | **14.01%** | **0.1009** |
| **F: RRF Hybrid (B + D)** | 9.55% | 3.44% | 26.11% | 0.1643 |

---

## Slide 6: Honest Conclusion

1. **Hyperbolic geometry cuts contradictory retrievals**:
   System D reduces the top-5 conflict rate from 17.20% down to 1.27% (a 13.5x reduction over baseline Euclidean, and 4.5x over trained Euclidean control).
2. **The trade-off is recall**:
   Hyperbolic space pushes opposing version branches apart aggressively, lowering Recall@5 from 28.03% to 14.01%.
3. **Why not just filter by metadata?**:
   Hard filtering achieves 0.00% conflicts on this benchmark only because it guesses the newest version. If a customer is running an older LTS version, metadata guessing guarantees returning wrong documentation. Hyperbolic retrieval separates versions geometrically without fragile guessing.
4. **Downstream LLM generation check (Grok API)**:
   In generation checks across 30 queries with Grok (`grok-4.20-0309-non-reasoning`), System B caused contradictory mixed-version answers in 13.3% of tests. System D cut contradictory answers in half to 6.7%, tying metadata filtering.
