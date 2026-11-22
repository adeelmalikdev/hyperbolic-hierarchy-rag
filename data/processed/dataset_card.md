# Dataset Card: Version-Conflicted Support Documentation Benchmark

## Overview

This dataset evaluates retrieval systems on technical documentation with cross-version conflicts.
Enterprise support knowledge bases often contain multiple versions of the same product documentation.
When users ask questions about the current version, standard semantic retrieval can pull older passages.
Those older passages look semantically identical to newer ones but contain outdated steps, settings, or commands.

## Corpus Counts

### Nextcloud Server Documentation
- Source: github.com/nextcloud/documentation (admin manual and user manual)
- Version 28 (stable28): 736 passages
- Version 29 (stable29): 786 passages
- Version 30 (stable30, target): 851 passages
- Total Nextcloud passages: 2,373

### Django Documentation
- Source: github.com/django/django docs folder
- Version 4.2 (stable/4.2.x): 3,800 passages
- Version 5.0 (stable/5.0.x): 3,826 passages
- Version 5.1 (stable/5.1.x, target): 3,878 passages
- Total Django passages: 11,504

### Overall Corpus
- Total passages across all products and versions: 13,877
- Passage length policy: bounded between 80 and 300 words per chunk.
- Hierarchy per passage: Product > Version > Category > Article > Passage.

## Benchmark Queries and Conflict Pairs

- Total test queries: 314
- Nextcloud queries: 114
- Django queries: 200
- Version stated queries: 157 (exactly 50%)
- Version unstated queries: 157 (exactly 50%)
- Total stale conflict links: 666 (216 Nextcloud, 450 Django)

Every query targets a changed section in the newest version of the product (Nextcloud 30 or Django 5.1).
The gold passage is the passage from the target version.
The stale passages are older passages for that same issue from prior versions.

## Staleness Decision Criteria

Staleness was determined through structured cross-version alignment:
1. Alignment by document path and section heading across product versions.
2. Token overlap filter requiring Jaccard similarity >= 0.25 to confirm identical subject matter.
3. Procedural change detection based on:
   - Differences in numeric parameters and version requirements.
   - Differences in CLI commands, configuration directives, and code identifiers.
   - Substantive textual revisions with edit distance below 0.90 ratio.
If a section is identical between versions, it is not marked as a conflict because identical instructions cannot produce contradictory answers.

## Limitations

1. Public documentation vs private support bases:
   Open source documentation models enterprise support content well, but enterprise knowledge bases also contain issue tickets, forum posts, and internal runbooks.
2. Template variety:
   Queries are synthesized with support question templates rather than collected from live end user telemetry.
3. Granularity:
   Passages are segmented by headings and paragraph boundaries. Highly granular line level changes within a single table might be embedded with surrounding context.
