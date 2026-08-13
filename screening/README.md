# Seed-242 applicability-domain screening module

This module contains the complete 4,295-unique-structure library, its 11,654 source rows, the seed-242 recomputation script, and the derived result tables.

The screening route is independent of neural-weight training:

1. Rebuild Morgan radius-2 fingerprints with 3,147 bits.
2. Fit the feature-ranking Random Forest on the 1,240-record training split only.
3. Retain the top 914 fingerprint bits.
4. Use training-set leave-one-out nearest-neighbour distances to define q95 strict and q99 borderline thresholds.
5. Flag pH outside the training range `[1.0, 12.0]` and inorganic/no-carbon structures separately.

Run from the release root:

```powershell
pwsh -File .\screening\scripts\run_ad_screening.ps1 -Python python
```

The included result has 3,423/4,295 unique structures in AD (79.7%).
