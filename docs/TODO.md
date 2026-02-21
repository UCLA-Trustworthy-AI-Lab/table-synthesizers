# TODO - table-synthesizers

## 2026-02-07 Session Goals

### Completed
- [x] Update docs/ folder documentation (README.md index, CLAUDE.md dev guide)
- [x] Update root README.md (recreated after deletion)
- [x] Create root CLAUDE.md for table-synthesizers project
- [x] Fix tableSynthesizer.py - conditional model registration (NameError on missing deps)
- [x] Fix SMOTE - upgrade imbalanced-learn 0.14.0 -> 0.14.1 (sklearn 1.8 compat)
- [x] Fix config mapping - BN -> BayesianNetwork, add missing TabSyn config
- [x] Add case-insensitive model name aliases in train_all_compatible_models.py
- [x] Convert insights report HTML to PDF
- [x] Split requirements.txt into 4 tiers:
  - [x] requirements.txt (base) - pandas, numpy, sklearn, scipy, no torch
  - [x] requirements-gpu.txt - torch>=2.7 + CUDA SM 12.1 / cu130 deps
  - [x] requirements-cpu.txt - torch>=2.7 CPU-only + LTM deps
  - [x] requirements-synthcity.txt - synthcity>=0.2.12 for ts311 env
- [x] Verify requirements dry-run resolves (base, gpu, cpu all clean)
- [x] Verify all 15 models load correctly after split
- [x] Update docs/README.md with 4-tier install table and commands
- [x] Update docs/CLAUDE.md dependency section with full tier documentation
- [x] Update root README.md installation section with tier table
- [x] Update root CLAUDE.md with requirements tier summary

### Known Issues
- LTM_VAE not available (missing LTM package dependencies - separate project)
- CUDA not available in current env (CPU-only torch installed)
- scipy pinned to 1.11.4 in old requirements but ts311 has 1.17.0 (relaxed to >=1.11.4)
