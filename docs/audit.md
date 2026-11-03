# Milestone 1 repository audit

Inspected baseline: `b01c6885aca017a511c6074d977d3c32ed91e2f0`. The only remote branch at inspection was `main`. All six baseline commits belong to Uoerim. History proceeds through project setup, dataset/augmentations, scratch encoder, trainer/tracking, Grad-CAM/plots, and CLI verification.

The complete tracked tree had two classification configurations, `requirements.txt`, `README.md`, `.gitignore`, package initializers, `src/cli.py`, `src/data/dataset.py`, `src/models/{base_encoder,scratch_cnn}.py`, `src/training/trainer.py`, `src/visualization/{gradcam,plots}.py`, and `tests/test_pipeline.py`. There were no repository-specific instruction files and no existing notebook or evaluation implementation.

## Ownership and extension points

Yosuf owns the package skeleton, dataset wrapper, image transforms, base/scratch encoder, trainer/MLflow, Grad-CAM, and training plots. Helmy owns text preprocessing and EDA, transfer encoder, grid/random search, evaluation, checkpoints, and packaging. The baseline CLI explicitly discovers tuning and evaluation `register_commands` hooks. The model factory imports the configured transfer module. These interfaces are preserved.

## Findings and changes

| Baseline component | Finding | Resolution |
| --- | --- | --- |
| Data loader | Correct official sources, stratified validation and deterministic workers; no duplicate/corruption audit | Add EDA manifest with dataset fingerprints and explicit training exclusions; leave test membership intact |
| Captions | Simple splitting and punctuation removal existed in wrapper | Reuse a dedicated normalizer with Unicode/entity handling and anomaly reports |
| Augmentations | Default crop could retain only 8% of the image; hue jitter could alter species cues | Restrict retained area to 65–100%, mild color jitter and 10-degree rotations |
| Scratch encoder | Five convolutional stages; reusable spatial features and classifier head | Preserve architecture and initialization |
| Transfer encoder | Configured module absent | Add ResNet-18/34/50 with the same base interface and fixed ImageNet weight versions |
| Trainer | Working AdamW, cosine schedule, validation scores, atomic best/latest checkpoints | Preserve loop; add checkpoint hashes/references, persistent run ID, strict tracking option and overwrite protection |
| Grad-CAM | Works for scratch/frozen features; pretrained ResNet uses in-place activations incompatible with backward hooks | Disable in-place activations inside the new transfer encoder |
| Evaluation/tuning | CLI extension hooks existed; implementations absent | Add validation-only searches, full-split evaluation, guarded comparisons and published benchmark |
| Original tests | Expected missing transfer module | Move missing-module assertion to an unknown model; add real transfer and CLI subprocess checks |

Work is limited to classification, encoder training and the required caption cleaning.
