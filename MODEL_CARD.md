# Model Card: DeepShield ViT-B/16 (v2.0.0, mixed data)

| | |
|---|---|
| Architecture | Vision Transformer, `timm vit_base_patch16_224`, ImageNet-pretrained, 2-class head |
| Input | 224x224 RGB face crop, normalization mean/std 0.5 (`minus_one_one`), bilinear resize |
| Output | 2 logits, index 1 = fake |
| Threshold | 0.763 (Youden J on validation set, not on test) |
| Training data | FaceForensics++ C23 faces + 140k real/fake faces + Stable Diffusion faces (identity-disjoint 70/15/15 split) |
| Training | 6 epochs, batch 32, AdamW, lr 2e-5 backbone / 3e-4 head, label smoothing 0.05, seed 42, AMP |
| Test (same distribution) | AUC 0.9858, balanced accuracy 0.9432, EER 0.0587 |
| Format | ONNX (`vit_b16.onnx`), run with ONNX Runtime on CPU |

## Intended use
Face-image forensic assistant: gives a *probability-style score* that a face image is manipulated or AI-generated, as one signal inside DeepShield's ensemble. For education / FYP demonstration.

## Limitations
- Scores are same-distribution; **cross-dataset (Celeb-DF) performance is lower and must be reported separately** (see `results/`).
- Frames of one video are correlated; the effective sample size is videos, not images.
- Needs a visible face. Heavy compression, tiny faces, or new generators can fool it.
- Not proof of authenticity or fakery, and not for legal decisions.
