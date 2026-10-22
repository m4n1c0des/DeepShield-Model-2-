# Publish this repository to GitHub

The ONNX model is about 343 MB, so it must be stored with Git LFS. The target repository is `https://github.com/m4n1c0des/DeepShield-Model-2-.git`.

Run these commands from this repository's root. The GitHub repository should be empty for the first push:

```powershell
git lfs install
git init -b main
git remote add origin https://github.com/m4n1c0des/DeepShield-Model-2-.git
git add .
git status --short
git lfs ls-files
git commit -m "Add DeepShield ViT-B/16 model"
git push -u origin main
```

Before committing, confirm `weights/vit_b16.onnx` appears in `git lfs ls-files` and is not added as a regular Git blob. Never commit `.env`, `.secret_key.dev`, `db.sqlite3`, or training checkpoints such as `last.pt`. Git LFS storage and bandwidth quotas apply; if they are insufficient, publish the weights as a release asset or on a model-hosting service and document the download URL and SHA-256 in `weights/README.md`.
