"""Install a trained model into DeepShield's trained_models/ and merge its entry into models.json.

Usage (from the DeepShield project folder, where manage.py is):
  python install_model.py "C:\\Users\\GC\\Downloads\\vit(mixed data)"
Needs in that folder: <key>.onnx (e.g. vit_b16.onnx) and models_entry_onnx.json.
"""
import sys, json, shutil, hashlib
from pathlib import Path

src = Path(sys.argv[1])
entry_file = src / 'models_entry_onnx.json'
entry = json.loads(entry_file.read_text(encoding='utf-8'))
(key, e), = entry.items()                       # {"vit_b16": {...}}
onnx = src / f'{key}.onnx'
if not onnx.exists():
    cands = list(src.glob('*.onnx'))
    assert len(cands) == 1, f'expected {key}.onnx in {src}, found {[c.name for c in cands]}'
    onnx = cands[0]

tm = Path('trained_models'); assert tm.is_dir(), 'run this from the folder that contains manage.py'
dest = tm / key / 'model.onnx'; dest.parent.mkdir(parents=True, exist_ok=True)
mj = tm / 'models.json'
if mj.exists():
    shutil.copy2(mj, tm / 'models.json.bak')
    cfg = json.loads(mj.read_text(encoding='utf-8'))
else:
    cfg = {}
if dest.exists():
    shutil.copy2(dest, dest.with_name('model_old_ffpp_only.onnx'))   # keep the old FF++-only model as backup
shutil.copy2(onnx, dest)

h = hashlib.sha256(dest.read_bytes()).hexdigest()
if e.get('sha256') and e['sha256'] != h:
    sys.exit(f"SHA mismatch: entry says {e['sha256'][:12]}, file is {h[:12]} (wrong/corrupt download?)")
e['sha256'] = h
e['file'] = f'{key}/model.onnx'
cfg[key] = e
mj.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
print(f'OK  {key}: {dest}  ({dest.stat().st_size/1e6:.0f} MB)  sha256 {h[:12]}...')
print('models.json updated (backup: models.json.bak). Now run: python manage.py check_models')
