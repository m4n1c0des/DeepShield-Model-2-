"""Standalone image test for a DeepShield ONNX model. No project folder, no Django, no website.

  pip install onnxruntime opencv-python numpy pillow scikit-learn
  python test_model_standalone.py --model efficientnet_b4.onnx --entry models_entry_onnx.json --images test_images

test_images/ may contain real/ and fake/ sub-folders (then accuracy + AUC are printed) or just images.
Settings (input size, normalization, fake_index, threshold) are read from the models_entry_*.json written by training.
Faces are found with OpenCV's built-in face detector and cropped (training used face crops); --no-detect skips this.
"""
import argparse, csv, json, os, sys
import numpy as np, cv2, onnxruntime as ort

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True, help='path to the .onnx file')
ap.add_argument('--entry', required=True, help='models_entry_onnx.json next to the model')
ap.add_argument('--images', required=True)
ap.add_argument('--threshold', type=float, default=None, help='override the threshold from the entry file')
ap.add_argument('--no-detect', action='store_true', help='images are already face crops')
ap.add_argument('--pad', type=float, default=0.3, help='extra margin around the detected face (fraction)')
a = ap.parse_args()

e = json.load(open(a.entry)); e = next(iter(e.values())) if 'input_size' not in e else e
SIZE, NORM = int(e['input_size']), e.get('normalization', 'imagenet')
MODE, FAKE = e.get('output_mode', 'logits'), e.get('fake_index', 1)
THR = a.threshold if a.threshold is not None else float(e.get('threshold', 0.5))
INTERP = {'bilinear': cv2.INTER_LINEAR, 'bicubic': cv2.INTER_CUBIC, 'area': cv2.INTER_AREA}.get(e.get('interpolation', 'bilinear'), cv2.INTER_LINEAR)
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
sess = ort.InferenceSession(a.model, providers=['CPUExecutionProvider']); inp = sess.get_inputs()[0].name
print(f'model {os.path.basename(a.model)} | size {SIZE} | norm {NORM} | output {MODE} fake_index {FAKE} | threshold {THR}\n')

def prep(rgb):
    x = cv2.resize(rgb, (SIZE, SIZE), interpolation=INTERP).astype(np.float32)
    x = x / 255.0 if NORM == 'zero_one' else x / 127.5 - 1.0 if NORM == 'minus_one_one' else (x / 255.0 - MEAN) / STD if NORM == 'imagenet' else x
    return np.transpose(x, (2, 0, 1))[None]

def p_fake(rgb):
    out = np.asarray(sess.run(None, {inp: prep(rgb)})[0], np.float64).ravel()
    if MODE == 'sigmoid': return float(out[0])
    if MODE == 'logit': return float(1 / (1 + np.exp(-out[0])))
    pr = out if MODE == 'softmax' else np.exp(out - out.max()) / np.exp(out - out.max()).sum()
    return float(pr[FAKE])

cascade = None if a.no_detect else cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
def crop_face(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    fs = cascade.detectMultiScale(g, 1.1, 5, minSize=(40, 40))
    if len(fs) == 0: return None
    x, y, w, h = max(fs, key=lambda f: f[2] * f[3]); p = int(a.pad * max(w, h))
    H, W = rgb.shape[:2]
    return rgb[max(0, y - p):min(H, y + h + p), max(0, x - p):min(W, x + w + p)]

EXT = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
files = []
for root, _, names in os.walk(a.images):
    part = os.path.relpath(root, a.images).replace('\\', '/').lower().split('/')
    lab = 1 if 'fake' in part else 0 if 'real' in part else None
    files += [(os.path.join(root, n), lab) for n in sorted(names) if os.path.splitext(n)[1].lower() in EXT]
if not files: sys.exit('No images found.')

rows, skipped = [], []
print('%-36s %-6s %-8s %s' % ('image', 'truth', 'p_fake', 'verdict'))
for path, lab in files:
    bgr = cv2.imread(path)
    if bgr is None: skipped.append(os.path.basename(path)); continue
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if cascade is not None:
        rgb = crop_face(rgb)
        if rgb is None: skipped.append(os.path.basename(path) + ' (no face)'); continue
    p = p_fake(rgb); rows.append((path, lab, p))
    print('%-36s %-6s %-8.3f %s' % (os.path.basename(path)[:36], {1: 'fake', 0: 'real', None: '-'}[lab], p, 'FAKE' if p >= THR else 'real'))

with open(os.path.join(a.images, 'scores.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['image', 'truth', 'p_fake'])
    for path, lab, p in rows: w.writerow([os.path.relpath(path, a.images), lab, round(p, 4)])
if skipped: print('\nSkipped:', skipped, '(try --no-detect if these are already face crops)')
lr = [r for r in rows if r[1] is not None]; fk = [r for r in lr if r[1] == 1]; rl = [r for r in lr if r[1] == 0]
if lr:
    print(f'\nthreshold {THR} | fake caught {sum(r[2] >= THR for r in fk)}/{len(fk)} | real wrongly flagged {sum(r[2] >= THR for r in rl)}/{len(rl)}', end='')
    if fk and rl:
        from sklearn.metrics import roc_auc_score
        print(' | AUC %.3f' % roc_auc_score([r[1] for r in lr], [r[2] for r in lr]), end='')
    print()
