"""DeepShield model demo: upload images in the browser and see the verdict. Standalone, no project/Django needed.

  pip install gradio onnxruntime opencv-python numpy pillow
  python demo_app.py --model efficientnet_b4.onnx --entry models_entry_onnx.json
then open http://127.0.0.1:7860
"""
import argparse, json, os
import numpy as np, cv2, onnxruntime as ort, gradio as gr

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True); ap.add_argument('--entry', required=True)
ap.add_argument('--title', default='DeepShield: model demo'); ap.add_argument('--port', type=int, default=7860)
a = ap.parse_args()

e = json.load(open(a.entry)); e = e if 'input_size' in e else next(iter(e.values()))
SIZE, NORM = int(e['input_size']), e.get('normalization', 'imagenet')
MODE, FAKE, THR0 = e.get('output_mode', 'logits'), e.get('fake_index', 1), float(e.get('threshold', 0.5))
INTERP = {'bilinear': cv2.INTER_LINEAR, 'bicubic': cv2.INTER_CUBIC, 'area': cv2.INTER_AREA}.get(e.get('interpolation', 'bilinear'), cv2.INTER_LINEAR)
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
sess = ort.InferenceSession(a.model, providers=['CPUExecutionProvider']); inp = sess.get_inputs()[0].name
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def p_fake(rgb):
    x = cv2.resize(rgb, (SIZE, SIZE), interpolation=INTERP).astype(np.float32)
    x = x / 255.0 if NORM == 'zero_one' else x / 127.5 - 1.0 if NORM == 'minus_one_one' else (x / 255.0 - MEAN) / STD if NORM == 'imagenet' else x
    out = np.asarray(sess.run(None, {inp: np.transpose(x, (2, 0, 1))[None]})[0], np.float64).ravel()
    if MODE == 'sigmoid': return float(out[0])
    if MODE == 'logit': return float(1 / (1 + np.exp(-out[0])))
    pr = out if MODE == 'softmax' else np.exp(out - out.max()) / np.exp(out - out.max()).sum()
    return float(pr[FAKE])

def analyse(rgb, use_face=True):
    """returns (p_fake, shown_image, note). Crops the largest face when found, else uses the whole image."""
    rgb = np.ascontiguousarray(rgb[..., :3]); note, shown, crop = '', rgb, rgb
    if use_face:
        fs = cascade.detectMultiScale(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 1.1, 5, minSize=(40, 40))
        if len(fs):
            x, y, w, h = max(fs, key=lambda f: f[2] * f[3]); p = int(0.3 * max(w, h)); H, W = rgb.shape[:2]
            crop = rgb[max(0, y - p):min(H, y + h + p), max(0, x - p):min(W, x + w + p)]
            shown = rgb.copy(); cv2.rectangle(shown, (x, y), (x + w, y + h), (0, 200, 0), max(2, W // 200))
        else:
            note = 'no face found: whole image used'
    return p_fake(crop), shown, note

def verdict(p, thr): return ('FAKE / manipulated' if p >= thr else 'REAL / authentic')

def single(img, thr, use_face):
    if img is None: return None, 'Upload an image first.'
    p, shown, note = analyse(img, use_face)
    return shown, f'{verdict(p, thr)}\nfake probability: {p:.1%}   (threshold {thr:.2f}){chr(10) + note if note else ""}'

def batch(files, thr, use_face):
    rows, gallery = [], []
    for f in files or []:
        path = f if isinstance(f, str) else getattr(f, 'name', None) or f.get('path')
        bgr = cv2.imread(path)
        if bgr is None: continue
        p, shown, note = analyse(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), use_face)
        v = 'FAKE' if p >= thr else 'real'
        rows.append([os.path.basename(path), round(p, 3), v, note]); gallery.append((shown, f'{os.path.basename(path)}: {v} ({p:.0%})'))
    n_fake = sum(r[2] == 'FAKE' for r in rows)
    return rows, gallery, f'{len(rows)} images | flagged FAKE: {n_fake} | real: {len(rows) - n_fake}'

with gr.Blocks(title=a.title) as demo:
    gr.Markdown(f'# {a.title}\nModel: `{os.path.basename(a.model)}`. Score = fake probability. Face-swap / deepfake faces are what it was trained on; fully AI-generated scenes may not be detected.')
    with gr.Row():
        thr = gr.Slider(0.05, 0.95, value=round(THR0, 2), step=0.01, label='Decision threshold (fake if score >= this)')
        use_face = gr.Checkbox(value=True, label='Detect and crop face first')
    with gr.Tab('One image'):
        with gr.Row():
            i1 = gr.Image(type='numpy', label='Image'); o1 = gr.Image(label='Analysed (green box = face used)')
        t1 = gr.Textbox(label='Result', lines=3); gr.Button('Analyse', variant='primary').click(single, [i1, thr, use_face], [o1, t1])
    with gr.Tab('Many images'):
        fi = gr.File(file_count='multiple', file_types=['image'], label='Select many images')
        s2 = gr.Textbox(label='Summary')
        tb = gr.Dataframe(headers=['image', 'p_fake', 'verdict', 'note'], label='Scores')
        ga = gr.Gallery(label='Images', columns=4, height=420)
        gr.Button('Analyse all', variant='primary').click(batch, [fi, thr, use_face], [tb, ga, s2])
demo.launch(server_name='127.0.0.1', server_port=a.port)
