import base64
import gc
import io
import threading
import webbrowser
from pathlib import Path

import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, render_template, request
from PIL import Image
from werkzeug.utils import secure_filename

CLASSES   = ['51', '52', '53', '54', '55']
NORM_MEAN = 0.1307
NORM_STD  = 0.3081

BASE_DIR        = Path(__file__).parent
MODELS_DIR      = BASE_DIR / 'models'
MODELS_DIR.mkdir(exist_ok=True)
ACTIVE_FILE     = MODELS_DIR / 'active.txt'
SUPPORTED_EXTS  = ('.onnx',)

DATA_DIR = BASE_DIR.parent / 'data' / 'dataset'
DATA_DIR.mkdir(parents=True, exist_ok=True)
for _cls in CLASSES:
    (DATA_DIR / _cls).mkdir(exist_ok=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

TTA_VARIANTS = [
    dict(angle=0,  translate=(0, 0),  scale=1.00),
    dict(angle=-4, translate=(0, 0),  scale=1.00),
    dict(angle=4,  translate=(0, 0),  scale=1.00),
    dict(angle=0,  translate=(0, 0),  scale=0.92),
    dict(angle=0,  translate=(0, 0),  scale=1.08),
    dict(angle=0,  translate=(-4, 0), scale=1.00),
    dict(angle=0,  translate=(4, 0),  scale=1.00),
    dict(angle=-3, translate=(0, 4),  scale=0.95),
    dict(angle=3,  translate=(0, -4), scale=1.05),
]


_lock  = threading.Lock()
_state = {
    'model':      None,
    'kind':       None,
    'classes':    CLASSES,
    'filename':   None,
    'input_type': 'raw',
}


# ---------------- preprocessing ----------------

def pil_affine(img, angle=0, translate=(0, 0), scale=1.0, fill=255):
    w, h = img.size
    img = img.rotate(angle, fillcolor=fill, resample=Image.BILINEAR)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    result = Image.new('L', (w, h), fill)
    ox = (w - new_w) // 2 + translate[0]
    oy = (h - new_h) // 2 + translate[1]
    result.paste(img, (ox, oy))
    return result


def preprocess_cnn(pil_img):
    arr = np.array(pil_img.convert('L').resize((28, 28), Image.BILINEAR), dtype=np.float32)
    arr = (arr / 255.0 - NORM_MEAN) / NORM_STD
    return arr[np.newaxis, np.newaxis, :, :]  # (1, 1, 28, 28)


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ---------------- model helpers ----------------

def load_onnx(path: Path):
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    input_names = [i.name for i in session.get_inputs()]
    kind = 'rf' if 'float_input' in input_names else 'cnn'
    return session, kind, CLASSES


def load_any(path: Path):
    ext = path.suffix.lower()
    if ext == '.onnx':
        session, kind, classes = load_onnx(path)
        return session, kind, classes, 'raw'
    raise ValueError(f'Unsupported model extension: {ext}')


def set_active(model, kind, classes, filename, input_type='raw'):
    with _lock:
        _state['model']      = model
        _state['kind']       = kind
        _state['classes']    = classes
        _state['filename']   = filename
        _state['input_type'] = input_type


def list_model_files():
    files = []
    for p in sorted(MODELS_DIR.iterdir()):
        if p.suffix.lower() in SUPPORTED_EXTS and not p.name.startswith('.'):
            files.append(p.name)
    return files


def get_saved_active():
    if ACTIVE_FILE.exists():
        name = ACTIVE_FILE.read_text().strip()
        p = MODELS_DIR / name
        if p.exists():
            return p
    for ext in SUPPORTED_EXTS:
        p = MODELS_DIR / f'current{ext}'
        if p.exists():
            return p
    return None


def save_active_name(filename):
    ACTIVE_FILE.write_text(filename)


# Try to restore the last active model on startup.
_active = get_saved_active()
if _active is not None:
    try:
        _m, _k, _c, _it = load_any(_active)
        set_active(_m, _k, _c, _active.name, _it)
        print(f'[startup] loaded {_k.upper()} ({_it}) model from {_active.name}')
    except Exception as e:
        print(f'[startup] failed to load existing model: {e}')


# ---------------- pages ----------------

@app.route('/')
def user_page():
    return render_template('user.html')


@app.route('/admin')
def admin_page():
    return render_template('admin.html')


@app.route('/collect')
def collect_page():
    return render_template('collect.html')


@app.route('/analysis')
def analysis_page():
    return render_template('analysis.html')


# ---------------- API ----------------

@app.route('/api/status')
def api_status():
    with _lock:
        return jsonify({
            'loaded':     _state['model'] is not None,
            'kind':       _state['kind'],
            'filename':   _state['filename'],
            'classes':    _state['classes'],
            'device':     'cpu',
            'input_type': _state['input_type'],
        })


@app.route('/api/models')
def api_models():
    with _lock:
        active = _state['filename']
    return jsonify({'models': list_model_files(), 'active': active})


@app.route('/api/set-active', methods=['POST'])
def api_set_active():
    payload  = request.get_json(silent=True) or {}
    filename = payload.get('filename')
    if not filename:
        return jsonify({'error': 'Missing "filename".'}), 400
    path = MODELS_DIR / secure_filename(filename)
    if not path.exists():
        return jsonify({'error': f'Model not found: {filename}'}), 404
    # Free old model before loading new one to avoid two models in RAM simultaneously
    with _lock:
        _state['model'] = None
    gc.collect()
    try:
        model, kind, classes, input_type = load_any(path)
    except Exception as e:
        return jsonify({'error': f'Could not load model: {e}'}), 400
    set_active(model, kind, classes, path.name, input_type)
    save_active_name(path.name)
    return jsonify({'success': True, 'filename': path.name, 'kind': kind, 'input_type': input_type})


@app.route('/api/predict', methods=['POST'])
def api_predict():
    with _lock:
        model      = _state['model']
        kind       = _state['kind']
        classes    = list(_state['classes'])
        input_type = _state['input_type']

    if model is None:
        return jsonify({'error': 'No model loaded. Upload one at /admin first.'}), 400

    payload   = request.get_json(silent=True) or {}
    image_b64 = payload.get('image')
    use_tta   = bool(payload.get('tta', True))
    if not image_b64:
        return jsonify({'error': 'Missing "image" (data URL or base64 PNG).'}), 400

    if image_b64.startswith('data:'):
        image_b64 = image_b64.split(',', 1)[1]
    try:
        img_bytes = base64.b64decode(image_b64)
        pil = Image.open(io.BytesIO(img_bytes)).convert('L')
    except Exception as e:
        return jsonify({'error': f'Could not decode image: {e}'}), 400

    variants = TTA_VARIANTS if use_tta else TTA_VARIANTS[:1]
    warped = [pil_affine(pil, **v, fill=255) for v in variants]

    if kind == 'cnn':
        batch = np.concatenate([preprocess_cnn(p) for p in warped], axis=0)
        logits = model.run(['output'], {'input': batch})[0]
        avg_probs = softmax(logits).mean(axis=0)

    elif kind == 'rf':
        X = np.stack([
            np.asarray(p.resize((28, 28), Image.BILINEAR), dtype=np.uint8).reshape(-1).astype(np.float32)
            for p in warped
        ])
        _, proba_dicts = model.run(None, {'float_input': X})
        proba = np.array([[d[c] for c in classes] for d in proba_dicts])
        avg_probs = proba.mean(axis=0)

    else:
        return jsonify({'error': f'Unknown model kind: {kind}'}), 500

    probs_list = [float(x) for x in avg_probs.tolist()]
    top = int(np.argmax(avg_probs))
    return jsonify({
        'prediction': classes[top],
        'confidence': probs_list[top],
        'probs':      dict(zip(classes, probs_list)),
        'tta_count':  len(variants),
        'kind':       kind,
    })


@app.route('/api/upload-model', methods=['POST'])
def api_upload_model():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded (field name must be "file").'}), 400

    f = request.files['file']
    if not f.filename:
        return jsonify({'error': 'Empty filename.'}), 400

    safe_name = secure_filename(f.filename)
    ext = Path(safe_name).suffix.lower()
    if ext not in SUPPORTED_EXTS:
        return jsonify({
            'error': f'Only these file types are accepted: {", ".join(SUPPORTED_EXTS)}.'
        }), 400

    staging = MODELS_DIR / ('.staging_' + safe_name)
    f.save(str(staging))
    try:
        new_model, new_kind, new_classes, new_input_type = load_any(staging)
    except Exception as e:
        try:
            staging.unlink()
        except Exception:
            pass
        return jsonify({'error': f'Could not load uploaded model: {e}'}), 400

    target = MODELS_DIR / safe_name
    staging.replace(target)
    set_active(new_model, new_kind, new_classes, safe_name, new_input_type)
    save_active_name(safe_name)

    return jsonify({
        'success':    True,
        'filename':   safe_name,
        'kind':       new_kind,
        'message':    f'{new_kind.upper()} model uploaded and activated.',
        'size_bytes': target.stat().st_size,
    })


def load_dataset_images():
    pil_images, true_labels = [], []
    for cls in CLASSES:
        for p in sorted((DATA_DIR / cls).glob('*.png')):
            try:
                pil_images.append(Image.open(str(p)).convert('L'))
                true_labels.append(cls)
            except Exception:
                pass
    return pil_images, true_labels


def predict_batch(model, kind, classes, pil_images, input_type='raw'):
    if kind == 'cnn':
        batch = np.concatenate([preprocess_cnn(img) for img in pil_images], axis=0)
        logits = model.run(['output'], {'input': batch})[0]
        preds = logits.argmax(axis=1)
        return [classes[int(p)] for p in preds]
    else:
        X = np.stack([
            np.asarray(img.resize((28, 28), Image.BILINEAR), dtype=np.uint8).reshape(-1).astype(np.float32)
            for img in pil_images
        ])
        labels, _ = model.run(None, {'float_input': X})
        return [str(p) for p in labels]


@app.route('/api/dataset-stats')
def api_dataset_stats():
    stats = {}
    for cls in CLASSES:
        stats[cls] = len(list((DATA_DIR / cls).glob('*.png')))
    return jsonify({'stats': stats, 'total': sum(stats.values())})


@app.route('/api/save-sample', methods=['POST'])
def api_save_sample():
    payload = request.get_json(silent=True) or {}
    image_b64 = payload.get('image')
    label = payload.get('label')

    if not image_b64:
        return jsonify({'error': 'Missing "image".'}), 400
    if label not in CLASSES:
        return jsonify({'error': f'Invalid label "{label}". Must be one of {CLASSES}.'}), 400

    if image_b64.startswith('data:'):
        image_b64 = image_b64.split(',', 1)[1]
    try:
        img_bytes = base64.b64decode(image_b64)
        pil = Image.open(io.BytesIO(img_bytes)).convert('L').resize((28, 28))
    except Exception as e:
        return jsonify({'error': f'Could not decode image: {e}'}), 400

    cls_dir = DATA_DIR / label
    existing = sorted(cls_dir.glob('*.png'))
    next_idx = len(existing) + 1
    filename = f'{next_idx:03d}.png'
    pil.save(str(cls_dir / filename))

    count = len(list(cls_dir.glob('*.png')))
    return jsonify({'success': True, 'filename': filename, 'label': label, 'count': count})


@app.route('/api/run-analysis')
def api_run_analysis():
    with _lock:
        model      = _state['model']
        kind       = _state['kind']
        classes    = list(_state['classes'])
        input_type = _state['input_type']
    if model is None:
        return jsonify({'error': 'No model loaded.'}), 400

    pil_images, true_labels = load_dataset_images()
    if not pil_images:
        return jsonify({'error': 'No dataset images found in data/dataset/.'}), 400

    pred_labels = predict_batch(model, kind, classes, pil_images, input_type)

    cm = [[0] * len(classes) for _ in classes]
    for t, p in zip(true_labels, pred_labels):
        cm[classes.index(t)][classes.index(p)] += 1

    per_class = {}
    for cls in classes:
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p != cls)
        support = tp + fn
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[cls] = {'precision': round(prec, 4), 'recall': round(rec, 4),
                          'f1': round(f1, 4), 'support': support, 'correct': tp}

    misclassified = []
    for img, t, p in zip(pil_images, true_labels, pred_labels):
        if t == p:
            continue
        buf = io.BytesIO()
        img.resize((56, 56), Image.NEAREST).save(buf, format='PNG')
        misclassified.append({
            'true': t, 'pred': p,
            'img': base64.b64encode(buf.getvalue()).decode(),
        })
        if len(misclassified) >= 20:
            break

    n_correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
    return jsonify({
        'accuracy': round(n_correct / len(true_labels), 4),
        'n_total': len(true_labels),
        'n_errors': len(true_labels) - n_correct,
        'cm': cm,
        'classes': classes,
        'per_class': per_class,
        'misclassified': misclassified,
        'kind': kind,
    })


@app.route('/api/run-cv')
def api_run_cv():
    with _lock:
        model_obj  = _state['model']
        kind       = _state['kind']
        classes    = list(_state['classes'])
        input_type = _state['input_type']
    if model_obj is None:
        return jsonify({'error': 'No model loaded.'}), 400

    pil_images, true_labels = load_dataset_images()
    if len(pil_images) < 10:
        return jsonify({'error': 'Not enough images (need at least 10).'}), 400

    from sklearn.model_selection import StratifiedKFold

    y = np.array([classes.index(lbl) for lbl in true_labels])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = []

    if kind == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        X = np.stack([
            np.asarray(img.resize((28, 28), Image.BILINEAR), dtype=np.uint8).reshape(-1)
            for img in pil_images
        ])
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            clf = RandomForestClassifier(
                n_estimators=200, max_features='sqrt',
                class_weight='balanced', random_state=42, n_jobs=-1,
            )
            clf.fit(X[train_idx], y[train_idx])
            preds = clf.predict(X[val_idx])
            acc = float((preds == y[val_idx]).mean())
            folds.append({'fold': fold_idx + 1, 'accuracy': round(acc, 4),
                          'n_val': int(len(val_idx))})

    elif kind == 'cnn':
        for fold_idx, (_, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
            val_imgs  = [pil_images[i] for i in val_idx]
            val_true  = [true_labels[i] for i in val_idx]
            val_preds = predict_batch(model_obj, kind, classes, val_imgs)
            acc = sum(1 for t, p in zip(val_true, val_preds) if t == p) / len(val_true)
            folds.append({'fold': fold_idx + 1, 'accuracy': round(acc, 4),
                          'n_val': len(val_idx)})

    accs = [f['accuracy'] for f in folds]
    return jsonify({
        'folds': folds,
        'mean_accuracy': round(float(np.mean(accs)), 4),
        'std_accuracy':  round(float(np.std(accs)),  4),
        'n_total': len(y),
        'kind': kind,
        'note': ('5-fold CV with retraining each fold' if kind == 'rf'
                 else 'Evaluation-only CV — current model tested on each fold without retraining'),
    })


@app.route('/api/compare-models')
def api_compare_models():
    pil_images, true_labels = load_dataset_images()
    if not pil_images:
        return jsonify({'error': 'No dataset images found in data/dataset/.'}), 400

    model_files = list_model_files()
    if not model_files:
        return jsonify({'error': 'No model files found in app/models/.'}), 400

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    results = []
    for fname in model_files:
        path = MODELS_DIR / fname
        try:
            m, kind, classes, input_type = load_any(path)
            pred_labels = predict_batch(m, kind, classes, pil_images, input_type)
            true_idx = [classes.index(t) if t in classes else -1 for t in true_labels]
            pred_idx = [classes.index(p) if p in classes else -1 for p in pred_labels]
            results.append({
                'name':       fname,
                'kind':       kind,
                'input_type': input_type,
                'accuracy':   round(accuracy_score(true_idx, pred_idx), 4),
                'precision':  round(precision_score(true_idx, pred_idx, average='macro', zero_division=0), 4),
                'recall':     round(recall_score(true_idx, pred_idx,    average='macro', zero_division=0), 4),
                'f1':         round(f1_score(true_idx, pred_idx,        average='macro', zero_division=0), 4),
            })
        except Exception as e:
            results.append({'name': fname, 'error': str(e)})

    results.sort(key=lambda r: r.get('f1', -1), reverse=True)
    return jsonify({'results': results, 'n_samples': len(pil_images)})


if __name__ == '__main__':
    import os
    PORT = int(os.environ.get('PORT', 5000))
    is_local = os.environ.get('RENDER') is None
    if is_local:
        URL = f'http://localhost:{PORT}/'
        print(f'\n  Server starting at {URL}\n  Press Ctrl+C to stop.\n')
        threading.Timer(1.5, lambda: webbrowser.open(URL)).start()
    app.run(host='0.0.0.0', port=PORT, debug=is_local, use_reloader=False)
