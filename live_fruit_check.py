#!/usr/bin/env python3
# live_fruit_check.py
import cv2, sys, numpy as np, argparse
from tensorflow.keras.models import load_model
from PIL import Image

def load_labels(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception:
        return None

def prepare_frame(frame_bgr, target_h, target_w, norm):
    # BGR to RGB
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img).resize((target_w, target_h))
    x = np.array(img_pil).astype('float32')

    if norm == '0-1':
        x = x / 255.0
    elif norm == 'imagenet':
        # Inception-style: scale to -1..1 OR subtract imagenet mean; many implementations for InceptionV3 expect [-1,1]
        # We'll try scale to [-1,1]
        x = (x / 127.5) - 1.0
    elif norm == 'standard':
        mean = np.mean(x, axis=(0,1))
        std = np.std(x, axis=(0,1)) + 1e-6
        x = (x - mean) / std
    else:
        raise ValueError("norm must be '0-1', 'imagenet', or 'standard'")

    return np.expand_dims(x, axis=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--labels', default=None)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--norm', default='imagenet', choices=['0-1','imagenet','standard'])
    parser.add_argument('--width', type=int, default=900)
    args = parser.parse_args()

    print("Loading model:", args.model)
    model = load_model(args.model, compile=False)
    print("Loaded. Model input shape:", model.input_shape, "output shape:", model.output_shape)

    labels = load_labels(args.labels) if args.labels else None
    if labels:
        print("Loaded labels:", labels)
    else:
        # fallback label names if unknown
        out_shape = model.output_shape
        if isinstance(out_shape, tuple) and len(out_shape) == 2:
            n = out_shape[1]
        else:
            n = 1
        labels = [f"class_{i}" for i in range(n)]
        print("No labels file provided. Using:", labels)

    # Determine target size from model input shape
    in_shape = model.input_shape
    # input_shape is typically (None, h, w, c) or (None, c, h, w)
    if len(in_shape) == 4:
        _, h, w, c = in_shape
    elif len(in_shape) == 3:
        h, w, c = in_shape
    else:
        raise RuntimeError("Unexpected input shape: " + str(in_shape))
    if h is None or w is None:
        # fallback
        h, w = 299, 299
    print("Using target size:", (h,w))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Cannot open camera index", args.camera)
        sys.exit(1)

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    scale = args.width / float(cam_w)
    display_size = (int(cam_w * scale), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale))

    print("Press q to quit, s to save snapshot. Normalization:", args.norm)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed read")
            break

        try:
            X = prepare_frame(frame, h, w, args.norm)
            preds = model.predict(X)
            # flatten properly
            if preds.ndim == 2:
                probs = preds[0]
            else:
                probs = preds.flatten()
        except Exception as e:
            cv2.putText(frame, "Model error: " + str(e), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            cv2.imshow('FruitCheck', cv2.resize(frame, display_size))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Top class
        top = int(np.argmax(probs))
        conf = float(probs[top])
        lab = labels[top] if top < len(labels) else f"class_{top}"
        text = f"{lab} {conf:.2f}"
        color = (0,255,0)
        if 'rot' in lab.lower() or 'bad' in lab.lower() or lab.lower().startswith('r'):
            color = (0,0,255)
        cv2.putText(frame, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # show top 3
        start_y = 60
        for i, idx in enumerate(np.argsort(-probs)[:3]):
            l = labels[idx] if idx < len(labels) else f"class_{idx}"
            cv2.putText(frame, f"{l}: {probs[idx]:.2f}", (10, start_y + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)

        frame_small = cv2.resize(frame, display_size)
        cv2.imshow('FruitCheck', frame_small)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        if k == ord('s'):
            import time
            fname = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(fname, frame)
            print("Saved", fname)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
