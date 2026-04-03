from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

from config import IMG_SIZE, MODEL_DIR

try:
    import torch
    from torchvision import transforms
    TORCH_OK = True
except Exception:
    TORCH_OK = False

try:
    import cv2
    CV2_OK = True
except Exception:
    CV2_OK = False

if TORCH_OK:
    _no_grad = torch.no_grad
else:
    def _no_grad():
        def _decorator(fn):
            return fn
        return _decorator


# ------------- model loader (UNCHANGED) -------------
def find_latest_model(model_dir="outputs") -> Optional[Tuple[str, str]]:
    try:
        if not os.path.exists(model_dir):
            return None
        subs = [os.path.join(model_dir, d) for d in os.listdir(model_dir)
                if os.path.isdir(os.path.join(model_dir, d))]
        subs.sort(key=lambda d: os.path.getmtime(d), reverse=True)
        for d in subs:
            m, l = os.path.join(d, "400x400resnet18_scripted.pt"), os.path.join(d, "labels.txt")
            if os.path.exists(m) and os.path.exists(l):
                return m, l
        m, l = os.path.join(model_dir, "400x400resnet18_scripted.pt"), os.path.join(model_dir, "labels.txt")
        if os.path.exists(m) and os.path.exists(l):
            return m, l
    except Exception:
        pass
    return None

class OptionalModel:
    def __init__(self):
        self.ready = False
        self.labels = []
        self.model = None
        self.tf = None
        self.device = "cpu"
   
        # --- NEW (display-only) ---
        self.model_path = ""
        self.device_tag = "CPU"

        if not TORCH_OK:
            print("[Model] torch not available; running without inference.")
            return

        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[Model] Using device: {self.device}")
        except Exception:
            self.device = "cpu"
        # --- NEW (display-only) ---
        self.device_tag = "CUDA" if self.device == "cuda" else "CPU"
        found = find_latest_model(MODEL_DIR)
        if not found:
            print("[Model] No model found; predictions will be N/A.")
            return

        try:
            mfile, lfile = found
            self.model_path = mfile
            self.model = torch.jit.load(mfile, map_location=self.device)
            self.model.eval()
            with open(lfile, "r", encoding="utf-8") as f:
                self.labels = [x.strip() for x in f if x.strip()]
            self.tf = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            self.ready = True
            print(f"[Model] Loaded {mfile} with {len(self.labels)} labels.")
        except Exception as e:
            print("[Model] Load failed:", e)
            self.ready = False

    @_no_grad()
    def predict(self, bgr: np.ndarray) -> Tuple[str, float, int]:
        if not self.ready or bgr is None:
            return "N/A", 0.0, -1
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if CV2_OK else bgr[:, :, ::-1]
        x = self.tf(img).unsqueeze(0)
        if self.device != "cpu":
            x = x.to(self.device, non_blocking=True)
        y = self.model(x)
        if self.device != "cpu":
            y = y.to("cpu")
        p = torch.softmax(y, dim=1)[0]
        conf, idx = torch.max(p, dim=0)
        idxi = int(idx)
        label = self.labels[idxi] if 0 <= idxi < len(self.labels) else f"class_{idxi}"
        return label, float(conf), idxi
