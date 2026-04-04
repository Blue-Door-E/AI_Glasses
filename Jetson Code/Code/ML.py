from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

from config import IMG_SIZE, MODEL_DIR

try:
    import torch
    import torch.nn.functional as F
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

# ImageNet normalization constants (as tensors for GPU preprocessing)
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


# ------------- model loader -------------
def find_latest_model(model_dir="outputs") -> Optional[Tuple[str, str]]:
    try:
        if not os.path.exists(model_dir):
            return None
        subs = [os.path.join(model_dir, d) for d in os.listdir(model_dir)
                if os.path.isdir(os.path.join(model_dir, d))]
        subs.sort(key=lambda d: os.path.getmtime(d), reverse=True)
        for d in subs:
            m, l = os.path.join(d, "resnet18_scripted.pt"), os.path.join(d, "labels.txt")
            if os.path.exists(m) and os.path.exists(l):
                return m, l
        m, l = os.path.join(model_dir, "resnet18_scripted.pt"), os.path.join(model_dir, "labels.txt")
        if os.path.exists(m) and os.path.exists(l):
            return m, l
    except Exception:
        pass
    return None





# ------------- GPU preprocessing (replaces torchvision PIL pipeline) -------------
def _preprocess_gpu(bgr: np.ndarray, device: str, img_size: int,
                    mean_t: torch.Tensor, std_t: torch.Tensor) -> torch.Tensor:
    """Preprocess a BGR numpy image entirely on GPU.

    Replaces the old pipeline:
        ToPILImage -> Resize -> ToTensor -> Normalize
    With:
        numpy BGR->RGB -> torch tensor on GPU -> resize on GPU -> normalize on GPU

    This eliminates CPU-side PIL conversion and keeps everything on the GPU
    after the initial transfer.
    """
    # BGR -> RGB via numpy (single slice, no copy)
    rgb = bgr[:, :, ::-1]

    # numpy uint8 HWC -> torch float32 CHW, transfer to GPU in one step
    t = torch.from_numpy(rgb.copy()).to(device, dtype=torch.float32, non_blocking=True)
    t = t.permute(2, 0, 1).unsqueeze(0)  # HWC -> 1xCxHxW
    t.div_(255.0)  # [0, 255] -> [0.0, 1.0] in-place

    # Resize on GPU using bilinear interpolation
    if t.shape[2] != img_size or t.shape[3] != img_size:
        t = F.interpolate(t, size=(img_size, img_size), mode="bilinear",
                          align_corners=False)

    # Normalize on GPU (ImageNet stats)
    t.sub_(mean_t).div_(std_t)

    return t





class OptionalModel:
    def __init__(self):
        self.ready = False
        self.labels = []
        self.model = None
        self.tf = None
        self.device = "cpu"
        self._trt_backend = None
        self._use_trt = False

        # Display-only state
        self.model_path = ""
        self.device_tag = "CPU"

        # GPU preprocessing tensors (allocated once)
        self._mean_t = None
        self._std_t = None

        if not TORCH_OK:
            print("[Model] torch not available; running without inference.")
            return

        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[Model] Using device: {self.device}")
        except Exception:
            self.device = "cpu"

        self.device_tag = "CUDA" if self.device == "cuda" else "CPU"

        found = find_latest_model(MODEL_DIR)
        if not found:
            print("[Model] No model found; predictions will be N/A.")
            return

        try:
            mfile, lfile = found
            self.model_path = mfile

            # Load labels
            with open(lfile, "r", encoding="utf-8") as f:
                self.labels = [x.strip() for x in f if x.strip()]

            # Pre-allocate normalization tensors on device (used every frame)
            if self.device == "cuda":
                self._mean_t = torch.tensor(_MEAN, device="cuda").view(1, 3, 1, 1)
                self._std_t = torch.tensor(_STD, device="cuda").view(1, 3, 1, 1)


            self.model = torch.jit.load(mfile, map_location=self.device)
            self.model.eval()

            # Use GPU preprocessing if CUDA is available, else keep
            # the old torchvision pipeline for CPU-only environments
            if self.device != "cuda":
                from torchvision import transforms
                self.tf = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((IMG_SIZE, IMG_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=_MEAN, std=_STD),
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

        # --- Preprocessing ---
        if self.device == "cuda":
            # GPU path: numpy -> GPU tensor (no PIL, no CPU resize)
            x = _preprocess_gpu(bgr, self.device, IMG_SIZE,
                                self._mean_t, self._std_t)
        else:
            # CPU fallback: original torchvision pipeline
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if CV2_OK else bgr[:, :, ::-1]
            x = self.tf(img).unsqueeze(0)

        # --- Inference ---
        if self._use_trt:
            y = self._trt_backend.infer(x)
        else:
            if self.device != "cpu":
                x = x.to(self.device, non_blocking=True)
            y = self.model(x)

        # --- Postprocessing (keep on GPU until final scalar extraction) ---
        if self.device != "cpu":
            p = torch.softmax(y, dim=1)[0]
            conf, idx = torch.max(p, dim=0)
            idxi = int(idx.item())
            conf_val = float(conf.item())
        else:
            y = y.to("cpu")
            p = torch.softmax(y, dim=1)[0]
            conf, idx = torch.max(p, dim=0)
            idxi = int(idx)
            conf_val = float(conf)

        label = self.labels[idxi] if 0 <= idxi < len(self.labels) else f"class_{idxi}"
        return label, conf_val, idxi
