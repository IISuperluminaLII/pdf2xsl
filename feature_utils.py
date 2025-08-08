import os
import pickle
from typing import List

import numpy as np
from PIL import Image

from tensorflow.keras.applications.inception_resnet_v2 import (
    InceptionResNetV2,
    preprocess_input,
)
from tensorflow.keras.preprocessing.image import img_to_array

try:
    from pdf2image import convert_from_path
except Exception:  # pragma: no cover - only executed when pdf2image missing
    convert_from_path = None

def load_or_generate_features(pdf_dir: str = "pdfs", cache_path: str = "featsV-1.pkl") -> np.ndarray:
    """Return visual features for PDF training data.

    If ``cache_path`` exists the features are loaded from that pickle file.
    Otherwise the PDFs (or images) found in ``pdf_dir`` are converted to
    299x299 RGB images, processed with :func:`preprocess_input`, and passed
    through a pretrained :class:`InceptionResNetV2` network to obtain
    ``8x8x1536`` feature maps.  The computed features are cached to
    ``cache_path`` for future runs.

    Parameters
    ----------
    pdf_dir:
        Directory containing training PDFs or already-rendered page images.
    cache_path:
        File path used to cache the extracted features.

    Returns
    -------
    np.ndarray
        Array of shape ``(N, 8, 8, 1536)`` where ``N`` is the number of
        documents found in ``pdf_dir``.
    """
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    if not os.path.isdir(pdf_dir):
        raise FileNotFoundError(f"PDF directory '{pdf_dir}' not found")

    images: List[np.ndarray] = []
    for fname in sorted(os.listdir(pdf_dir)):
        path = os.path.join(pdf_dir, fname)
        if fname.lower().endswith(".pdf"):
            if convert_from_path is None:
                raise ImportError(
                    "pdf2image is required to convert PDFs. Install it via 'pip install pdf2image'"
                )
            page = convert_from_path(path, dpi=200, first_page=1, last_page=1)[0]
        else:
            page = Image.open(path)
        page = page.resize((299, 299)).convert("RGB")
        images.append(img_to_array(page))

    images = np.array(images)
    images = preprocess_input(images)

    model = InceptionResNetV2(include_top=False, weights="imagenet")
    features = model.predict(images, batch_size=16, verbose=1)

    with open(cache_path, "wb") as f:
        pickle.dump(features, f)

    return features
