"""图像模糊度检测模块"""

import cv2
import numpy as np
from typing import Dict, Any

DEFAULT_BLUR_THRESHOLD = 100.0

def check_blur(image: np.ndarray, threshold: float = DEFAULT_BLUR_THRESHOLD) -> Dict[str, Any]:
    """拉普拉斯方差法检测模糊度"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return {
        "is_blur": variance < threshold,
        "variance": float(variance),
        "threshold": threshold
    }