"""图像划痕检测模块"""

import cv2
import numpy as np
from typing import Dict, Any

DEFAULT_MIN_SCRATCH_LENGTH = 20

def detect_scratches(
    image: np.ndarray,
    min_length: int = DEFAULT_MIN_SCRATCH_LENGTH,
    line_threshold: int = 50
) -> Dict[str, Any]:
    """霍夫变换检测线性特征（划痕）"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, 
        threshold=line_threshold,
        minLineLength=min_length,
        maxLineGap=5
    )
    
    count = len(lines) if lines is not None else 0
    return {"count": count}