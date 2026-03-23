"""细胞密度/覆盖率检测模块"""

import cv2
import numpy as np
from typing import Dict, Any

DEFAULT_CELL_COVERAGE_THRESHOLD = 5.0

def detect_cell_coverage(
    image: np.ndarray, 
    threshold: float = DEFAULT_CELL_COVERAGE_THRESHOLD
) -> Dict[str, Any]:
    """
    检测细胞密度/覆盖率
    方法：灰度化 -> 高斯模糊 -> Otsu 自动阈值分割 -> 计算前景占比
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 高斯模糊去噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Otsu 自动阈值 (二值化)
    ret, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    total_pixels = gray.size
    foreground_pixels = cv2.countNonZero(binary)
    
    coverage_rate = (foreground_pixels / total_pixels) * 100.0
    
    return {
        "coverage_rate": float(coverage_rate),
        "is_low_coverage": coverage_rate < threshold,
        "threshold": threshold
    }