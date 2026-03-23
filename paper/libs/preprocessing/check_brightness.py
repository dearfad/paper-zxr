"""图像亮度与对比度检测模块"""

import cv2
import numpy as np
from typing import Dict, Any

def detect_brightness(image: np.ndarray) -> Dict[str, float]:
    """计算图像亮度统计"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return {
        "mean": float(np.mean(gray)),
        "std": float(np.std(gray))
    }


def detect_contrast(image: np.ndarray) -> Dict[str, float]:
    """计算图像对比度（拉普拉斯方差）"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return {"variance": float(variance)}