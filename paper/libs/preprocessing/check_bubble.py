"""图像气泡检测模块"""

import cv2
import numpy as np
from typing import Dict, Any

DEFAULT_MIN_BUBBLE_AREA = 100

def detect_bubbles(
    image: np.ndarray, 
    min_area: int = DEFAULT_MIN_BUBBLE_AREA,
    circularity_threshold: float = 0.7
) -> Dict[str, Any]:
    """检测圆形暗区（气泡）"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity >= circularity_threshold:
            count += 1
            
    return {"count": count}