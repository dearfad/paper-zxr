"""细胞聚集/团块检测模块

通过形态学操作和轮廓分析，识别图像中面积异常大的连通区域，
这些区域通常代表细胞过度聚集、团块或杂质。
"""

import cv2
import numpy as np
from typing import Dict, Any

# 默认阈值：最大轮廓面积超过此值视为存在异常团块
# 该值高度依赖图像分辨率和放大倍数，需根据实际数据调整
# 假设在特定倍率下，正常单个细胞或小簇面积 < 5000 像素
DEFAULT_CLUMP_SIZE_THRESHOLD = 5000


def detect_cell_clumping(
    image_bgr: np.ndarray, 
    area_threshold: int = DEFAULT_CLUMP_SIZE_THRESHOLD
) -> Dict[str, Any]:
    """
    检测是否存在异常的细胞聚集或大团块。
    
    逻辑：
    1. 转灰度并高斯模糊去噪。
    2. 自适应阈值或 Otsu 二值化分割前景。
    3. 形态学闭操作连接邻近细胞，形成团块轮廓。
    4. 查找轮廓，统计超过面积阈值的轮廓数量。
    
    :param image_bgr: BGR 格式图像
    :param area_threshold: 判定为团块的最小面积阈值
    :return: 包含团块数量和判定结果的字典
    """
    if image_bgr is None or image_bgr.size == 0:
        return {"count": 0, "has_clumps": False}

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 使用 Otsu 自动阈值分割
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 形态学操作：连接相邻的前景区域，使聚集团块成为一个整体
    kernel = np.ones((3,3),np.uint8)
    dilated_thresh = cv2.dilate(thresh, kernel, iterations=2)
    eroded_thresh = cv2.erode(dilated_thresh, kernel, iterations=1)
    
    contours, _ = cv2.findContours(eroded_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    clump_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > area_threshold:
            clump_count += 1
            
    has_clumps = clump_count > 0
    
    return {
        "count": clump_count,
        "has_clumps": has_clumps,
        "threshold_used": area_threshold
    }