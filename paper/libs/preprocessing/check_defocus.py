"""局部离焦/多焦点检测模块

将图像划分为多个网格，分别计算每个网格的清晰度（拉普拉斯方差）。
如果部分网格清晰而部分网格模糊，且模糊比例超过阈值，则判定为局部离焦。
这通常由样本表面不平整或镜头倾斜引起。
"""

import cv2
import numpy as np
from typing import Dict, Any

# 默认阈值：模糊区域占比超过此值视为局部离焦
DEFAULT_DEFOCUS_RATIO_THRESHOLD = 0.15

# 内部使用的清晰度阈值 (拉普拉斯方差)
# 低于此值认为该小块是模糊的
LOCAL_BLUR_THRESHOLD = 100.0 


def detect_partial_defocus(
    image_bgr: np.ndarray, 
    ratio_threshold: float = DEFAULT_DEFOCUS_RATIO_THRESHOLD
) -> Dict[str, Any]:
    """
    检测图像是否存在局部离焦现象。
    
    逻辑：
    1. 转灰度。
    2. 将图像划分为 N x M 的网格 (例如 4x4 或 5x5)。
    3. 对每个网格计算拉普拉斯方差。
    4. 统计方差低于局部阈值的网格数量。
    5. 计算模糊网格占比。
    
    :param image_bgr: BGR 格式图像
    :param ratio_threshold: 模糊区域占比阈值
    :return: 包含模糊占比和判定结果的字典
    """
    if image_bgr is None or image_bgr.size == 0:
        return {"ratio": 0.0, "is_defocused": True}

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # 定义网格大小 (例如 5x5)
    grid_h, grid_w = 5, 5
    step_y, step_x = h // grid_h, w // grid_w
    
    total_blocks = grid_h * grid_w
    blur_blocks = 0
    
    for i in range(grid_h):
        for j in range(grid_w):
            y1, y2 = i * step_y, (i + 1) * step_y
            x1, x2 = j * step_x, (j + 1) * step_x
            
            # 提取块
            block = gray[y1:y2, x1:x2]
            
            # 计算拉普拉斯方差
            laplacian_var = cv2.Laplacian(block, cv2.CV_64F).var()
            
            if laplacian_var < LOCAL_BLUR_THRESHOLD:
                blur_blocks += 1
                
    ratio = blur_blocks / total_blocks if total_blocks > 0 else 0.0
    is_defocused = ratio > ratio_threshold
    
    return {
        "ratio": float(ratio),
        "is_defocused": bool(is_defocused),
        "threshold_used": ratio_threshold,
        "blur_block_count": blur_blocks,
        "total_blocks": total_blocks
    }