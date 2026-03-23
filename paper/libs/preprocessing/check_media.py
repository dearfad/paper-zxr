"""培养基颜色/状态检测模块

通过分析图像的整体色调（Hue）和饱和度，判断培养基是否出现异常变色。
正常的培养基通常呈现特定的粉红色/橙红色（酚红指示剂），
变黄（酸性/污染）或变紫（碱性）均视为异常。
"""

import cv2
import numpy as np
from typing import Dict, Any

# 默认阈值：颜色评分偏离标准值的程度
# 评分逻辑：基于 Hue 通道与标准培养基 Hue 值的距离
# 范围归一化后，超过此值视为异常
DEFAULT_COLOR_DEVIATION_THRESHOLD = 0.25

# 标准培养基的预期 Hue 值 (OpenCV 中 0-180)
# 酚红培养基通常为粉红/橙红，Hue 大约在 0-15 或 160-180 之间
# 这里假设主要分布在 0-15 (橙红) 附近，具体需根据实际白平衡校准
EXPECTED_MEDIA_HUE = 10.0 


def detect_media_color(
    image_bgr: np.ndarray, 
    deviation_threshold: float = DEFAULT_COLOR_DEVIATION_THRESHOLD
) -> Dict[str, Any]:
    """
    检测培养基颜色是否异常。
    
    逻辑：
    1. 转 HSV 色彩空间。
    2. 屏蔽低饱和度区域（排除灰色/白色背景干扰）。
    3. 计算有效区域的平均 Hue 值。
    4. 计算平均 Hue 与预期标准 Hue 的归一化距离。
    
    :param image_bgr: BGR 格式图像
    :param deviation_threshold: 颜色偏差阈值 (0.0 - 1.0)
    :return: 包含颜色评分和判定结果的字典
    """
    if image_bgr is None or image_bgr.size == 0:
        return {"score": 0.0, "is_abnormal": True}

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # 创建掩膜：只考虑饱和度较高的区域，排除背景和白色反光
    # 饱和度阈值设为 40 (OpenCV 0-255)
    mask = cv2.inRange(s, 40, 255)
    
    if np.sum(mask) == 0:
        # 如果没有彩色区域，无法判断，暂定为正常或根据业务需求定为异常
        # 这里假设若无颜色信息则无法检测出异常，返回低分
        return {"score": 0.0, "is_abnormal": False}
    
    # 计算掩膜区域内的平均 Hue
    mean_hue = cv2.mean(h, mask=mask)[0]
    
    # 处理 Hue 的循环特性 (0 和 180 是接近的)
    # 计算与预期值的绝对差，考虑圆周距离
    diff = abs(mean_hue - EXPECTED_MEDIA_HUE)
    circular_diff = min(diff, 180 - diff)
    
    # 归一化到 0-1 范围 (最大可能差值为 90)
    score = circular_diff / 90.0
    
    is_abnormal = score > deviation_threshold
    
    return {
        "score": float(score),
        "is_abnormal": bool(is_abnormal),
        "mean_hue": float(mean_hue),
        "threshold_used": deviation_threshold
    }