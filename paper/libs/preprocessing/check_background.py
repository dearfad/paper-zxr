"""背景均匀性检测模块

通过计算图像背景区域（或整体）的灰度标准差来评估背景均匀性。
标准差过大意味着背景存在光照不均、污渍或噪声干扰。
"""

import numpy as np
from typing import Dict, Any

# 默认阈值：标准差超过此值视为背景不均匀
# 对于干净的显微图像，背景通常非常均一，Std 应很小 (例如 < 10-15)
DEFAULT_BACKGROUND_STD_THRESHOLD = 15.0


def detect_background_uniformity(
    image_bgr: np.ndarray, 
    threshold: float = DEFAULT_BACKGROUND_STD_THRESHOLD
) -> Dict[str, Any]:
    """
    检测图像背景的均匀性。
    
    逻辑：
    1. 转换为灰度图。
    2. (可选优化) 尝试掩膜去除前景细胞，仅计算背景区域的标准差。
       此处为通用性，先采用全图统计，若前景占比小则影响有限；
       若需高精度，可结合大核高斯模糊估算背景层。
    3. 计算整图或背景层的标准差。
    
    :param image_bgr: BGR 格式的图像数组 (numpy array)
    :param threshold: 判定为不均匀的标准差阈值
    :return: 包含标准差值和判定结果的字典
    """
    if image_bgr is None or image_bgr.size == 0:
        return {"std_dev": 0.0, "is_non_uniform": True}

    # 转为灰度
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # 策略：使用大尺寸高斯模糊提取“背景层”，计算原图与背景层的差异或直接计算背景层的平滑度
    # 这里采用直接计算全图标准差的简化版，对于背景占比较大的显微图有效
    # 进阶：可以通过阈值分割去掉前景细胞再算背景 Std
    mean_val = np.mean(gray)
    std_dev = np.std(gray)
    
    # 如果图像整体很亮或很暗，标准差可能受前景影响大。
    # 更鲁棒的方法：只取像素值接近背景均值的那些像素来计算标准差
    # 假设背景是出现频率最高的亮度区间 (直方图峰值)
    # 简单实现：直接比较全局标准差
    is_non_uniform = std_dev > threshold

    return {
        "std_dev": float(std_dev),
        "is_non_uniform": bool(is_non_uniform),
        "threshold_used": threshold
    }

# 需要导入 cv2，放在文件顶部或函数内确保依赖
import cv2