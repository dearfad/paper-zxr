"""图像读取与基础元数据获取模块"""

from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np
from PIL import Image

def load_image_bgr(path: Path) -> np.ndarray:
    """使用 PIL 读取图像并转换为 BGR 格式（兼容 OpenCV）"""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    rgb = np.array(img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def get_image_basic_info(path: Path) -> Dict[str, Any]:
    """获取图像基本元数据（格式、尺寸等）"""
    try:
        with Image.open(path) as img:
            img.load()  # 强制加载以检测损坏
            return {
                "success": True,
                "format": img.format,
                "mode": img.mode,
                "width": img.width,
                "height": img.height,
                "error": ""
            }
    except Exception as e:
        return {
            "success": False,
            "format": "",
            "mode": "",
            "width": 0,
            "height": 0,
            "error": str(e)
        }