"""类器官图像自动化分析模块

针对单视野下存在多个类器官团的实际情况，采用"先分团识别、再单独提取特征"的自动化流程：
1. 图像预处理（去噪、灰度化、阈值分割、背景去除）
2. 连通区域分析（自动识别并分离每个类器官团）
3. 核心形态特征提取（面积、圆度、灰度值、等效直径等）
4. 数据整理与导出（结构化数据集）

使用方法：
    # 分析单张图片
    python analysis_images.py --image image.jpg

    # 分析目录下所有图片
    python analysis_images.py --directory ./images/

    # 指定输出文件
    python analysis_images.py --directory ./images/ -o results.xlsx

    # 调整参数
    python analysis_images.py --image image.jpg --min-area 500 --gaussian-kernel 5
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

import cv2
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm


@dataclass
class OrganoidFeature:
    """类器官特征数据类"""

    # 图像信息
    image_file: str = ""  # 原始图像文件名
    field_of_view: str = ""  # 视野标识

    # 类器官标识
    organoid_id: int = 0  # 类器官编号（在同一图像内的编号）

    # 形态特征
    area: float = 0.0  # 面积（像素数），反映类器官体积大小与细胞聚集程度
    perimeter: float = 0.0  # 周长（像素），反映边缘规则性，周长越大边缘越不规则
    circularity: float = (
        0.0  # 圆度：4π×Area/(Perimeter)²，越接近1越接近标准球形，越小越不规则
    )
    equivalent_diameter: float = (
        0.0  # 等效直径（像素）：√(4×Area/π)，消除形状干扰统一量化尺寸
    )

    # 灰度特征
    mean_gray: float = 0.0  # 平均灰度值，反映内部密度
    std_gray: float = 0.0  # 灰度标准差，反映内部均匀性
    min_gray: float = 0.0  # 最小灰度值
    max_gray: float = 0.0  # 最大灰度值

    # 边缘清晰度特征
    edge_clarity: float = 0.0  # 边缘清晰度：边缘区域灰度梯度平均值，值越大边缘越清晰
    edge_gradient_mean: float = 0.0  # 边缘平均梯度强度
    edge_gradient_std: float = 0.0  # 边缘梯度标准差

    # 纹理特征（GLCM灰度共生矩阵）
    texture_contrast: float = 0.0  # 纹理对比度，反映局部灰度变化
    texture_homogeneity: float = 0.0  # 纹理均匀性，值越大纹理越均匀
    texture_energy: float = 0.0  # 纹理能量（ASM），反映灰度分布均匀性
    texture_correlation: float = 0.0  # 纹理相关性，反映局部灰度线性关系

    # 形状特征
    bounding_box_width: float = 0.0  # 边界框宽度
    bounding_box_height: float = 0.0  # 边界框高度
    aspect_ratio: float = 0.0  # 纵横比：宽/高
    solidity: float = 0.0  # 实体度：面积/凸包面积，反映凹凸程度
    extent: float = 0.0  # 范围度：面积/边界框面积

    # 位置信息
    centroid_x: float = 0.0  # 质心X坐标
    centroid_y: float = 0.0  # 质心Y坐标


def preprocess_image(
    image_bgr: np.ndarray,
    gaussian_kernel: int = 15,
    adaptive_block_size: int = 151,
    adaptive_c: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    图像预处理：去噪、灰度化、阈值分割、背景去除

    Args:
        image_bgr: BGR格式原始图像
        gaussian_kernel: 高斯滤波核大小（必须为奇数）
        adaptive_block_size: 自适应阈值块大小（必须为奇数），应大于类器官尺寸
        adaptive_c: 自适应阈值常数，越小分割越严格

    Returns:
        binary_mask: 二值化掩膜（类器官为白色255，背景为黑色0）
        gray_image: 预处理后的灰度图像
    """
    # 1. 灰度化：将RGB三通道彩色图像转换为单通道灰度图像
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # 2. 高斯滤波：平滑图像噪声，避免微小杂质被误识别
    # 确保核大小为奇数
    if gaussian_kernel % 2 == 0:
        gaussian_kernel += 1
    blurred = cv2.GaussianBlur(gray, (gaussian_kernel, gaussian_kernel), 0)

    # 3. 自适应阈值分割：根据灰度差异将图像二值化
    # 使类器官区域为前景（白色），培养基与背景为黑色
    if adaptive_block_size % 2 == 0:
        adaptive_block_size += 1
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        adaptive_block_size,
        adaptive_c,
    )

    # 4. 形态学操作：去除小噪声、填充空洞
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # 5. 膨胀操作：填补类器官内部空洞，使轮廓更完整
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.dilate(binary, dilate_kernel, iterations=2)

    return binary, gray


def calculate_edge_features(
    gray_image: np.ndarray, mask: np.ndarray
) -> Tuple[float, float, float]:
    """
    计算边缘清晰度特征

    Args:
        gray_image: 灰度图像
        mask: 类器官掩膜

    Returns:
        (edge_clarity, edge_gradient_mean, edge_gradient_std)
    """
    # 使用Sobel算子计算梯度
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

    # 提取边缘区域的梯度（掩膜边界附近）
    # 先腐蚀掩膜得到内部区域
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_eroded = cv2.erode(mask, kernel, iterations=1)

    # 边缘区域 = 原掩膜 - 腐蚀后的掩膜
    edge_region = cv2.subtract(mask, mask_eroded)

    # 提取边缘区域的梯度统计
    edge_pixels = gradient_magnitude[edge_region > 0]
    if len(edge_pixels) > 0:
        edge_clarity = float(np.mean(edge_pixels))
        edge_gradient_mean = float(np.mean(edge_pixels))
        edge_gradient_std = float(np.std(edge_pixels))
    else:
        edge_clarity = 0.0
        edge_gradient_mean = 0.0
        edge_gradient_std = 0.0

    return edge_clarity, edge_gradient_mean, edge_gradient_std


def calculate_texture_features(
    gray_image: np.ndarray, mask: np.ndarray
) -> Dict[str, float]:
    """
    计算纹理特征（基于灰度共生矩阵GLCM）

    Args:
        gray_image: 灰度图像
        mask: 类器官掩膜

    Returns:
        纹理特征字典
    """
    # 提取掩膜区域内的像素
    region_pixels = gray_image[mask > 0]
    if len(region_pixels) < 10:
        return {"contrast": 0.0, "homogeneity": 0.0, "energy": 0.0, "correlation": 0.0}

    # 获取边界框以提取ROI
    coords = np.where(mask > 0)
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()

    roi = gray_image[y_min : y_max + 1, x_min : x_max + 1]
    roi_mask = mask[y_min : y_max + 1, x_min : x_max + 1]

    # 量化灰度图像为16级（GLCM标准做法）
    roi_quantized = np.floor(roi.astype(np.float64) * 15 / 255).astype(np.uint8)

    # 计算GLCM（灰度共生矩阵）
    # 使用4个方向：0°, 45°, 90°, 135°
    glcm_features = compute_glcm(roi_quantized, roi_mask)

    return glcm_features


def compute_glcm(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    计算简化版GLCM纹理特征
    """
    # 简化的GLCM实现，使用单方向（0度，水平）
    distances = [1]
    angles = [0]  # 水平方向

    # 量化级别
    levels = 16

    # 初始化GLCM矩阵
    glcm = np.zeros((levels, levels), dtype=np.float64)

    # 遍历图像计算共现矩阵
    h, w = image.shape
    for d in distances:
        for i in range(h):
            for j in range(w - d):
                if mask[i, j] > 0 and mask[i, j + d] > 0:
                    val1 = image[i, j]
                    val2 = image[i, j + d]
                    glcm[val1, val2] += 1
                    glcm[val2, val1] += 1  # 对称

    # 归一化
    total = glcm.sum()
    if total == 0:
        return {"contrast": 0.0, "homogeneity": 0.0, "energy": 0.0, "correlation": 0.0}
    glcm = glcm / total

    # 计算纹理特征
    # 对比度
    contrast = 0.0
    for i in range(levels):
        for j in range(levels):
            contrast += (i - j) ** 2 * glcm[i, j]

    # 均匀性（逆差矩）
    homogeneity = 0.0
    for i in range(levels):
        for j in range(levels):
            homogeneity += glcm[i, j] / (1 + (i - j) ** 2)

    # 能量（ASM角二阶矩）
    energy = 0.0
    for i in range(levels):
        for j in range(levels):
            energy += glcm[i, j] ** 2

    # 相关性
    mean_i = 0.0
    mean_j = 0.0
    std_i = 0.0
    std_j = 0.0
    for i in range(levels):
        row_sum = glcm[i, :].sum()
        mean_i += i * row_sum
        std_i += (i**2) * row_sum
    mean_j = mean_i  # 对称
    std_j = std_i
    std_i = np.sqrt(std_i - mean_i**2)
    std_j = np.sqrt(std_j - mean_j**2)

    correlation = 0.0
    if std_i > 0 and std_j > 0:
        for i in range(levels):
            for j in range(levels):
                correlation += (i - mean_i) * (j - mean_j) * glcm[i, j]
        correlation = correlation / (std_i * std_j)

    return {
        "contrast": float(contrast),
        "homogeneity": float(homogeneity),
        "energy": float(energy),
        "correlation": float(correlation),
    }


def separate_and_extract_organoids(
    binary_mask: np.ndarray,
    gray_image: np.ndarray,
    min_area: float = 500.0,
    max_area: Optional[float] = None,
) -> Tuple[List[OrganoidFeature], List[np.ndarray]]:
    """
    连通域分析与类器官分离，提取每个类器官的核心形态特征

    Args:
        binary_mask: 二值化掩膜
        gray_image: 灰度图像
        min_area: 最小面积阈值（像素），过滤微小杂质
        max_area: 最大面积阈值（像素），None表示不限制

    Returns:
        Tuple[类器官特征列表, 对应的轮廓列表]
    """
    # 使用 cv2.findContours() 执行连通域分析
    # 在二值化图像中，相互连接的像素集合构成独立连通区域
    # 算法通过逐行扫描识别所有连通轮廓并完成编号与标记
    contours, hierarchy = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,  # 只检测外部轮廓
        # cv2.CHAIN_APPROX_SIMPLE,  # 压缩水平、垂直和对角线段
        cv2.CHAIN_APPROX_NONE,  # 压缩水平、垂直和对角线段
    )

    organoid_features = []
    filtered_contours = []  # 保存过滤后的轮廓，用于可视化
    organoid_id = 0

    for contour in contours:
        # 计算面积
        area = cv2.contourArea(contour)

        # 面积阈值过滤：剔除微小杂质与非目标连通区域
        if area < min_area:
            continue

        # 可选：最大面积过滤
        if max_area is not None and area > max_area:
            continue

        organoid_id += 1
        filtered_contours.append(contour)  # 保存通过的轮廓

        # 计算周长
        perimeter = cv2.arcLength(contour, True)

        # 计算圆度：Circularity = 4π×Area / (Perimeter)²
        # 完美圆形结构圆度值为1，形态越不规则，圆度值越小
        if perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter**2)
        else:
            circularity = 0.0

        # 计算等效直径：D = √(4×Area/π)
        equivalent_diameter = np.sqrt(4 * area / np.pi)

        # 创建掩膜以提取该特定类器官的灰度特征
        mask = np.zeros(gray_image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        # 提取掩膜区域内的灰度统计
        mean_gray = cv2.mean(gray_image, mask=mask)[0]

        # 计算标准差
        _, std_val = cv2.meanStdDev(gray_image, mask=mask)
        std_gray = std_val[0][0]

        # 计算最小值、最大值
        min_val, max_val, _, _ = cv2.minMaxLoc(gray_image, mask=mask)

        # 计算边界框
        x, y, w, h = cv2.boundingRect(contour)

        # 计算纵横比
        aspect_ratio = w / h if h > 0 else 0.0

        # 计算实体度：面积/凸包面积
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0

        # 计算范围度：面积/边界框面积
        extent = area / (w * h) if (w * h) > 0 else 0.0

        # 计算质心
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx, cy = 0.0, 0.0

        # 计算边缘清晰度特征
        edge_clarity, edge_gradient_mean, edge_gradient_std = calculate_edge_features(
            gray_image, mask
        )

        # 计算纹理特征
        texture_feats = calculate_texture_features(gray_image, mask)

        # 构建特征对象
        feature = OrganoidFeature(
            organoid_id=organoid_id,
            area=area,
            perimeter=perimeter,
            circularity=circularity,
            equivalent_diameter=equivalent_diameter,
            mean_gray=mean_gray,
            std_gray=std_gray,
            min_gray=min_val,
            max_gray=max_val,
            edge_clarity=edge_clarity,
            edge_gradient_mean=edge_gradient_mean,
            edge_gradient_std=edge_gradient_std,
            texture_contrast=texture_feats["contrast"],
            texture_homogeneity=texture_feats["homogeneity"],
            texture_energy=texture_feats["energy"],
            texture_correlation=texture_feats["correlation"],
            bounding_box_width=w,
            bounding_box_height=h,
            aspect_ratio=aspect_ratio,
            solidity=solidity,
            extent=extent,
            centroid_x=cx,
            centroid_y=cy,
        )

        organoid_features.append(feature)

    return organoid_features, filtered_contours


def analyze_single_image(
    image_path: Path,
    min_area: float = 500.0,
    max_area: Optional[float] = None,
    gaussian_kernel: int = 15,
    adaptive_block_size: int = 151,
    adaptive_c: int = 5,
    output_contour_dir: Optional[Path] = None,
) -> List[OrganoidFeature]:
    """
    分析单张图像，返回该类器官特征列表

    Args:
        image_path: 图像文件路径
        min_area: 最小面积阈值
        max_area: 最大面积阈值
        gaussian_kernel: 高斯滤波核大小
        adaptive_block_size: 自适应阈值块大小
        adaptive_c: 自适应阈值常数
        output_contour_dir: 输出轮廓标记图像目录（可选）

    Returns:
        类器官特征列表
    """
    # 读取图像
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"警告：无法读取图像 {image_path}，跳过")
        return []

    # 图像预处理
    binary_mask, gray_image = preprocess_image(
        image_bgr,
        gaussian_kernel=gaussian_kernel,
        adaptive_block_size=adaptive_block_size,
        adaptive_c=adaptive_c,
    )

    # 连通域分析与特征提取
    features, filtered_contours = separate_and_extract_organoids(
        binary_mask, gray_image, min_area=min_area, max_area=max_area
    )

    # 为每个特征添加图像信息
    for feature in features:
        feature.image_file = image_path.name
        feature.field_of_view = image_path.stem

    # 可选：保存带有轮廓标记的图像
    if output_contour_dir is not None and len(features) > 0:
        output_contour_dir.mkdir(parents=True, exist_ok=True)
        output_image = image_bgr.copy()

        # 为每个类器官绘制轮廓和编号
        for idx, feature in enumerate(features):
            contour = filtered_contours[idx]
            # 绘制绿色轮廓线，线宽为2
            cv2.drawContours(output_image, [contour], -1, (0, 255, 0), 2)

            # 标记编号（红色字体，白色背景框）
            text = f"#{feature.organoid_id}"
            text_pos = (int(feature.centroid_x), int(feature.centroid_y))

            # 获取文字大小
            font_scale = 0.6
            font_thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )

            # 计算背景框位置
            text_x = text_pos[0] - text_width // 2
            text_y = text_pos[1] + text_height // 2

            # 绘制白色背景框
            cv2.rectangle(
                output_image,
                (text_x - 2, text_y - text_height - 2),
                (text_x + text_width + 2, text_y + baseline + 2),
                (255, 255, 255),
                -1,
            )

            # 绘制红色文字
            cv2.putText(
                output_image,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 255),
                font_thickness,
            )

        output_path = (
            output_contour_dir / f"{image_path.stem}_contours{image_path.suffix}"
        )
        cv2.imwrite(str(output_path), output_image)

    return features


def analyze_images_in_directory(
    directory: Path,
    output_excel: Path,
    min_area: float = 500.0,
    max_area: Optional[float] = None,
    gaussian_kernel: int = 15,
    adaptive_block_size: int = 151,
    adaptive_c: int = 5,
    output_contour_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    批量分析目录中的所有图像

    Args:
        directory: 包含图像的目录
        output_excel: 输出Excel文件路径
        min_area: 最小面积阈值
        max_area: 最大面积阈值
        gaussian_kernel: 高斯滤波核大小
        adaptive_block_size: 自适应阈值块大小
        adaptive_c: 自适应阈值常数
        output_contour_dir: 输出轮廓标记图像目录

    Returns:
        包含所有特征的DataFrame
    """
    # 支持的图像格式
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    # 获取所有图像文件
    image_files = sorted(
        [
            f
            for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
    )

    if not image_files:
        print(f"警告：在目录 {directory} 中未找到支持的图像文件")
        return pd.DataFrame()

    print(f"找到 {len(image_files)} 张图像，开始批量分析...")

    all_features = []

    # 使用tqdm显示进度条
    for image_path in tqdm(image_files, desc="分析进度"):
        try:
            features = analyze_single_image(
                image_path,
                min_area=min_area,
                max_area=max_area,
                gaussian_kernel=gaussian_kernel,
                adaptive_block_size=adaptive_block_size,
                adaptive_c=adaptive_c,
                output_contour_dir=output_contour_dir,
            )
            all_features.extend(features)
        except Exception as e:
            print(f"处理 {image_path.name} 时出错：{e}")
            continue

    if not all_features:
        print("警告：未检测到任何类器官")
        return pd.DataFrame()

    # 转换为DataFrame
    df = pd.DataFrame([asdict(f) for f in all_features])

    # 重新排列列顺序
    column_order = [
        "image_file",
        "field_of_view",
        "organoid_id",
        "area",
        "perimeter",
        "circularity",
        "equivalent_diameter",
        "mean_gray",
        "std_gray",
        "min_gray",
        "max_gray",
        "edge_clarity",
        "edge_gradient_mean",
        "edge_gradient_std",
        "texture_contrast",
        "texture_homogeneity",
        "texture_energy",
        "texture_correlation",
        "bounding_box_width",
        "bounding_box_height",
        "aspect_ratio",
        "solidity",
        "extent",
        "centroid_x",
        "centroid_y",
    ]
    df = df[column_order]

    # 导出为Excel
    output_excel.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_excel, index=False, engine="openpyxl")
    print(f"\n分析完成！共检测到 {len(all_features)} 个类器官")
    print(f"结果已导出至：{output_excel}")

    return df


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="类器官图像自动化分析工具：单视野多类器官批量特征提取",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 分析单张图片
  python analysis_images.py --image image.jpg
  
  # 分析目录下所有图片
  python analysis_images.py --directory ./images/
  
  # 指定输出文件
  python analysis_images.py --directory ./images/ -o results.xlsx
  
  # 调整参数
  python analysis_images.py --image image.jpg --min-area 500 --gaussian-kernel 5
        """,
    )

    # 创建互斥参数组：--image 和 --directory 只能指定一个
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image",
        type=str,
        default=None,
        help="输入图像文件路径",
    )
    input_group.add_argument(
        "--directory",
        type=str,
        default=None,
        help="输入图像目录路径",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="输出Excel文件路径（默认：analysis_results.xlsx 或 输入目录/analysis_results.xlsx）",
    )

    parser.add_argument(
        "--min-area",
        type=float,
        default=4000.0,
        help="最小面积阈值（像素），用于过滤微小杂质（默认：500）",
    )

    parser.add_argument(
        "--max-area",
        type=float,
        default=None,
        help="最大面积阈值（像素），None表示不限制（默认：None）",
    )

    parser.add_argument(
        "--gaussian-kernel",
        type=int,
        default=35,
        help="高斯滤波核大小，必须为奇数（默认：15）",
    )

    parser.add_argument(
        "--adaptive-block-size",
        type=int,
        default=151,
        help="自适应阈值块大小，必须为奇数（默认：151）",
    )

    parser.add_argument(
        "--adaptive-c", type=int, default=5, help="自适应阈值常数（默认：5）"
    )

    parser.add_argument(
        "--save-contours", action="store_true", help="保存带有轮廓标记的图像"
    )

    parser.add_argument(
        "--contour-dir",
        type=str,
        default=None,
        help="轮廓标记图像输出目录（默认：输入目录/contours 或 当前目录/contours）",
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()

    # 获取项目根目录（脚本所在目录的父目录，即 paper 的上一级）
    script_dir = Path(__file__).parent
    # 如果脚本在 paper/ 下，则项目根目录是其父目录；否则就是脚本所在目录
    if script_dir.name == "paper":
        project_root = script_dir.parent
    else:
        project_root = script_dir

    # 根据哪个变量有值来决定分析模式
    if args.image:
        input_path = Path(args.image)
        is_file_mode = True
    else:
        input_path = Path(args.directory)
        is_file_mode = False

    # 验证输入路径
    if not input_path.exists():
        print(f"错误：输入路径不存在：{input_path}")
        sys.exit(1)

    # 确定输出文件路径：默认放在项目根目录下
    if args.output:
        output_excel = Path(args.output)
    else:
        output_excel = project_root / "analysis_results.xlsx"

    # 确定轮廓输出目录：默认放在项目根目录下
    output_contour_dir = None
    if args.save_contours:
        if args.contour_dir:
            output_contour_dir = Path(args.contour_dir)
        else:
            output_contour_dir = project_root / "contours"

    print("=" * 60)
    print("类器官图像自动化分析工具")
    print("=" * 60)
    print(f"输入路径：{input_path}")
    print(f"输出文件：{output_excel}")
    print(f"参数设置：")
    print(f"  - 最小面积：{args.min_area} 像素")
    if args.max_area:
        print(f"  - 最大面积：{args.max_area} 像素")
    print(f"  - 高斯核大小：{args.gaussian_kernel}")
    print(f"  - 自适应块大小：{args.adaptive_block_size}")
    print(f"  - 自适应常数：{args.adaptive_c}")
    if args.save_contours:
        print(f"  - 轮廓输出目录：{output_contour_dir}")
    print("=" * 60)

    try:
        if is_file_mode:
            # 处理单个文件
            print(f"\n开始分析单张图像：{input_path.name}")
            features = analyze_single_image(
                input_path,
                min_area=args.min_area,
                max_area=args.max_area,
                gaussian_kernel=args.gaussian_kernel,
                adaptive_block_size=args.adaptive_block_size,
                adaptive_c=args.adaptive_c,
                output_contour_dir=output_contour_dir,
            )

            if features:
                df = pd.DataFrame([asdict(f) for f in features])
                column_order = [
                    "image_file",
                    "field_of_view",
                    "organoid_id",
                    "area",
                    "perimeter",
                    "circularity",
                    "equivalent_diameter",
                    "mean_gray",
                    "std_gray",
                    "min_gray",
                    "max_gray",
                    "edge_clarity",
                    "edge_gradient_mean",
                    "edge_gradient_std",
                    "texture_contrast",
                    "texture_homogeneity",
                    "texture_energy",
                    "texture_correlation",
                    "bounding_box_width",
                    "bounding_box_height",
                    "aspect_ratio",
                    "solidity",
                    "extent",
                    "centroid_x",
                    "centroid_y",
                ]
                df = df[column_order]

                output_excel.parent.mkdir(parents=True, exist_ok=True)
                df.to_excel(output_excel, index=False, engine="openpyxl")
                print(f"\n分析完成！共检测到 {len(features)} 个类器官")
                print(f"结果已导出至：{output_excel}")
            else:
                print(f"\n警告：未在 {input_path.name} 中检测到类器官")

        else:
            # 处理目录
            analyze_images_in_directory(
                input_path,
                output_excel,
                min_area=args.min_area,
                max_area=args.max_area,
                gaussian_kernel=args.gaussian_kernel,
                adaptive_block_size=args.adaptive_block_size,
                adaptive_c=args.adaptive_c,
                output_contour_dir=output_contour_dir,
            )

    except Exception as e:
        print(f"\n错误：{e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
