"""图像验证模块

提供图像文件命名验证、读取测试、完整性检查和质量检测功能。
"""

import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

RAW_IMAGE_DIR = Path(".") / "data" / "raw"

PATTERN = "{cell}_{id}_{time}_{passage}_{well}_{location}_{magnification}.tif"


# =============================================================================
# 文件命名验证
# =============================================================================


def validate_filename(filename: str, pattern: str) -> bool:
    """检验文件名是否符合预设模板"""
    regex_pattern = pattern.replace("{cell}", r"(MCF7|MB231)")
    regex_pattern = regex_pattern.replace("{id}", r"\d+")
    regex_pattern = regex_pattern.replace("{time}", r"(0|6|24|48)h")
    regex_pattern = regex_pattern.replace("{passage}", r"(P0|P1|P2)")
    regex_pattern = regex_pattern.replace("{well}", r"W1")
    regex_pattern = regex_pattern.replace("{location}", r"(CT|LL|LR|UL|UR)")
    regex_pattern = regex_pattern.replace("{magnification}", r"(10x|20x|40x)")
    return bool(re.match(f"^{regex_pattern}$", filename))


def validate_directory_filenames(directory: Path, pattern: str) -> dict:
    """批量检验目录下所有图像文件名是否符合命名规范

    Args:
        directory: 要检验的目录路径 (Path 对象)
        pattern: 文件名模板字符串

    Returns:
        dict: 包含验证结果的字典
            - valid (list): 符合命名规范的文件名列表
            - invalid (list): 不符合命名规范的文件名列表
    """
    valid = []
    invalid = []
    for file_path in directory.iterdir():
        if file_path.is_file():
            filename = file_path.name
            if validate_filename(filename, pattern):
                valid.append(filename)
            else:
                invalid.append(filename)
                print(f"文件名异常：{filename}")
    print(f"文件名正确：{len(valid)}")
    print(f"文件名异常：{len(invalid)}")
    return {"valid": valid, "invalid": invalid}


# =============================================================================
# 图像读取测试
# =============================================================================


def test_read_images(directory: Path) -> dict:
    """测试读取目录下所有图像文件并输出基本信息

    Args:
        directory: 要测试的目录路径 (Path 对象)

    Returns:
        dict: 包含读取结果的字典
            - readable (list): 可以正常读取的图片信息列表
            - unreadable (list): 无法正常读取的文件名列表
    """
    readable = []
    unreadable = []

    for file_path in sorted(directory.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in [".tif"]:
            try:
                with Image.open(file_path) as img:
                    img.load()
                    info = {
                        "filename": file_path.name,
                        "format": img.format,
                        "mode": img.mode,
                        "size": img.size,
                        "width": img.width,
                        "height": img.height,
                    }
                    readable.append(info)
                    print(
                        f"✓ {info['filename']}: {info['format']} | {info['mode']} | {info['width']}x{info['height']}"
                    )
            except Exception as e:
                unreadable.append(file_path.name)
                print(f"✗ {file_path.name}: 读取失败 - {e}")

    print(f"\n可正常读取：{len(readable)}")
    print(f"读取失败：{len(unreadable)}")
    return {"readable": readable, "unreadable": unreadable}


# =============================================================================
# 缺失图像检查
# =============================================================================


def check_missing_images(directory: Path) -> dict:
    """检查图片文件是否缺失，验证数据完整性

    检查规则：
    1. 每个细胞系 + 编号组合必须有 24h 和 48h 两个时间点
    2. 在每个时间点下必须有 5 个位置（CT, LL, LR, UL, UR）
    3. 在每个位置下必须有 3 个放大倍数（10x, 20x, 40x）

    Args:
        directory: 要检查的目录路径 (Path 对象)

    Returns:
        dict: 包含检查结果的字典
            - complete (list): 完整的图片组合列表
            - missing (list): 缺失的图片文件名列表
    """
    required_cells = ["MCF7", "MB231"]
    required_times = ["24h", "48h"]
    required_locations = ["CT", "LL", "LR", "UL", "UR"]
    required_magnifications = ["10x", "20x", "40x"]

    # 建立完整的应该存在的文件名数组
    expected_files = set()
    for cell in required_cells:
        id_nums = set()
        for f in directory.iterdir():
            if f.is_file() and f.suffix.lower() == ".tif":
                match = re.match(rf"^{cell}_(\d+)_", f.name)
                if match:
                    id_nums.add(match.group(1))

        for id_num in id_nums:
            for time in required_times:
                for location in required_locations:
                    for mag in required_magnifications:
                        filename = f"{cell}_{id_num}_{time}_{location}_{mag}.tif"
                        expected_files.add(filename)

    # 扫描文件夹获取实际存在的文件
    actual_files = {
        f.name
        for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() == ".tif"
    }

    # 计算缺失的文件
    missing_files = sorted(expected_files - actual_files)

    # 统计完整的样本
    complete = set()
    for cell in required_cells:
        for id_num in set(
            m.group(1) for f in actual_files if (m := re.match(rf"^{cell}_(\d+)_", f))
        ):
            is_complete = all(
                f"{cell}_{id_num}_{time}_{location}_{mag}.tif" in actual_files
                for time in required_times
                for location in required_locations
                for mag in required_magnifications
            )
            if is_complete:
                complete.add(f"{cell}_{id_num}")

    print(f"完整数据集：{len(complete)} 个")
    for item in sorted(complete):
        print(f"  ✓ {item}")

    print(f"\n缺失图片：{len(missing_files)} 个")
    for filename in missing_files:
        print(f"  ✗ {filename}")

    return {"complete": sorted(complete), "missing": missing_files}


# =============================================================================
# 图像信息一致性检查
# =============================================================================


def check_image_info_consistency(directory: Path) -> dict:
    """检查所有图片信息的一致性

    Args:
        directory: 要检查的目录路径 (Path 对象)

    Returns:
        dict: 包含检查结果的字典
            - consistent (bool): 所有图片信息是否一致
            - total (int): 成功读取的图片总数
            - reference (dict): 参考图片信息
            - inconsistent (list): 信息不一致的图片列表
            - errors (list): 读取失败的图片列表
    """
    image_info = {}
    errors = []
    reference = None

    # 读取所有图片的基本信息
    for file_path in sorted(directory.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() == ".tif":
            try:
                with Image.open(file_path) as img:
                    img.load()
                    info = {
                        "format": img.format,
                        "mode": img.mode,
                        "size": img.size,
                        "width": img.width,
                        "height": img.height,
                    }
                    image_info[file_path.name] = info
                    # 以第一张图片为参考
                    if reference is None:
                        reference = info
            except Exception as e:
                errors.append({"filename": file_path.name, "error": str(e)})

    # 检查所有图片是否与参考一致
    inconsistent = []
    for filename, info in image_info.items():
        if (
            info["format"] != reference["format"]
            or info["mode"] != reference["mode"]
            or info["size"] != reference["size"]
        ):
            inconsistent.append(
                {
                    "filename": filename,
                    "info": info,
                    "expected": reference,
                }
            )

    # 打印结果
    print(f"成功读取图片：{len(image_info)} 个")
    print(f"读取失败：{len(errors)} 个")

    print(
        f"\n参考信息：{reference['format']} | {reference['mode']} | "
        f"{reference['size'][0]}x{reference['size'][1]}"
    )

    if inconsistent:
        print(f"\n信息不一致的图片：{len(inconsistent)} 个")
        for item in inconsistent:
            print(
                f"  ✗ {item['filename']}: "
                f"{item['info']['format']} | {item['info']['mode']} | "
                f"{item['info']['size'][0]}x{item['info']['size'][1]}"
            )
    else:
        print("\n所有图片信息一致 ✓")

    if errors:
        print(f"\n读取失败的图片：{len(errors)} 个")
        for err in errors:
            print(f"  ✗ {err['filename']}: {err['error']}")

    return {
        "consistent": len(inconsistent) == 0,
        "total": len(image_info),
        "reference": reference,
        "inconsistent": inconsistent,
        "errors": errors,
    }


# =============================================================================
# 图像质量检测
# =============================================================================


def load_image(path: Path) -> np.ndarray:
    """使用 PIL 读取图像并转换为 BGR 格式（兼容 OpenCV）

    Args:
        path: 图像文件路径

    Returns:
        np.ndarray: BGR 格式的图像数组
    """
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    rgb = np.array(img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def detect_blur(image: np.ndarray, threshold: float = 100.0) -> dict:
    """使用拉普拉斯方差检测图像模糊度

    Args:
        image: 输入图像（BGR 格式）
        threshold: 模糊阈值

    Returns:
        dict: 检测结果
            - is_blur (bool): 是否模糊
            - variance (float): 拉普拉斯方差值
            - threshold (float): 使用的阈值
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return {
        "is_blur": variance < threshold,
        "variance": float(variance),
        "threshold": threshold,
    }


def detect_bubbles(
    image: np.ndarray,
    min_area: int = 100,
    circularity_threshold: float = 0.7,
) -> dict:
    """检测图像中的气泡（圆形暗区）

    Args:
        image: 输入图像（BGR 格式）
        min_area: 最小气泡面积（像素）
        circularity_threshold: 圆度阈值（0-1）

    Returns:
        dict: 检测结果
            - count (int): 检测到的气泡数量
            - bubbles (list): 气泡信息列表
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bubbles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity >= circularity_threshold:
            x, y, w, h = cv2.boundingRect(contour)
            bubbles.append(
                {
                    "center": (x + w // 2, y + h // 2),
                    "size": (w, h),
                    "area": float(area),
                    "circularity": float(circularity),
                }
            )

    return {"count": len(bubbles), "bubbles": bubbles}


def detect_scratches(
    image: np.ndarray,
    min_length: int = 20,
    line_threshold: int = 50,
) -> dict:
    """检测图像中的划痕（线性特征）

    Args:
        image: 输入图像（BGR 格式）
        min_length: 最小划痕长度（像素）
        line_threshold: 直线检测阈值（累加器阈值）

    Returns:
        dict: 检测结果
            - count (int): 检测到的划痕数量
            - scratches (list): 划痕信息列表
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    edges = cv2.dilate(edges, kernel, iterations=1)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=int(line_threshold),
        minLineLength=min_length,
        maxLineGap=5,
    )

    scratches = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            scratches.append(
                {
                    "start": (int(x1), int(y1)),
                    "end": (int(x2), int(y2)),
                    "length": float(length),
                }
            )

    return {"count": len(scratches), "scratches": scratches}


def detect_brightness(image: np.ndarray) -> dict:
    """检测图像亮度

    Args:
        image: 输入图像（BGR 格式）

    Returns:
        dict: 检测结果
            - mean (float): 平均亮度值 (0-255)
            - std (float): 亮度标准差
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    return {"mean": mean, "std": std}


def detect_contrast(image: np.ndarray) -> dict:
    """使用拉普拉斯方差检测图像对比度

    Args:
        image: 输入图像（BGR 格式）

    Returns:
        dict: 检测结果
            - variance (float): 拉普拉斯方差值
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return {"variance": float(variance)}


def check_image_quality(
    image_path: Path,
    blur_threshold: float = 100.0,
    min_bubble_area: int = 100,
    min_scratch_length: int = 20,
) -> dict:
    """综合检查单张图片的质量

    Args:
        image_path: 图片文件路径
        blur_threshold: 模糊检测阈值
        min_bubble_area: 最小气泡面积
        min_scratch_length: 最小划痕长度

    Returns:
        dict: 质量检查结果
    """
    try:
        image = load_image(image_path)
        if image is None:
            return {
                "filename": image_path.name,
                "success": False,
                "error": "无法读取图像",
            }

        blur_result = detect_blur(image, blur_threshold)
        bubble_result = detect_bubbles(image, min_bubble_area)
        scratch_result = detect_scratches(image, min_scratch_length)

        is_qualified = (
            not blur_result["is_blur"]
            and bubble_result["count"] == 0
            and scratch_result["count"] == 0
        )

        return {
            "filename": image_path.name,
            "success": True,
            "is_qualified": is_qualified,
            "blur": blur_result,
            "bubbles": bubble_result,
            "scratches": scratch_result,
        }
    except Exception as e:
        return {
            "filename": image_path.name,
            "success": False,
            "error": str(e),
        }


def inspect_directory_quality(
    directory: Path,
    blur_threshold: float = 100.0,
    min_bubble_area: int = 100,
    min_scratch_length: int = 20,
    sample_count: int = None,
) -> dict:
    """批量检查目录下所有图片的质量

    Args:
        directory: 要检查的目录路径
        blur_threshold: 模糊检测阈值
        min_bubble_area: 最小气泡面积
        min_scratch_length: 最小划痕长度
        sample_count: 采样数量（None 表示检查全部）

    Returns:
        dict: 质量检查结果
            - total (int): 检查的图片总数
            - qualified (int): 合格的图片数量
            - unqualified (int): 不合格的图片数量
            - errors (int): 读取失败的图片数量
            - results (list): 详细结果列表
            - summary (dict): 汇总统计
    """
    image_files = sorted(
        [f for f in directory.iterdir() if f.is_file() and f.suffix.lower() == ".tif"]
    )

    if sample_count and sample_count < len(image_files):
        import random

        random.seed(42)
        image_files = random.sample(image_files, sample_count)

    results = []
    qualified_count = 0
    unqualified_count = 0
    error_count = 0
    blur_count = 0
    bubble_count = 0
    scratch_count = 0
    variance_values = []

    for i, image_path in enumerate(image_files):
        result = check_image_quality(
            image_path,
            blur_threshold,
            min_bubble_area,
            min_scratch_length,
        )
        results.append(result)

        if not result["success"]:
            error_count += 1
        else:
            variance_values.append(result["blur"]["variance"])
            if result["is_qualified"]:
                qualified_count += 1
            else:
                unqualified_count += 1
                if result["blur"]["is_blur"]:
                    blur_count += 1
                if result["bubbles"]["count"] > 0:
                    bubble_count += 1
                if result["scratches"]["count"] > 0:
                    scratch_count += 1

        variance = result["blur"]["variance"] if result["success"] else None
        print(
            f"[{i + 1}/{len(image_files)}] {image_path.name}: var={variance:.1f}",
            end="",
        )
        if not result["success"]:
            print(f" - 读取失败：{result['error']}")
        elif result["is_qualified"]:
            print(" - 合格 ✓")
        else:
            issues = []
            if result["blur"]["is_blur"]:
                issues.append("模糊")
            if result["bubbles"]["count"] > 0:
                issues.append(f"气泡 ({result['bubbles']['count']}个)")
            if result["scratches"]["count"] > 0:
                issues.append(f"划痕 ({result['scratches']['count']}条)")
            print(f" - {', '.join(issues)}")

    # 计算方差统计
    if variance_values:
        variance_mean = np.mean(variance_values)
        variance_std = np.std(variance_values)
        variance_min = np.min(variance_values)
        variance_max = np.max(variance_values)
    else:
        variance_mean = variance_std = variance_min = variance_max = 0

    summary = {
        "total": len(image_files),
        "qualified": qualified_count,
        "unqualified": unqualified_count,
        "errors": error_count,
        "blur_images": blur_count,
        "bubble_images": bubble_count,
        "scratch_images": scratch_count,
        "qualification_rate": qualified_count / len(image_files) if image_files else 0,
        "variance_mean": variance_mean,
        "variance_std": variance_std,
        "variance_min": variance_min,
        "variance_max": variance_max,
    }

    print("\n" + "=" * 50)
    print(f"检查总数：{summary['total']} 个")
    print(
        f"合格：{summary['qualified']} 个 ({summary['qualification_rate'] * 100:.1f}%)"
    )
    print(f"不合格：{summary['unqualified']} 个")
    print(f"读取失败：{summary['errors']} 个")
    print(f"\n问题统计:")
    print(f"  模糊：{summary['blur_images']} 个")
    print(f"  气泡：{summary['bubble_images']} 个")
    print(f"  划痕：{summary['scratch_images']} 个")
    print(f"\n拉普拉斯方差统计:")
    print(f"  平均值：{summary['variance_mean']:.1f}")
    print(f"  标准差：{summary['variance_std']:.1f}")
    print(f"  最小值：{summary['variance_min']:.1f}")
    print(f"  最大值：{summary['variance_max']:.1f}")

    return {
        "total": summary["total"],
        "qualified": summary["qualified"],
        "unqualified": summary["unqualified"],
        "errors": summary["errors"],
        "results": results,
        "summary": summary,
    }


# =============================================================================
# 主函数
# =============================================================================


def main():
    """执行所有验证检查"""
    # 1. 图片文件名校验
    print("\n" + "=" * 50 + "\n")
    validate_directory_filenames(RAW_IMAGE_DIR, PATTERN)

    # 2. 图片文件读取校验
    print("\n" + "=" * 50 + "\n")
    test_read_images(RAW_IMAGE_DIR)

    # 3. 图片文件缺失检测
    print("\n" + "=" * 50 + "\n")
    check_missing_images(RAW_IMAGE_DIR)

    # 4. 图片信息一致性检验
    print("\n" + "=" * 50 + "\n")
    check_image_info_consistency(RAW_IMAGE_DIR)

    # 5. 图像质量检测
    print("\n" + "=" * 50 + "\n")
    inspect_directory_quality(RAW_IMAGE_DIR)

    print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()
