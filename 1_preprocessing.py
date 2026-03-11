"""
MCF-7 / MDA-MB-231 3D类器官图像自动质控模块
Image Auto Quality Control Module for Breast Cancer Organoid Project

功能：
- 拉普拉斯方差计算（清晰度）
- 亮度统计分析
- 划痕检测
- 气泡检测
- 综合质量评估
- 批量目录处理

作者: 临床医学硕士项目
日期: 2026-03-10
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
from PIL import Image


# ==================== 默认配置 ====================

DEFAULT_CONFIG = {
    'laplacian_threshold': 50.0,
    'brightness_mean_min': 50.0,
    'brightness_mean_max': 200.0,
    'brightness_std_min': 20.0,
    'scratch_line_threshold': 100,
    'max_allowed_lines': 5,
    'bubble_threshold': 200,
    'min_circularity': 0.7,
    'min_bubble_area': 50
}


# ==================== 核心质控函数 ====================

def calculate_laplacian_variance(image_path: str) -> float:
    """
    计算拉普拉斯方差（清晰度指标）
    值越大表示图像越清晰
    
    Args:
        image_path: 图像文件路径
    
    Returns:
        拉普拉斯方差值
    """
    # 使用PIL读取图像，避免OpenCV的TIFF警告
    image_pil = Image.open(image_path)
    if image_pil.mode in ('RGBA', 'LA'):
        image_pil = image_pil.convert('RGB')
    elif image_pil.mode != 'L':
        image_pil = image_pil.convert('L')
    
    image = np.array(image_pil)
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    
    return float(variance)


def calculate_brightness_stats(image_path: str) -> Dict[str, float]:
    """
    计算亮度统计特征
    
    Args:
        image_path: 图像文件路径
    
    Returns:
        {'mean': 均值, 'std': 标准差, 'min': 最小值, 'max': 最大值}
    """
    image_pil = Image.open(image_path)
    if image_pil.mode in ('RGBA', 'LA'):
        image_pil = image_pil.convert('RGB')
    elif image_pil.mode != 'L':
        image_pil = image_pil.convert('L')
    
    image = np.array(image_pil)
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    return {
        'mean': float(np.mean(gray)),
        'std': float(np.std(gray)),
        'min': float(np.min(gray)),
        'max': float(np.max(gray))
    }


def detect_scratches(image_path: str, 
                    line_threshold: int = 100,
                    max_allowed_lines: int = 5) -> Dict:
    """
    检测划痕和亮线干扰
    使用Canny边缘检测 + 霍夫直线检测
    
    Args:
        image_path: 图像文件路径
        line_threshold: 霍夫直线检测阈值
        max_allowed_lines: 允许的最大直线数量
    
    Returns:
        {
            'has_scratch': bool,
            'line_count': int,
            'severity': 'none' | 'medium' | 'high'
        }
    """
    image_pil = Image.open(image_path)
    if image_pil.mode in ('RGBA', 'LA'):
        image_pil = image_pil.convert('RGB')
    elif image_pil.mode != 'L':
        image_pil = image_pil.convert('L')
    
    image = np.array(image_pil)
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    edges = cv2.Canny(gray, 50, 150)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                           threshold=line_threshold, 
                           minLineLength=50, 
                           maxLineGap=10)
    
    scratch_info = {
        'has_scratch': False,
        'line_count': 0,
        'severity': 'none'
    }
    
    if lines is not None:
        scratch_info['line_count'] = len(lines)
        if len(lines) > max_allowed_lines:
            scratch_info['has_scratch'] = True
            scratch_info['severity'] = 'high' if len(lines) > 10 else 'medium'
    
    return scratch_info


def detect_bubbles(image_path: str,
                  threshold: int = 200,
                  min_circularity: float = 0.7,
                  min_area: int = 50) -> Dict:
    """
    检测气泡（圆形亮斑）
    
    Args:
        image_path: 图像文件路径
        threshold: 二值化阈值
        min_circularity: 最小圆形度
        min_area: 最小面积
    
    Returns:
        {
            'has_bubble': bool,
            'bubble_count': int,
            'severity': 'low' | 'medium' | 'high'
        }
    """
    image_pil = Image.open(image_path)
    if image_pil.mode in ('RGBA', 'LA'):
        image_pil = image_pil.convert('RGB')
    elif image_pil.mode != 'L':
        image_pil = image_pil.convert('L')
    
    image = np.array(image_pil)
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bubble_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity > min_circularity:
                    bubble_count += 1
    
    return {
        'has_bubble': bubble_count > 0,
        'bubble_count': bubble_count,
        'severity': 'high' if bubble_count > 5 else ('medium' if bubble_count > 2 else 'low')
    }


def evaluate_image(
    image_path: str,
    laplacian_threshold: float = 50.0,
    brightness_mean_min: float = 50.0,
    brightness_mean_max: float = 200.0,
    brightness_std_min: float = 20.0,
    scratch_line_threshold: int = 100,
    max_allowed_lines: int = 5,
    bubble_threshold: int = 200,
    min_circularity: float = 0.7,
    min_bubble_area: int = 50
) -> Dict:
    """
    综合评估单张图像质量
    
    Args:
        image_path: 图像文件路径
        laplacian_threshold: 拉普拉斯方差阈值
        brightness_mean_min: 亮度均值下限
        brightness_mean_max: 亮度均值上限
        brightness_std_min: 亮度标准差下限
        scratch_line_threshold: 划痕检测阈值
        max_allowed_lines: 允许的最大直线数量
        bubble_threshold: 气泡检测阈值
        min_circularity: 气泡最小圆形度
        min_bubble_area: 气泡最小面积
    
    Returns:
        {
            'passed': bool,
            'laplacian_var': float,
            'brightness_mean': float,
            'brightness_std': float,
            'has_scratch': bool,
            'scratch_count': int,
            'scratch_severity': str,
            'has_bubble': bool,
            'bubble_count': int,
            'bubble_severity': str,
            'fail_reasons': List[str],
            'error': bool
        }
    """
    image_pil = Image.open(image_path)
    if image_pil.mode in ('RGBA', 'LA'):
        image_pil = image_pil.convert('RGB')
    elif image_pil.mode != 'L':
        image_pil = image_pil.convert('L')
    
    image = np.array(image_pil)
    
    fail_reasons = []
    
    # 1. 清晰度检测
    lap_var = calculate_laplacian_variance(image_path)
    if lap_var < laplacian_threshold:
        fail_reasons.append(
            f"清晰度不足 (拉普拉斯方差: {lap_var:.2f} < {laplacian_threshold})"
        )
    
    # 2. 亮度检测
    brightness = calculate_brightness_stats(image_path)
    if not (brightness_mean_min <= brightness['mean'] <= brightness_mean_max):
        fail_reasons.append(
            f"亮度均值异常 ({brightness['mean']:.2f} 不在 [{brightness_mean_min}, {brightness_mean_max}] 区间)"
        )
    if brightness['std'] < brightness_std_min:
        fail_reasons.append(
            f"对比度不足 (标准差: {brightness['std']:.2f} < {brightness_std_min})"
        )
    
    # 3. 划痕检测
    scratch_info = detect_scratches(
        image_path,
        line_threshold=scratch_line_threshold,
        max_allowed_lines=max_allowed_lines
    )
    if scratch_info['has_scratch'] and scratch_info['severity'] == 'high':
        fail_reasons.append(
            f"检测到严重划痕干扰 (直线数: {scratch_info['line_count']})"
        )
    
    # 4. 气泡检测
    bubble_info = detect_bubbles(
        image_path,
        threshold=bubble_threshold,
        min_circularity=min_circularity,
        min_area=min_bubble_area
    )
    if bubble_info['has_bubble'] and bubble_info['severity'] == 'high':
        fail_reasons.append(
            f"检测到严重气泡干扰 (气泡数: {bubble_info['bubble_count']})"
        )
    
    return {
        'passed': len(fail_reasons) == 0,
        'laplacian_var': lap_var,
        'brightness_mean': brightness['mean'],
        'brightness_std': brightness['std'],
        'has_scratch': scratch_info['has_scratch'],
        'scratch_count': scratch_info['line_count'],
        'scratch_severity': scratch_info['severity'],
        'has_bubble': bubble_info['has_bubble'],
        'bubble_count': bubble_info['bubble_count'],
        'bubble_severity': bubble_info['severity'],
        'fail_reasons': fail_reasons,
        'error': False
    }


# ==================== 批量处理函数 ====================

def find_image_files(directory: str, 
                    extensions: Tuple[str, ...] = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')) -> List[Path]:
    """
    在指定目录下查找所有图像文件
    
    Args:
        directory: 搜索目录
        extensions: 支持的图像格式
    
    Returns:
        图像文件路径列表
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise ValueError(f"目录不存在: {directory}")
    
    image_files = []
    for ext in extensions:
        image_files.extend(dir_path.rglob(f"*{ext}"))
    
    return sorted(image_files)


def evaluate_directory(
    directory: str,
    laplacian_threshold: float = 50.0,
    brightness_mean_min: float = 50.0,
    brightness_mean_max: float = 200.0,
    brightness_std_min: float = 20.0,
    scratch_line_threshold: int = 100,
    max_allowed_lines: int = 5,
    bubble_threshold: int = 200,
    min_circularity: float = 0.7,
    min_bubble_area: int = 50,
    verbose: bool = True
) -> Dict:
    """
    批量评估目录下所有图像的5项指标
    
    Args:
        directory: 图像目录路径
        laplacian_threshold: 拉普拉斯方差阈值
        brightness_mean_min: 亮度均值下限
        brightness_mean_max: 亮度均值上限
        brightness_std_min: 亮度标准差下限
        scratch_line_threshold: 划痕检测阈值
        max_allowed_lines: 允许的最大直线数量
        bubble_threshold: 气泡检测阈值
        min_circularity: 气泡最小圆形度
        min_bubble_area: 气泡最小面积
        verbose: 是否显示进度条
    
    Returns:
        {
            'total': int,
            'passed': int,
            'failed': int,
            'results': List[Dict],
            'summary': Dict
        }
    """
    image_files = find_image_files(directory)
    
    if verbose:
        print(f"发现 {len(image_files)} 张图像待评估...")
    
    results = []
    passed_count = 0
    failed_count = 0
    
    for img_path in tqdm(image_files, desc="评估进度", disable=not verbose):
        try:
            result = evaluate_image(
                str(img_path),
                laplacian_threshold=laplacian_threshold,
                brightness_mean_min=brightness_mean_min,
                brightness_mean_max=brightness_mean_max,
                brightness_std_min=brightness_std_min,
                scratch_line_threshold=scratch_line_threshold,
                max_allowed_lines=max_allowed_lines,
                bubble_threshold=bubble_threshold,
                min_circularity=min_circularity,
                min_bubble_area=min_bubble_area
            )
            result['filename'] = img_path.name
            result['filepath'] = str(img_path)
            
            if result['passed']:
                passed_count += 1
            else:
                failed_count += 1
            
            results.append(result)
            
        except Exception as e:
            results.append({
                'filename': img_path.name,
                'filepath': str(img_path),
                'passed': False,
                'error': True,
                'fail_reasons': [str(e)]
            })
            failed_count += 1
    
    summary = {
        'total': len(results),
        'passed': passed_count,
        'failed': failed_count,
        'pass_rate': passed_count / len(results) * 100 if results else 0,
        'avg_laplacian': np.mean([r['laplacian_var'] for r in results if not r.get('error')]) if results else 0,
        'avg_brightness_mean': np.mean([r['brightness_mean'] for r in results if not r.get('error')]) if results else 0,
        'avg_brightness_std': np.mean([r['brightness_std'] for r in results if not r.get('error')]) if results else 0,
        'total_scratches': sum(1 for r in results if r.get('has_scratch')),
        'total_bubbles': sum(1 for r in results if r.get('has_bubble'))
    }
    
    return {
        'total': len(results),
        'passed': passed_count,
        'failed': failed_count,
        'results': results,
        'summary': summary
    }


def export_results_to_csv(results: Dict, output_path: str) -> None:
    """
    将评估结果导出为CSV文件
    
    Args:
        results: evaluate_directory返回的结果字典
        output_path: 输出CSV文件路径
    """
    import pandas as pd
    
    df = pd.DataFrame(results['results'])
    
    columns = [
        'filename', 'filepath', 'passed', 'laplacian_var', 
        'brightness_mean', 'brightness_std', 'has_scratch', 
        'scratch_count', 'has_bubble', 'bubble_count', 'fail_reasons'
    ]
    
    available_columns = [col for col in columns if col in df.columns]
    df = df[available_columns]
    
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"结果已导出到: {output_path}")


def print_summary(results: Dict) -> None:
    """
    打印评估摘要
    
    Args:
        results: evaluate_directory返回的结果字典
    """
    summary = results['summary']
    
    print("\n" + "=" * 60)
    print("评估摘要")
    print("=" * 60)
    print(f"总图像数: {summary['total']}")
    print(f"通过质控: {summary['passed']} ({summary['pass_rate']:.1f}%)")
    print(f"未通过质控: {summary['failed']}")
    print(f"\n平均指标:")
    print(f"  拉普拉斯方差: {summary['avg_laplacian']:.2f}")
    print(f"  亮度均值: {summary['avg_brightness_mean']:.2f}")
    print(f"  亮度标准差: {summary['avg_brightness_std']:.2f}")
    print(f"\n问题统计:")
    print(f"  检测到划痕: {summary['total_scratches']} 张")
    print(f"  检测到气泡: {summary['total_bubbles']} 张")
    print("=" * 60)


# ==================== 使用示例 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("图像自动质控模块示例")
    print("=" * 60)
    
    # 示例1: 单张图像完整评估
    print("\n示例1: 完整质量评估")
    image_path = r".\data\MCF7复苏_0_24h\复苏24h\MCF7_0_24h_CT_10x.tif"
    
    result = evaluate_image(image_path)
    
    if result['error']:
        print(f"错误: {result['fail_reasons']}")
    else:
        if result['passed']:
            print("✅ 图像质量合格")
        else:
            print("❌ 图像质量不合格")
            for reason in result['fail_reasons']:
                print(f"   - {reason}")
        
        print(f"\n详细指标:")
        print(f"  拉普拉斯方差: {result['laplacian_var']:.2f}")
        print(f"  亮度均值: {result['brightness_mean']:.2f}")
        print(f"  亮度标准差: {result['brightness_std']:.2f}")
        print(f"  划痕检测: {result['has_scratch']} (数量: {result['scratch_count']})")
        print(f"  气泡检测: {result['has_bubble']} (数量: {result['bubble_count']})")
    
    # 示例2: 单独评估清晰度
    print("\n示例2: 单独评估清晰度")
    lap_var = calculate_laplacian_variance(image_path)
    print(f"拉普拉斯方差: {lap_var:.2f}")
    
    # 示例3: 单独评估亮度
    print("\n示例3: 单独评估亮度")
    brightness = calculate_brightness_stats(image_path)
    print(f"亮度统计: {brightness}")
    
    # 示例4: 单独检测划痕
    print("\n示例4: 单独检测划痕")
    scratch = detect_scratches(image_path)
    print(f"划痕检测结果: {scratch}")
    
    # 示例5: 单独检测气泡
    print("\n示例5: 单独检测气泡")
    bubble = detect_bubbles(image_path)
    print(f"气泡检测结果: {bubble}")
    
    # 示例6: 批量评估目录
    print("\n示例6: 批量评估目录")
    print("评估目录下所有图像的5项指标...")
    directory = r".\data\MCF7复苏_0_24h\复苏24h"
    
    try:
        results = evaluate_directory(directory)
        print_summary(results)
        
        # 导出结果到CSV
        export_results_to_csv(results, "./qc_results.csv")
        
    except Exception as e:
        print(f"批量处理失败: {e}")
