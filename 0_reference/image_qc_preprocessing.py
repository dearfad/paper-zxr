"""
MCF-7 / MDA-MB-231 3D类器官图像质控与预处理系统
Image Quality Control & Preprocessing Pipeline for Breast Cancer Organoid Project

功能模块：
1. 人工初筛记录模块 (Manual QC Log)
2. 自动质控模块 (Auto QC - 拉普拉斯方差、亮度统计)
3. 图像预处理模块 (Preprocessing - 尺寸统一、CLAHE增强、去噪、对齐)
4. 批量处理工作流 (Batch Processing Workflow)

作者: 临床医学硕士项目
日期: 2026-03-02
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json
import logging
from dataclasses import dataclass, asdict
import argparse
from tqdm import tqdm
import shutil

# ==================== 配置类 ====================

@dataclass
class QCConfig:
    """质控参数配置"""
    # 清晰度阈值 (拉普拉斯方差)
    LAPLACIAN_VAR_THRESHOLD: float = 50.0
    
    # 亮度阈值
    BRIGHTNESS_MEAN_MIN: float = 50.0
    BRIGHTNESS_MEAN_MAX: float = 200.0
    BRIGHTNESS_STD_MIN: float = 20.0
    
    # 图像尺寸 (适配CLIP/ViT)
    TARGET_SIZE: Tuple[int, int] = (336, 336)
    CROP_RATIO: float = 0.8  # 中心裁剪比例
    
    # CLAHE参数
    CLAHE_CLIP_LIMIT: float = 2.0
    CLAHE_GRID_SIZE: Tuple[int, int] = (8, 8)
    
    # 去噪参数
    GAUSSIAN_SIGMA: float = 0.5
    MEDIAN_KERNEL_SIZE: int = 3
    
    # 划痕检测参数
    SCRATCH_DETECTION: bool = True
    SCRATCH_THRESHOLD: float = 200  # 亮线检测阈值
    
    # 支持的图像格式
    SUPPORTED_FORMATS: Tuple[str, ...] = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')


# ==================== 1. 人工初筛记录模块 ====================

class ManualQCLogger:
    """
    人工初筛记录器
    用于记录人工检查时发现的不合格图像及其原因
    """
    
    REJECT_REASONS = [
        "图像模糊",
        "曝光过度", 
        "曝光不足",
        "划痕干扰",
        "气泡干扰",
        "杂质污染",
        "视野偏移",
        "细胞脱落",
        "培养失败形态",
        "其他"
    ]
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.output_dir / "manual_qc_log.csv"
        self.notes_file = self.output_dir / "manual_qc_notes.json"
        
        # 初始化或加载现有记录
        self.records = self._load_existing_records()
        self.notes = self._load_existing_notes()
        
    def _load_existing_records(self) -> pd.DataFrame:
        """加载已有记录"""
        if self.log_file.exists():
            return pd.read_csv(self.log_file)
        else:
            return pd.DataFrame(columns=[
                'timestamp', 'sample_id', 'cell_line', 'timepoint', 
                'view_id', 'image_path', 'reject_reason', 'severity', 
                'operator', 'notes', 'reviewed'
            ])
    
    def _load_existing_notes(self) -> Dict:
        """加载详细备注"""
        if self.notes_file.exists():
            with open(self.notes_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def log_rejection(self, 
                     image_path: str,
                     cell_line: str,
                     sample_id: str,
                     timepoint: str,
                     view_id: str,
                     reject_reason: str,
                     severity: str = "high",
                     operator: str = "operator",
                     notes: str = "") -> None:
        """
        记录不合格图像
        
        Args:
            image_path: 图像完整路径
            cell_line: 细胞系类型 ('MCF7' 或 'MDA231')
            sample_id: 样本ID (如: MCF7_001)
            timepoint: 时间点 (如: 24h)
            view_id: 视野编号 (如: F01)
            reject_reason: 不合格原因
            severity: 严重程度
            operator: 操作员
            notes: 详细备注
        """
        if reject_reason not in self.REJECT_REASONS and reject_reason != "其他":
            raise ValueError(f"无效原因: {reject_reason}. 可选: {self.REJECT_REASONS}")
        
        record = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'sample_id': sample_id,
            'cell_line': cell_line,
            'timepoint': timepoint,
            'view_id': view_id,
            'image_path': str(image_path),
            'reject_reason': reject_reason,
            'severity': severity,
            'operator': operator,
            'notes': notes,
            'reviewed': False
        }
        
        self.records = pd.concat([self.records, pd.DataFrame([record])], ignore_index=True)
        
        key = f"{sample_id}_{timepoint}_{view_id}"
        self.notes[key] = {
            'reject_reason': reject_reason,
            'notes': notes,
            'timestamp': record['timestamp']
        }
        
        self._save()
        
        print(f"✗ 已记录不合格图像: {Path(image_path).name} | 原因: {reject_reason}")
    
    def mark_as_passed(self,
                      image_path: str,
                      cell_line: str,
                      sample_id: str,
                      timepoint: str,
                      view_id: str,
                      operator: str = "operator") -> None:
        """标记图像通过人工初筛"""
        record = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'sample_id': sample_id,
            'cell_line': cell_line,
            'timepoint': timepoint,
            'view_id': view_id,
            'image_path': str(image_path),
            'reject_reason': 'PASSED',
            'severity': 'none',
            'operator': operator,
            'notes': '通过人工初筛',
            'reviewed': True
        }
        self.records = pd.concat([self.records, pd.DataFrame([record])], ignore_index=True)
        self._save()
    
    def _save(self):
        """保存记录到文件"""
        self.records.to_csv(self.log_file, index=False)
        with open(self.notes_file, 'w', encoding='utf-8') as f:
            json.dump(self.notes, f, ensure_ascii=False, indent=2)
    
    def get_rejection_summary(self) -> pd.DataFrame:
        """获取不合格原因统计"""
        rejected = self.records[self.records['reject_reason'] != 'PASSED']
        return rejected.groupby(['cell_line', 'reject_reason']).size().unstack(fill_value=0)
    
    def export_rejection_report(self, output_path: str = None):
        """导出不合格图像报告"""
        if output_path is None:
            output_path = self.output_dir / f"rejection_report_{datetime.now().strftime('%Y%m%d')}.xlsx"
        
        rejected = self.records[self.records['reject_reason'] != 'PASSED']
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            rejected.to_excel(writer, sheet_name='不合格记录', index=False)
            self.get_rejection_summary().to_excel(writer, sheet_name='原因统计')
        
        print(f"报告已导出: {output_path}")


# ==================== 2. 自动质控模块 ====================

class AutoImageQC:
    """
    自动图像质控类
    计算拉普拉斯方差、亮度统计、检测划痕/气泡
    """
    
    def __init__(self, config: QCConfig = None):
        self.config = config or QCConfig()
        self.clahe = cv2.createCLAHE(
            clipLimit=self.config.CLAHE_CLIP_LIMIT,
            tileGridSize=self.config.CLAHE_GRID_SIZE
        )
    
    def calculate_laplacian_variance(self, image: np.ndarray) -> float:
        """
        计算拉普拉斯方差 (清晰度指标)
        值越大表示图像越清晰
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        return float(variance)
    
    def calculate_brightness_stats(self, image: np.ndarray) -> Dict[str, float]:
        """
        计算亮度统计特征
        Returns: {'mean': 均值, 'std': 标准差, 'min': 最小值, 'max': 最大值}
        """
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
    
    def detect_scratches(self, image: np.ndarray) -> Dict:
        """
        检测划痕和亮线干扰
        使用霍夫直线检测
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        edges = cv2.Canny(gray, 50, 150)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                               threshold=100, 
                               minLineLength=50, 
                               maxLineGap=10)
        
        scratch_info = {
            'has_scratch': False,
            'line_count': 0,
            'severity': 'none'
        }
        
        if lines is not None:
            scratch_info['line_count'] = len(lines)
            if len(lines) > 5:
                scratch_info['has_scratch'] = True
                scratch_info['severity'] = 'high' if len(lines) > 10 else 'medium'
        
        return scratch_info
    
    def detect_bubbles(self, image: np.ndarray) -> Dict:
        """
        检测气泡 (圆形亮斑)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bubble_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 50:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    if circularity > 0.7:
                        bubble_count += 1
        
        return {
            'has_bubble': bubble_count > 0,
            'bubble_count': bubble_count,
            'severity': 'high' if bubble_count > 5 else ('medium' if bubble_count > 2 else 'low')
        }
    
    def evaluate_image(self, image_path: str) -> Dict:
        """
        综合评估单张图像质量
        
        Returns:
            {
                'passed': bool,
                'laplacian_var': float,
                'brightness_mean': float,
                'brightness_std': float,
                'has_scratch': bool,
                'has_bubble': bool,
                'fail_reasons': List[str]
            }
        """
        image = cv2.imread(str(image_path))
        if image is None:
            return {
                'passed': False,
                'fail_reasons': ['无法读取图像'],
                'error': True
            }
        
        fail_reasons = []
        
        lap_var = self.calculate_laplacian_variance(image)
        if lap_var < self.config.LAPLACIAN_VAR_THRESHOLD:
            fail_reasons.append(f"清晰度不足 (拉普拉斯方差: {lap_var:.2f} < {self.config.LAPLACIAN_VAR_THRESHOLD})")
        
        brightness = self.calculate_brightness_stats(image)
        if not (self.config.BRIGHTNESS_MEAN_MIN <= brightness['mean'] <= self.config.BRIGHTNESS_MEAN_MAX):
            fail_reasons.append(f"亮度均值异常 ({brightness['mean']:.2f} 不在 [{self.config.BRIGHTNESS_MEAN_MIN}, {self.config.BRIGHTNESS_MEAN_MAX}] 区间)")
        if brightness['std'] < self.config.BRIGHTNESS_STD_MIN:
            fail_reasons.append(f"对比度不足 (标准差: {brightness['std']:.2f} < {self.config.BRIGHTNESS_STD_MIN})")
        
        scratch_info = self.detect_scratches(image)
        if scratch_info['has_scratch'] and scratch_info['severity'] == 'high':
            fail_reasons.append(f"检测到严重划痕干扰 (直线数: {scratch_info['line_count']})")
        
        bubble_info = self.detect_bubbles(image)
        if bubble_info['has_bubble'] and bubble_info['severity'] == 'high':
            fail_reasons.append(f"检测到严重气泡干扰 (气泡数: {bubble_info['bubble_count']})")
        
        return {
            'passed': len(fail_reasons) == 0,
            'laplacian_var': lap_var,
            'brightness_mean': brightness['mean'],
            'brightness_std': brightness['std'],
            'has_scratch': scratch_info['has_scratch'],
            'scratch_count': scratch_info['line_count'],
            'has_bubble': bubble_info['has_bubble'],
            'bubble_count': bubble_info['bubble_count'],
            'fail_reasons': fail_reasons,
            'error': False
        }


# ==================== 3. 图像预处理模块 ====================

class ImagePreprocessor:
    """
    图像预处理流水线
    尺寸统一 -> 中心裁剪 -> CLAHE增强 -> 去噪 -> 对齐
    """
    
    def __init__(self, config: QCConfig = None):
        self.config = config or QCConfig()
        self.clahe = cv2.createCLAHE(
            clipLimit=self.config.CLAHE_CLIP_LIMIT,
            tileGridSize=self.config.CLAHE_GRID_SIZE
        )
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """尺寸统一为336x336"""
        return cv2.resize(image, self.config.TARGET_SIZE, interpolation=cv2.INTER_AREA)
    
    def center_crop(self, image: np.ndarray) -> np.ndarray:
        """中心裁剪 (0.8比例)"""
        h, w = image.shape[:2]
        crop_h, crop_w = int(h * self.config.CROP_RATIO), int(w * self.config.CROP_RATIO)
        
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        
        cropped = image[start_h:start_h+crop_h, start_w:start_w+crop_w]
        return cropped
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """CLAHE自适应直方图均衡化"""
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            return self.clahe.apply(image)
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """去噪: 高斯模糊 + 中值滤波"""
        denoised = cv2.GaussianBlur(image, (0, 0), self.config.GAUSSIAN_SIGMA)
        ksize = self.config.MEDIAN_KERNEL_SIZE
        denoised = cv2.medianBlur(denoised, ksize)
        return denoised
    
    def preprocess(self, image_path: str, output_path: str = None) -> np.ndarray:
        """
        完整预处理流程
        
        Args:
            image_path: 输入图像路径
            output_path: 输出路径 (可选)
        
        Returns:
            预处理后的图像数组
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        image = self.center_crop(image)
        image = self.resize_image(image)
        image = self.apply_clahe(image)
        image = self.denoise(image)
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), image)
        
        return image


# ==================== 4. 批量处理工作流 ====================

class BatchProcessor:
    """
    批量处理工作流
    整合人工初筛 -> 自动质控 -> 预处理 -> 数据集划分
    """
    
    def __init__(self, 
                 input_dir: str,
                 output_dir: str,
                 config: QCConfig = None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.config = config or QCConfig()
        
        self.manual_logger = ManualQCLogger(output_dir)
        self.auto_qc = AutoImageQC(config)
        self.preprocessor = ImagePreprocessor(config)
        
        self._create_directory_structure()
        
        self.stats = {
            'total': 0,
            'manual_rejected': 0,
            'auto_rejected': 0,
            'passed': 0,
            'by_cell_line': {'MCF7': {'total': 0, 'passed': 0}, 
                           'MDA231': {'total': 0, 'passed': 0}}
        }
    
    def _create_directory_structure(self):
        """创建标准目录结构"""
        dirs = [
            '01_passed/MCF7',
            '01_passed/MDA231', 
            '02_rejected_manual/MCF7',
            '02_rejected_manual/MDA231',
            '03_rejected_auto/MCF7',
            '03_rejected_auto/MDA231',
            '04_preprocessed/MCF7',
            '04_preprocessed/MDA231',
            'logs',
            'reports'
        ]
        for d in dirs:
            (self.output_dir / d).mkdir(parents=True, exist_ok=True)
    
    def parse_filename(self, filename: str) -> Dict:
        """
        解析标准命名格式: MCF7_001_24h_F01.tif
        Returns: {'cell_line': 'MCF7', 'sample_id': 'MCF7_001', 'timepoint': '24h', 'view_id': 'F01'}
        """
        parts = Path(filename).stem.split('_')
        if len(parts) >= 4:
            cell_line = parts[0]
            sample_num = parts[1]
            timepoint = parts[2]
            view_id = parts[3]
            return {
                'cell_line': cell_line,
                'sample_id': f"{cell_line}_{sample_num}",
                'timepoint': timepoint,
                'view_id': view_id
            }
        return None
    
    def process_single_image(self, image_path: Path) -> Dict:
        """
        处理单张图像的完整流程
        """
        filename = image_path.name
        parsed = self.parse_filename(filename)
        
        if not parsed:
            return {'status': 'error', 'reason': '文件名格式不符合规范', 'path': str(image_path)}
        
        cell_line = parsed['cell_line']
        sample_id = parsed['sample_id']
        timepoint = parsed['timepoint']
        view_id = parsed['view_id']
        
        self.stats['total'] += 1
        self.stats['by_cell_line'][cell_line]['total'] += 1
        
        qc_result = self.auto_qc.evaluate_image(str(image_path))
        
        if not qc_result['passed']:
            self.stats['auto_rejected'] += 1
            reject_dir = self.output_dir / f"03_rejected_auto/{cell_line}"
            shutil.copy2(image_path, reject_dir / filename)
            
            log_entry = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'sample_id': sample_id,
                'cell_line': cell_line,
                'timepoint': timepoint,
                'view_id': view_id,
                'image_path': str(image_path),
                'reject_reason': ' | '.join(qc_result['fail_reasons']),
                'severity': 'auto',
                'operator': 'AutoQC',
                'notes': json.dumps(qc_result, ensure_ascii=False),
                'reviewed': False
            }
            self.manual_logger.records = pd.concat([
                self.manual_logger.records, 
                pd.DataFrame([log_entry])
            ], ignore_index=True)
            
            return {
                'status': 'auto_rejected',
                'cell_line': cell_line,
                'reason': qc_result['fail_reasons'],
                'qc_metrics': qc_result
            }
        
        try:
            preprocessed_dir = self.output_dir / f"04_preprocessed/{cell_line}"
            output_path = preprocessed_dir / f"{sample_id}_{timepoint}_{view_id}_preprocessed.tif"
            
            self.preprocessor.preprocess(str(image_path), str(output_path))
            
            passed_dir = self.output_dir / f"01_passed/{cell_line}"
            shutil.copy2(image_path, passed_dir / filename)
            
            self.stats['passed'] += 1
            self.stats['by_cell_line'][cell_line]['passed'] += 1
            
            return {
                'status': 'passed',
                'cell_line': cell_line,
                'qc_metrics': qc_result,
                'output_path': str(output_path)
            }
            
        except Exception as e:
            return {'status': 'error', 'reason': str(e), 'cell_line': cell_line}
    
    def run_batch(self, cell_line: str = None):
        """
        批量处理所有图像
        
        Args:
            cell_line: 指定处理特定细胞系 ('MCF7' 或 'MDA231')，None则处理全部
        """
        image_files = []
        for ext in self.config.SUPPORTED_FORMATS:
            if cell_line:
                pattern = f"{cell_line}*{ext}"
                image_files.extend(self.input_dir.rglob(pattern))
            else:
                image_files.extend(self.input_dir.rglob(f"*{ext}"))
        
        print(f"\n发现 {len(image_files)} 张图像待处理...")
        print(f"质控标准: 拉普拉斯方差 > {self.config.LAPLACIAN_VAR_THRESHOLD}, 亮度均值 [50, 200], 标准差 > 20\n")
        
        results = []
        for img_path in tqdm(image_files, desc="处理进度"):
            result = self.process_single_image(img_path)
            results.append(result)
        
        self.manual_logger._save()
        
        self._generate_report(results)
        
        return results
    
    def _generate_report(self, results: List[Dict]):
        """生成处理报告"""
        report_path = self.output_dir / f"reports/processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("图像质控与预处理报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("【处理统计】\n")
            f.write(f"总图像数: {self.stats['total']}\n")
            f.write(f"通过质控: {self.stats['passed']} ({self.stats['passed']/max(self.stats['total'],1)*100:.1f}%)\n")
            f.write(f"自动拒绝: {self.stats['auto_rejected']} ({self.stats['auto_rejected']/max(self.stats['total'],1)*100:.1f}%)\n")
            f.write(f"人工拒绝: {self.stats['manual_rejected']}\n\n")
            
            f.write("【细胞系分布】\n")
            for cl, data in self.stats['by_cell_line'].items():
                if data['total'] > 0:
                    f.write(f"{cl}: 总计 {data['total']}, 通过 {data['passed']} ({data['passed']/data['total']*100:.1f}%)\n")
            
            f.write("\n【自动拒绝原因分析】\n")
            auto_rejected = [r for r in results if r.get('status') == 'auto_rejected']
            if auto_rejected:
                reason_counts = {}
                for r in auto_rejected:
                    for reason in r.get('reason', []):
                        reason_counts[reason] = reason_counts.get(reason, 0) + 1
                
                for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
                    f.write(f"  - {reason}: {count}次\n")
        
        print(f"\n报告已生成: {report_path}")
        print(f"统计: 通过 {self.stats['passed']}/{self.stats['total']} ({self.stats['passed']/max(self.stats['total'],1)*100:.1f}%)")


# ==================== 5. 命令行接口 ====================

def main():
    parser = argparse.ArgumentParser(
        description='MCF-7/MDA-MB-231 3D类器官图像质控与预处理系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 批量处理全部图像
  python image_qc_preprocessing.py --input_dir ./raw_images --output_dir ./processed
  
  # 仅处理MCF-7细胞系
  python image_qc_preprocessing.py --input_dir ./raw_images --output_dir ./processed --cell_line MCF7
  
  # 调整质控阈值
  python image_qc_preprocessing.py --input_dir ./raw --output_dir ./out --laplacian_threshold 60
  
  # 人工标记不合格图像
  python image_qc_preprocessing.py --manual_reject --image MCF7_001_24h_F01.tif \
      --reason "划痕干扰" --cell_line MCF7 --sample_id MCF7_001 --timepoint 24h --view_id F01
        """
    )
    
    parser.add_argument('--input_dir', type=str, help='原始图像目录')
    parser.add_argument('--output_dir', type=str, help='输出目录')
    parser.add_argument('--cell_line', type=str, choices=['MCF7', 'MDA231'], 
                       help='指定处理的细胞系')
    
    parser.add_argument('--laplacian_threshold', type=float, default=50.0,
                       help='拉普拉斯方差阈值 (默认: 50)')
    parser.add_argument('--brightness_min', type=float, default=50.0,
                       help='亮度均值下限 (默认: 50)')
    parser.add_argument('--brightness_max', type=float, default=200.0,
                       help='亮度均值上限 (默认: 200)')
    
    parser.add_argument('--manual_reject', action='store_true',
                       help='启用手动标记不合格图像模式')
    parser.add_argument('--image', type=str, help='图像文件名')
    parser.add_argument('--reason', type=str, 
                       choices=ManualQCLogger.REJECT_REASONS,
                       help='不合格原因')
    parser.add_argument('--sample_id', type=str, help='样本ID (如: MCF7_001)')
    parser.add_argument('--timepoint', type=str, help='时间点 (如: 24h)')
    parser.add_argument('--view_id', type=str, help='视野ID (如: F01)')
    parser.add_argument('--operator', type=str, default='operator',
                       help='操作员姓名')
    parser.add_argument('--notes', type=str, default='',
                       help='详细备注')
    parser.add_argument('--severity', type=str, default='high',
                       choices=['high', 'medium', 'low'],
                       help='严重程度')
    
    args = parser.parse_args()
    
    config = QCConfig(
        LAPLACIAN_VAR_THRESHOLD=args.laplacian_threshold,
        BRIGHTNESS_MEAN_MIN=args.brightness_min,
        BRIGHTNESS_MEAN_MAX=args.brightness_max
    )
    
    if args.manual_reject:
        if not all([args.image, args.reason, args.cell_line, args.sample_id, args.timepoint, args.view_id]):
            parser.error("人工标记模式需要提供: --image, --reason, --cell_line, --sample_id, --timepoint, --view_id")
        
        logger = ManualQCLogger(args.output_dir or './qc_logs')
        logger.log_rejection(
            image_path=args.image,
            cell_line=args.cell_line,
            sample_id=args.sample_id,
            timepoint=args.timepoint,
            view_id=args.view_id,
            reject_reason=args.reason,
            severity=args.severity,
            operator=args.operator,
            notes=args.notes
        )
        return
    
    if not args.input_dir or not args.output_dir:
        parser.error("批量处理模式需要提供 --input_dir 和 --output_dir")
    
    processor = BatchProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config=config
    )
    
    processor.run_batch(cell_line=args.cell_line)


if __name__ == '__main__':
    main()
