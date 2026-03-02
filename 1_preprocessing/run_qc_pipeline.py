
# 创建快速启动脚本和配置文件

# 1. 快速启动脚本 (run_qc_pipeline.py)
quick_start = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速启动脚本 - MCF-7/MDA-MB-231图像质控流水线
简化版命令行接口，适合日常使用
"""

import sys
import os

# 将当前目录加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from image_qc_preprocessing import BatchProcessor, QCConfig, ManualQCLogger

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='快速启动质控流水线')
    parser.add_argument('input_dir', help='原始图像目录')
    parser.add_argument('output_dir', help='输出目录')
    parser.add_argument('--cell_line', '-c', choices=['MCF7', 'MDA231'], 
                       help='仅处理特定细胞系')
    parser.add_argument('--threshold', '-t', type=float, default=50,
                       help='清晰度阈值 (默认50)')
    
    args = parser.parse_args()
    
    # 创建配置
    config = QCConfig(LAPLACIAN_VAR_THRESHOLD=args.threshold)
    
    # 运行处理
    print("\\n" + "="*60)
    print("🧬 MCF-7 / MDA-MB-231 类器官图像质控系统")
    print("="*60 + "\\n")
    
    processor = BatchProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config=config
    )
    
    results = processor.run_batch(cell_line=args.cell_line)
    
    print("\\n✅ 处理完成！")
    print(f"📊 详细报告请查看: {args.output_dir}/reports/")
    print(f"📋 质控日志请查看: {args.output_dir}/manual_qc_log.csv\\n")

if __name__ == '__main__':
    main()
'''

# 2. 配置文件 (config.yaml)
config_yaml = '''# MCF-7 / MDA-MB-231 3D类器官图像质控配置

# ==================== 质控参数 ====================
quality_control:
  # 清晰度阈值 (拉普拉斯方差)
  # MCF-7建议: 50-60 (结构紧密，边缘清晰)
  # MDA-MB-231建议: 40-50 (边缘不规则，允许稍低清晰度)
  laplacian_variance_threshold: 50.0
  
  # 亮度阈值
  brightness:
    mean_min: 50.0    # 避免曝光不足
    mean_max: 200.0   # 避免曝光过度
    std_min: 20.0     # 确保足够对比度
  
  # 划痕检测
  scratch_detection:
    enabled: true
    line_threshold: 100
    max_allowed_lines: 5  # 超过此数量判定为不合格

# ==================== 预处理参数 ====================
preprocessing:
  # 目标尺寸 (适配CLIP/ViT模型)
  target_size: [336, 336]
  
  # 中心裁剪比例 (聚焦培养核心区，排除边缘)
  center_crop_ratio: 0.8
  
  # CLAHE自适应直方图均衡化
  clahe:
    clip_limit: 2.0
    tile_grid_size: [8, 8]
  
  # 去噪
  denoising:
    gaussian_sigma: 0.5
    median_kernel_size: 3

# ==================== 细胞系特定参数 ====================
cell_line_specific:
  MCF7:
    # MCF-7: 紧密球状结构，要求高圆度
    laplacian_threshold: 55.0
    min_circularity: 0.8
    expected_diameter_range: [50, 400]  # μm
    
  MDA231:
    # MDA-MB-231: 不规则侵袭性结构，允许较低圆度
    laplacian_threshold: 45.0
    min_circularity: 0.4
    expected_diameter_range: [30, 500]  # μm (变异大)

# ==================== 采集参数 ====================
acquisition:
  # 时间点设置 (小时)
  timepoints: [0, 6, 12, 24, 48, 72, 96, 168, 336]
  
  # 每孔视野数
  views_per_well: 5  # 中心 + 四角
  
  # 图像格式
  format: "TIFF"
  
  # 命名规范
  naming_convention: "{CellLine}_{SampleID}_{Timepoint}_{ViewID}.tif"

# ==================== 数据集划分 ====================
dataset_split:
  train_ratio: 0.6
  val_ratio: 0.2
  test_ratio: 0.2
  stratify_by: ["cell_line", "outcome"]  # 按细胞系和结果分层
'''

# 3. 使用示例脚本 (example_usage.py)
example_usage = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用示例: 如何调用质控系统进行人工初筛和批量处理
"""

import sys
sys.path.insert(0, '.')

from image_qc_preprocessing import (
    ManualQCLogger, 
    AutoImageQC, 
    ImagePreprocessor,
    BatchProcessor,
    QCConfig
)

def example_1_manual_qc():
    """示例1: 人工初筛记录"""
    print("="*60)
    print("示例1: 人工初筛记录")
    print("="*60)
    
    # 初始化记录器
    logger = ManualQCLogger(output_dir="./qc_output")
    
    # 场景A: 发现划痕干扰
    logger.log_rejection(
        image_path="./raw/MCF7_001_24h_F01.tif",
        cell_line="MCF7",
        sample_id="MCF7_001",
        timepoint="24h",
        view_id="F01",
        reject_reason="划痕干扰",
        severity="high",
        operator="张实验员",
        notes="图像中央有明显水平亮线，疑似载玻片划痕，影响边缘检测"
    )
    
    # 场景B: 曝光过度
    logger.log_rejection(
        image_path="./raw/MDA231_005_48h_F03.tif",
        cell_line="MDA231",
        sample_id="MDA231_005",
        timepoint="48h",
        view_id="F03",
        reject_reason="曝光过度",
        severity="medium",
        operator="张实验员",
        notes="中心区域过曝，细胞细节丢失"
    )
    
    # 场景C: 标记通过
    logger.mark_as_passed(
        image_path="./raw/MCF7_002_12h_F02.tif",
        cell_line="MCF7",
        sample_id="MCF7_002",
        timepoint="12h",
        view_id="F02",
        operator="张实验员"
    )
    
    # 导出报告
    logger.export_rejection_report()
    print("✅ 人工初筛记录完成\\n")


def example_2_auto_qc():
    """示例2: 自动质控评估单张图像"""
    print("="*60)
    print("示例2: 自动质控评估")
    print("="*60)
    
    qc = AutoImageQC()
    
    # 评估图像 (假设路径)
    result = qc.evaluate_image("./raw/MCF7_001_24h_F01.tif")
    
    print(f"评估结果:")
    print(f"  是否通过: {result['passed']}")
    print(f"  拉普拉斯方差: {result['laplacian_var']:.2f} (阈值: 50)")
    print(f"  亮度均值: {result['brightness_mean']:.2f}")
    print(f"  亮度标准差: {result['brightness_std']:.2f}")
    print(f"  划痕检测: {result['has_scratch']} (数量: {result.get('scratch_count', 0)})")
    print(f"  气泡检测: {result['has_bubble']} (数量: {result.get('bubble_count', 0)})")
    
    if not result['passed']:
        print(f"  ❌ 不合格原因: {result['fail_reasons']}")
    else:
        print(f"  ✅ 图像质量合格")
    print()


def example_3_preprocess():
    """示例3: 单张图像预处理"""
    print("="*60)
    print("示例3: 图像预处理")
    print("="*60)
    
    preprocessor = ImagePreprocessor()
    
    # 预处理单张图像
    output = preprocessor.preprocess(
        image_path="./raw/MCF7_001_24h_F01.tif",
        output_path="./processed/MCF7_001_24h_F01_preprocessed.tif"
    )
    
    print(f"✅ 预处理完成，输出尺寸: {output.shape}")
    print(f"   处理流程: 中心裁剪(0.8) ->  resize(336x336) -> CLAHE -> 去噪\\n")


def example_4_batch_process():
    """示例4: 批量处理"""
    print("="*60)
    print("示例4: 批量处理工作流")
    print("="*60)
    
    # 自定义配置 (针对MDA-MB-231调整阈值)
    config = QCConfig(
        LAPLACIAN_VAR_THRESHOLD=45.0,  # MDA-MB-231边缘不规则，降低阈值
        BRIGHTNESS_MEAN_MIN=40.0,
        BRIGHTNESS_MEAN_MAX=220.0
    )
    
    processor = BatchProcessor(
        input_dir="./raw_images",
        output_dir="./processed_output",
        config=config
    )
    
    # 仅处理MDA-MB-231
    print("开始批量处理 MDA-MB-231 细胞系...")
    results = processor.run_batch(cell_line="MDA231")
    
    print(f"\\n处理完成，共处理 {len(results)} 张图像")


if __name__ == '__main__':
    print("\\n🧬 MCF-7 / MDA-MB-231 图像质控系统使用示例\\n")
    
    # 运行示例 (实际使用时取消注释)
    # example_1_manual_qc()
    # example_2_auto_qc()
    # example_3_preprocess()
    # example_4_batch_process()
    
    print("提示: 请根据实际路径修改示例中的文件路径后运行")
    print("命令行快速使用:")
    print("  python run_qc_pipeline.py ./raw_images ./output --cell_line MCF7")
'''

# 保存所有文件
files_to_save = [
    ("/mnt/kimi/output/run_qc_pipeline.py", quick_start),
    ("/mnt/kimi/output/config.yaml", config_yaml),
    ("/mnt/kimi/output/example_usage.py", example_usage)
]

for path, content in files_to_save:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ 已创建: {path}")

print("\n" + "="*60)
print("📦 文件清单:")
print("="*60)
print("1. image_qc_preprocessing.py  - 主程序 (完整功能)")
print("2. run_qc_pipeline.py         - 快速启动脚本")
print("3. config.yaml                - 配置文件")
print("4. example_usage.py           - 使用示例")
