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
    
    logger = ManualQCLogger(output_dir="./qc_output")
    
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
    
    logger.mark_as_passed(
        image_path="./raw/MCF7_002_12h_F02.tif",
        cell_line="MCF7",
        sample_id="MCF7_002",
        timepoint="12h",
        view_id="F02",
        operator="张实验员"
    )
    
    logger.export_rejection_report()
    print("✅ 人工初筛记录完成\n")


def example_2_auto_qc():
    """示例2: 自动质控评估单张图像"""
    print("="*60)
    print("示例2: 自动质控评估")
    print("="*60)
    
    qc = AutoImageQC()
    
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
    
    output = preprocessor.preprocess(
        image_path="./raw/MCF7_001_24h_F01.tif",
        output_path="./processed/MCF7_001_24h_F01_preprocessed.tif"
    )
    
    print(f"✅ 预处理完成，输出尺寸: {output.shape}")
    print(f"   处理流程: 中心裁剪(0.8) ->  resize(336x336) -> CLAHE -> 去噪\n")


def example_4_batch_process():
    """示例4: 批量处理"""
    print("="*60)
    print("示例4: 批量处理工作流")
    print("="*60)
    
    config = QCConfig(
        LAPLACIAN_VAR_THRESHOLD=45.0,
        BRIGHTNESS_MEAN_MIN=40.0,
        BRIGHTNESS_MEAN_MAX=220.0
    )
    
    processor = BatchProcessor(
        input_dir="./raw_images",
        output_dir="./processed_output",
        config=config
    )
    
    print("开始批量处理 MDA-MB-231 细胞系...")
    results = processor.run_batch(cell_line="MDA231")
    
    print(f"\n处理完成，共处理 {len(results)} 张图像")


if __name__ == '__main__':
    print("\n🧬 MCF-7 / MDA-MB-231 图像质控系统使用示例\n")
    
    print("提示: 请根据实际路径修改示例中的文件路径后运行")
    print("命令行快速使用:")
    print("  python run_qc_pipeline.py ./raw_images ./output --cell_line MCF7")
