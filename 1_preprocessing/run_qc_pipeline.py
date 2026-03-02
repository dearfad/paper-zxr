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
