我将为你编写完整的图像质控与预处理Python脚本，包含人工初筛记录功能和自动质控算法。
现在我来创建一个配套的快速使用脚本和示例配置文件：  
 最后创建一个README文档说明使用方法：  
 已为你完成完整的图像质控与预处理Python脚本系统。以下是交付内容和使用指南：

---

## 📦 交付文件包（位于 `/mnt/kimi/output/`）

| 文件                        | 功能                         | 大小   |
| --------------------------- | ---------------------------- | ------ |
| `image_qc_preprocessing.py` | **主程序** - 包含全部功能类  | 25.8KB |
| `run_qc_pipeline.py`        | 快速启动脚本（简化命令行）   | 1.5KB  |
| `config.yaml`               | 配置文件（含细胞系特定参数） | 2.1KB  |
| `example_usage.py`          | 使用示例代码                 | 3.2KB  |
| `README.md`                 | 完整使用文档                 | 4.8KB  |

---

## 🎯 核心功能实现

### 1. 人工初筛记录 (`ManualQCLogger`)

**功能**：

- 标准化记录不合格图像及原因（图像模糊、曝光异常、划痕、气泡等9类原因）
- 支持严重程度分级（high/medium/low）
- 自动导出Excel报告和JSON备注
- 计算Kappa一致性系数

**使用方式**：

```bash
python image_qc_preprocessing.py --manual_reject \
    --image "MCF7_001_24h_F01.tif" \
    --cell_line "MCF7" \
    --sample_id "MCF7_001" \
    --timepoint "24h" \
    --view_id "F01" \
    --reason "划痕干扰" \
    --severity "high" \
    --operator "张实验员" \
    --notes "图像中央有水平亮线"
```

### 2. 自动质控 (`AutoImageQC`)

**算法实现**：

- **清晰度**：拉普拉斯方差计算 `cv2.Laplacian(gray, cv2.CV_64F).var()`
- **亮度统计**：均值 `[50, 200]`，标准差 `> 20`
- **划痕检测**：Canny边缘检测 + 霍夫直线检测（>5条直线为异常）
- **气泡检测**：二值化阈值200 + 圆形度分析（>0.7为气泡）

**细胞系差异化标准**：
| 指标 | MCF-7 | MDA-MB-231 | 原因 |
|------|-------|------------|------|
| 清晰度阈值 | > 55 | > 45 | MDA-MB-231边缘不规则 |
| 亮度范围 | [50, 200] | [40, 220] | 侵袭性结构透光性差异 |

### 3. 预处理流程 (`ImagePreprocessor`)

**处理步骤**：

1. **中心裁剪**（0.8比例）- 聚焦培养核心区
2. **尺寸统一**（336×336px）- 适配CLIP/ViT输入
3. **CLAHE增强** - `clipLimit=2.0, tileGridSize=(8,8)`
4. **去噪** - 高斯模糊(σ=0.5) + 中值滤波(3×3)

### 4. 批量处理工作流 (`BatchProcessor`)

**自动分类输出**：

```
output/
├── 01_passed/           # 通过质控的原始图像（按细胞系分）
├── 02_rejected_manual/  # 人工标记不合格
├── 03_rejected_auto/    # 自动质控不合格
└── 04_preprocessed/     # 预处理后图像（336x336）
```

---

## 🚀 快速使用指南

### 环境准备

```bash
pip install opencv-python numpy pandas tqdm openpyxl
```

### 批量处理（最常用）

```bash
# 处理全部图像
python run_qc_pipeline.py ./raw_images ./output

# 仅处理MCF-7（高清晰度标准）
python run_qc_pipeline.py ./raw_images ./output --cell_line MCF7

# 处理MDA-MB-231（降低阈值适应不规则形态）
python run_qc_pipeline.py ./raw_images ./output --cell_line MDA231 --threshold 45
```

### Python代码调用

```python
from image_qc_preprocessing import BatchProcessor, QCConfig

# 自定义配置（针对MDA-MB-231）
config = QCConfig(
    LAPLACIAN_VAR_THRESHOLD=45.0,  # 降低清晰度要求
    BRIGHTNESS_MEAN_MIN=40.0,
    BRIGHTNESS_MEAN_MAX=220.0
)

processor = BatchProcessor(
    input_dir="./raw_images",
    output_dir="./processed",
    config=config
)

# 运行批量处理
processor.run_batch(cell_line="MDA231")
```

---

## ⚠️ 关键设计要点

1. **双细胞系适配**：MDA-MB-231因侵袭性生长边缘不规则，自动降低清晰度阈值至45（MCF-7为55）

2. **时序对齐**：预处理包含刚性对齐功能，确保同一孔不同时间点视野一致

3. **命名解析**：自动解析 `MCF7_001_24h_F01.tif` → cell_line/sample_id/timepoint/view_id

4. **质控日志**：所有操作记录到CSV，支持后续数据清洗报告和论文方法学描述

5. **无损格式**：强制使用TIFF格式，避免JPG压缩伪影

这套脚本可直接用于项目第二阶段（数据基建），处理0h-72h的关键预测时间窗图像。需要我解释任何具体算法的实现细节吗？
