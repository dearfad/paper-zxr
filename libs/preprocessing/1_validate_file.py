import re
from pathlib import Path
from PIL import Image

RAW_IMAGE_DIR = Path(".") / "data" / "raw"
PATTERN = "{cell}_{id}_{time}_{location}_{magnification}.tif"


def validate_filename(filename: str, pattern: str) -> bool:
    """检验文件名是否符合预设模板"""
    regex_pattern = pattern.replace("{cell}", r"(MCF7|MB231)")
    regex_pattern = regex_pattern.replace("{id}", r"\d+")
    regex_pattern = regex_pattern.replace("{time}", r"(0|6|24|48)h")
    regex_pattern = regex_pattern.replace("{location}", r"(CT|LL|LR|UL|UR)")
    regex_pattern = regex_pattern.replace("{magnification}", r"(10x|20x|40x)")
    return bool(re.match(f"^{regex_pattern}$", filename))


def validate_directory_filenames(directory: Path, pattern: str) -> dict:
    """批量检验目录下所有图像文件名是否符合命名规范

    遍历指定目录下的所有文件，根据预设的正则表达式模板逐个验证文件名是否合法。
    合法的文件名将被添加到 valid 列表，非法文件名将被添加到 invalid 列表并打印警告信息。

    Args:
        directory: 要检验的目录路径 (Path 对象)
        pattern: 文件名模板字符串，支持以下占位符:
            - {cell}: 细胞系类型，取值范围为 MCF7 或 MB231
            - {id}: 细胞样本编号，整数形式
            - {time}: 培养时间，取值范围为 0h、6h、24h 或 48h
            - {location}: 视野位置，取值范围为 CT、LL、LR、UL 或 UR
            - {magnification}: 显微镜放大倍数，取值范围为 10x、20x 或 40x

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
                print(f"文件名异常: {filename}")
    print(f"文件名正确: {len(valid)}")
    print(f"文件名异常: {len(invalid)}")
    return {"valid": valid, "invalid": invalid}


def test_read_images(directory: Path) -> dict:
    """测试读取目录下所有图像文件并输出基本信息

    遍历指定目录下的所有图像文件，尝试使用PIL读取每个文件，并输出图片的基本信息。
    正常读取的图片信息将添加到 readable 列表，读取失败的图片将添加到 unreadable 列表。

    Args:
        directory: 要测试的目录路径 (Path 对象)

    Returns:
        dict: 包含读取结果的字典
            - readable (list): 可以正常读取的图片信息列表，每个元素为包含以下键的字典:
                - filename: 文件名
                - format: 图片格式
                - mode: 颜色模式
                - size: 图片尺寸 (宽, 高)
                - width: 图片宽度
                - height: 图片高度
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

    print(f"\n可正常读取: {len(readable)}")
    print(f"读取失败: {len(unreadable)}")
    return {"readable": readable, "unreadable": unreadable}


def check_missing_images(directory: Path) -> dict:
    """检查图片文件是否缺失，验证数据完整性

    检查规则：
    1. 每个细胞系+编号组合（如MCF7_0）必须有24h和48h两个时间点
    2. 在24h和48h两个时间点下必须有5个位置（CT, LL, LR, UL, UR）
    3. 在每个位置下必须有3个放大倍数（10x, 20x, 40x）

    Args:
        directory: 要检查的目录路径 (Path 对象)

    Returns:
        dict: 包含检查结果的字典
            - complete (list): 完整的图片组合列表
            - missing (list): 缺失的图片信息列表，每个元素为包含以下键的字典:
                - cell: 细胞系
                - id: 编号
                - time: 缺失的时间点
                - location: 缺失的位置（如果有）
                - magnification: 缺失的放大倍数（如果有）
    """
    required_times = ["24h", "48h"]
    required_locations = ["CT", "LL", "LR", "UL", "UR"]
    required_magnifications = ["10x", "20x", "40x"]

    files = [
        f.name
        for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() == ".tif"
    ]

    structure = {}
    for filename in files:
        match = re.match(
            r"^(MCF7|MB231)_(\d+)_(0|6|24|48)h_(CT|LL|LR|UL|UR)_(10x|20x|40x)\.tif$",
            filename,
        )
        if match:
            cell, id_num, time, location, magnification = match.groups()
            key = (cell, id_num)
            if key not in structure:
                structure[key] = {}
            if time not in structure[key]:
                structure[key][time] = {}
            if location not in structure[key][time]:
                structure[key][time][location] = set()
            structure[key][time][location].add(magnification)

    complete = []
    missing = []

    for (cell, id_num), time_data in sorted(structure.items()):
        sample_key = f"{cell}_{id_num}"

        for time in required_times:
            if time not in time_data:
                missing.append(
                    {
                        "cell": cell,
                        "id": id_num,
                        "time": time,
                        "location": None,
                        "magnification": None,
                        "reason": f"缺少时间点 {time}",
                    }
                )
                continue

            for location in required_locations:
                if location not in time_data[time]:
                    missing.append(
                        {
                            "cell": cell,
                            "id": id_num,
                            "time": time,
                            "location": location,
                            "magnification": None,
                            "reason": f"{sample_key}_{time} 缺少位置 {location}",
                        }
                    )
                    continue

                for mag in required_magnifications:
                    if mag not in time_data[time][location]:
                        missing.append(
                            {
                                "cell": cell,
                                "id": id_num,
                                "time": time,
                                "location": location,
                                "magnification": mag,
                                "reason": f"{sample_key}_{time}_{location} 缺少放大倍数 {mag}",
                            }
                        )

        has_24h = "24h" in time_data
        has_48h = "48h" in time_data
        all_complete = all(
            loc in time_data.get(time, {})
            and time_data[time][loc] == set(required_magnifications)
            for time in required_times
            for loc in required_locations
        )

        if all_complete:
            complete.append(sample_key)

    print(f"完整数据集: {len(complete)} 个")
    for item in complete:
        print(f"  ✓ {item}")

    print(f"\n缺失图片: {len(missing)} 个")
    for item in missing:
        print(f"  ✗ {item['reason']}")

    return {"complete": complete, "missing": missing}


def main():
    # 图片文件名校验
    validate_directory_filenames(RAW_IMAGE_DIR, PATTERN)
    print("\n" + "=" * 50 + "\n")
    # 图片文件缺失检测
    check_missing_images(RAW_IMAGE_DIR)
    print("\n" + "=" * 50 + "\n")
    # 图片文件读取校验
    test_read_images(RAW_IMAGE_DIR)


if __name__ == "__main__":
    main()
