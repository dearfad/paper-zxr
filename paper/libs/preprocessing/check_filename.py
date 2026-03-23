"""文件名命名规范验证模块"""

import re
from typing import Dict, Any

# 默认文件名模板
FILENAME_PATTERN = "{cell}_{id}_{time}_{passage}_{well}_{location}_{magnification}.tif"

def check_filename(filename: str, pattern: str = FILENAME_PATTERN) -> Dict[str, Any]:
    """检查单个文件名是否符合命名规范"""
    regex = pattern
    regex = regex.replace("{cell}", r"(?P<cell>MCF7|MB231)")
    regex = regex.replace("{id}", r"(?P<id>\d+)")
    regex = regex.replace("{time}", r"(?P<time>0|6|24|48)h")
    regex = regex.replace("{passage}", r"(?P<passage>P0|P1|P2)")
    regex = regex.replace("{well}", r"(?P<well>W1)")
    regex = regex.replace("{location}", r"(?P<location>CT|LL|LR|UL|UR)")
    regex = regex.replace("{magnification}", r"(?P<magnification>10x|20x|40x)")

    match = re.match(f"^{regex}$", filename)
    
    if match:
        return {
            "is_valid": True,
            **match.groupdict()
        }
    return {"is_valid": False}