"""
URDF 路径辅助工具

用于修复 URDF 文件中的相对 mesh 路径问题
"""
import os
import tempfile
from pathlib import Path
from typing import Optional


def prepare_urdf_path(urdf_path: str) -> str:
    """
    准备 URDF 文件路径，将相对 mesh 路径替换为绝对路径。
    
    Args:
        urdf_path: URDF 文件路径（可以是相对或绝对路径）
        
    Returns:
        处理后的 URDF 文件路径（如果是临时文件，返回绝对路径；否则返回原始路径）
    """
    urdf_file = Path(urdf_path).resolve()
    
    if not urdf_file.exists():
        return str(urdf_file)
    
    try:
        content = urdf_file.read_text(encoding="utf-8")
    except Exception:
        return str(urdf_file)
    
    # 检查是否包含相对路径的 mesh 引用
    marker = "../meshes/"
    if marker not in content:
        return str(urdf_file)
    
    # 计算 meshes 目录的绝对路径
    # URDF 文件在 resources/urdf/ 目录下
    # meshes 目录应该在 resources/meshes/ 目录下
    urdf_dir = urdf_file.parent
    resources_dir = urdf_dir.parent  # resources 目录
    meshes_dir = resources_dir / "meshes"
    
    if not meshes_dir.exists():
        # 如果 meshes 目录不存在，尝试其他可能的路径
        # 可能 URDF 路径已经是绝对路径，尝试从 URDF 路径推断
        return str(urdf_file)
    
    # 将相对路径替换为绝对路径
    meshes_absolute = meshes_dir.resolve().as_posix()
    patched_content = content.replace(marker, f"{meshes_absolute}/")
    
    # 创建临时文件
    try:
        fd, tmp_path = tempfile.mkstemp(prefix="urdf_", suffix=".urdf")
        os.close(fd)
        tmp_file = Path(tmp_path)
        tmp_file.write_text(patched_content, encoding="utf-8")
        return str(tmp_file)
    except Exception:
        # 如果创建临时文件失败，返回原始路径
        return str(urdf_file)

