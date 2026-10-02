"""Path validation shared by the tools."""
from pathlib import Path


def input_path(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe input path: {relative}")
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"Input escapes data root: {relative}")
    return result


def output_path(workspace, path):
    workspace = Path(workspace).resolve()
    path = Path(path).resolve()
    if not path.is_relative_to(workspace / "outputs"):
        raise ValueError("Output must stay inside outputs/")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    return path
