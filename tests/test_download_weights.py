import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import download_weights as dw  # noqa: E402

FILES = ["unet.pth", "seg/unet.pth", "cd/unet.pth", "cd/bit.pth", "README.md"]
WEIGHTS = [f for f in FILES if f.endswith(dw.WEIGHT_SUFFIXES)]


def test_flat_file_preferred_without_task():
    assert dw._select(WEIGHTS, "unet", None, False) == ["unet.pth"]


def test_task_prefix_selected():
    assert dw._select(WEIGHTS, "unet", "cd", False) == ["cd/unet.pth"]


def test_all_with_task_filters_folder():
    assert dw._select(WEIGHTS, "x", "cd", True) == ["cd/unet.pth", "cd/bit.pth"]


def test_all_keeps_distinct_same_named_files():
    chosen = dw._select(WEIGHTS, "x", None, True)
    assert len(chosen) == len(set(chosen)) == 4  # seg/unet and cd/unet both kept


def test_parse_sums():
    digest = "ab" * 32
    assert dw._parse_sums(f"{digest}  cd/bit.pth\n# comment\n") == {"cd/bit.pth": digest}
