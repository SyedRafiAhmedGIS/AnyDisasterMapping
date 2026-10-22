"""Shared helpers for the extended disaster-domain dataset adapters.

These utilities centralise the loading behaviour that the original adapters
(e.g. ``flood_cau.py``) each re-implement inline: robust multi-format image
reading, channel layout fixes, and file-list handling. Only the *extended*
domain adapters import from here; the original adapters remain untouched.
"""

import os
from typing import List, Optional

import numpy as np
from torch.utils.data import Dataset

try:
    import tifffile
except ImportError:  # pragma: no cover
    tifffile = None

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None


IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


# ---------------------------------------------------------------------------
# Reading helpers
# ---------------------------------------------------------------------------

def read_image_any(path: str, return_invalid: bool = False):
    """Load an image (TIFF / PNG / JPG / HDF5) as float32 (H, W, C).

    Returns ``(array, invalid_mask_or_None)`` when *return_invalid* is True.
    NaN / Inf pixels are replaced by 0.0; the invalid mask marks them so
    adapters can map them to ``ignore_index = 255``.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        if tifffile is None:
            raise ImportError("tifffile is required to read TIFF images")
        array = np.asarray(tifffile.imread(path), dtype=np.float32)
    elif ext in (".h5", ".hdf5"):
        if h5py is None:
            raise ImportError("h5py is required to read HDF5 images")
        with h5py.File(path, "r") as f:
            key = next(iter(f.keys()))
            array = np.asarray(f[key], dtype=np.float32)
    else:
        import imageio
        array = np.asarray(imageio.imread(path), dtype=np.float32)

    invalid = None
    if return_invalid:
        invalid = ~np.isfinite(array)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    array = move_channels_last(array)
    if return_invalid and invalid is not None:
        invalid = move_channels_last(invalid.astype(bool))
        if invalid.ndim == 3:
            invalid = np.any(invalid, axis=-1)
    return array.astype(np.float32, copy=False), invalid


def move_channels_last(array: np.ndarray) -> np.ndarray:
    """Move channels-first to channels-last if the shape looks like (C, H, W)."""
    if (array.ndim == 3
            and array.shape[0] <= 16
            and array.shape[0] <= array.shape[1]
            and array.shape[0] <= array.shape[2]):
        return np.moveaxis(array, 0, -1)
    return array


def ensure_3d(array: np.ndarray) -> np.ndarray:
    """Ensure the array is at least 3-D (H, W, C)."""
    if array.ndim == 2:
        return array[:, :, None]
    return array


def tile_channels(array: np.ndarray, target_channels: int) -> np.ndarray:
    """Tile the last axis to reach *target_channels*, then truncate exactly."""
    channels = array.shape[-1]
    if channels == target_channels:
        return array
    repeats = (target_channels + channels - 1) // channels
    tiled = np.tile(array, (1, 1, repeats))
    return tiled[..., :target_channels]


def match_channels(*arrays: np.ndarray) -> list:
    """Tile all arrays to the maximum channel count among them."""
    target = max(a.shape[-1] for a in arrays)
    return [tile_channels(a, target) for a in arrays]


def to_chw(image: np.ndarray) -> np.ndarray:
    """(H, W, C) -> contiguous (C, H, W) float32."""
    image = ensure_3d(image)
    out = np.ascontiguousarray(image.transpose(2, 0, 1))
    return out.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# File-list helpers
# ---------------------------------------------------------------------------

def read_data_list(path: str) -> List[str]:
    """Read a split file: one sample ID per line, blank lines skipped."""
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def find_with_ext(directory: str, name: str) -> Optional[str]:
    """Return the first existing path for *name* with a known extension."""
    base = os.path.join(directory, name)
    if os.path.isfile(base):
        return base
    for ext in IMG_EXTENSIONS:
        candidate = base + ext
        if os.path.isfile(candidate):
            return candidate
    return None


# ---------------------------------------------------------------------------
# Base class for the extended adapters
# ---------------------------------------------------------------------------

class DisasterDatasetBase(Dataset):
    """Common scaffolding for the 9 extended disaster-domain datasets.

    Handles split-file reading, missing-sample filtering, and the
    ``sample_exists`` protocol used across the repository. Subclasses
    implement :meth:`_sample_paths` and :meth:`_load_sample`.
    """

    def __init__(self, dataset_path: str, data_list_path: str,
                 crop_size: Optional[int] = None, split: str = "train",
                 transforms=None) -> None:
        self.dataset_path = dataset_path
        self.split = split
        self.transforms = transforms
        self.crop_size = crop_size  # kept for interface compatibility

        raw_list = read_data_list(data_list_path)
        self.data_list = [d for d in raw_list if self._sample_exists(d)]
        if len(self.data_list) < len(raw_list):
            print(f"[{self.__class__.__name__}] Filtered "
                  f"{len(raw_list) - len(self.data_list)} missing samples "
                  f"(kept {len(self.data_list)}/{len(raw_list)})")
        if not self.data_list:
            raise RuntimeError(
                f"[{self.__class__.__name__}] No usable samples found. "
                f"dataset_path={dataset_path!r} data_list_path={data_list_path!r}. "
                f"Run scripts/data_prep/<domain>/make_dummy.py to generate "
                f"synthetic data, or check the preparation README.")

    def _sample_exists(self, sample_id: str) -> bool:  # pragma: no cover
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int):  # pragma: no cover
        raise NotImplementedError

    # -- shared augmentation application ---------------------------------

    def _apply_seg_transforms(self, image, mask):
        """Apply albumentations for seg mode, return channel-first arrays."""
        if self.transforms is not None:
            blob = self.transforms(image=image, mask=mask)
            image, mask = blob["image"], blob["mask"]
        return to_chw(image), np.asarray(mask, dtype=np.int64)

    def _apply_cd_transforms(self, pre, post, mask):
        """Apply albumentations for cd mode, return channel-first triple."""
        if self.transforms is not None:
            blob = self.transforms(image=pre, image_post=post, mask=mask)
            pre, post, mask = blob["image"], blob["image_post"], blob["mask"]
        return to_chw(pre), to_chw(post), np.asarray(mask, dtype=np.int64)
