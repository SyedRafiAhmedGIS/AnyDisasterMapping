"""Glacial-lake outburst flood (GLOF) dataset (extended domain: GLOF).

Optical (Sentinel-2 / Landsat) glacial-lake monitoring in the style of the
ICIMOD glacial-lake inventories and GAMDAM-CKD style multi-date outlines.
Dual-mode adapter:

* ``task="seg"`` — single-date lake-extent segmentation. The image stacks
  T1 + T2 channels; the label is the T2 water extent.
* ``task="scd"`` — semantic change detection: binary change (new/expanded
  water) plus per-date water/land semantic labels. This is the FIRST
  ``task: scd`` configuration shipped in the repository (the runtime
  supported scd via ``TaskSCD`` + ``changemamba_scd`` but no config existed).

Directory layout expected under *dataset_path*::

    {dataset_path}/T1/{id}.tif       -- pre-date image   (4 ch: B2,B3,B4,B8)
    {dataset_path}/T2/{id}.tif       -- post-date image  (4 ch)
    {dataset_path}/GT_CD/{id}.tif    -- 0=no-change, 1=new/expanded water
    {dataset_path}/GT_T1/{id}.tif    -- T1 semantic {0=land, 1=water, 2=snow/ice}
    {dataset_path}/GT_T2/{id}.tif    -- T2 semantic {0=land, 1=water, 2=snow/ice}

Split files (one ID per line): train.txt / val.txt / test.txt.

Returns
-------
Seg mode -- 3-tuple : (image, label, sample_id)
    - image    : float32 (8, H, W)  -- T1+T2 stacked
    - label    : int64   (H, W)     -- T2 water extent {0,1}, 255 = ignore
    - sample_id: str

SCD mode -- 6-tuple : (pre, post, cd_label, t1_label, t2_label, sample_id)
    - pre, post: float32 (4, H, W)
    - cd_label : int64   (H, W)     -- binary change {0,1}
    - t1/t2    : int64   (H, W)     -- semantic {0..1}; note TaskSCD masks
                                            label 0 as ignore for the
                                            semantic heads (see src/tasks/scd.py)
    - sample_id: str
"""

import os

import numpy as np

from ._disaster_common import (
    DisasterDatasetBase,
    find_with_ext,
    read_image_any,
    to_chw,
)


class GLOFICIMODDataset(DisasterDatasetBase):
    """Glacial-lake extent / expansion dataset (seg + scd modes)."""

    NUM_SEG_CLASSES = 2   # seg label: 0=land, 1=water (lake extent)
    NUM_SEM_CLASSES = 3  # scd semantic: 0=land, 1=water, 2=snow/ice
    NUM_CD_CLASSES = 2    # 0=no-change, 1=new/expanded water
    IN_CHANNELS_PER_DATE = 4

    def __init__(self, dataset_path: str, data_list_path: str,
                 crop_size=None, split: str = "train", transforms=None,
                 task: str = "seg") -> None:
        if task.lower() not in ("seg", "scd"):
            raise ValueError(f"GLOFICIMODDataset supports 'seg' or 'scd', got {task!r}")
        self.task = task.lower()
        super().__init__(dataset_path, data_list_path, crop_size, split,
                         transforms)

    def _sample_exists(self, sample_id: str) -> bool:
        name = str(sample_id)
        subdirs = ("T1", "T2", "GT_CD", "GT_T1", "GT_T2")
        return all(
            find_with_ext(os.path.join(self.dataset_path, sd), name) is not None
            for sd in subdirs
        )

    def _load_date(self, subdir: str, name: str):
        path = find_with_ext(os.path.join(self.dataset_path, subdir), name)
        img, invalid = read_image_any(path, return_invalid=True)
        return img, invalid

    def __getitem__(self, index: int):
        sample_id = self.data_list[index]
        name = str(sample_id)

        t1, invalid_t1 = self._load_date("T1", name)
        t2, invalid_t2 = self._load_date("T2", name)
        cd_label = read_image_any(
            find_with_ext(os.path.join(self.dataset_path, "GT_CD"), name)
        )[0].squeeze()
        t1_label = read_image_any(
            find_with_ext(os.path.join(self.dataset_path, "GT_T1"), name)
        )[0].squeeze()
        t2_label = read_image_any(
            find_with_ext(os.path.join(self.dataset_path, "GT_T2"), name)
        )[0].squeeze()

        invalid = np.zeros(t1.shape[:2], dtype=bool)
        if invalid_t1 is not None:
            invalid |= invalid_t1 if invalid_t1.ndim == 2 else np.any(invalid_t1, axis=-1)
        if invalid_t2 is not None:
            invalid |= invalid_t2 if invalid_t2.ndim == 2 else np.any(invalid_t2, axis=-1)
        if np.any(invalid):
            cd_label = cd_label.copy(); cd_label[invalid] = 255
            t1_label = t1_label.copy(); t1_label[invalid] = 255
            t2_label = t2_label.copy(); t2_label[invalid] = 255

        cd_label = cd_label.astype(np.int64)
        t1_label = t1_label.astype(np.int64)
        t2_label = t2_label.astype(np.int64)

        if self.task == "seg":
            image = np.concatenate((t1, t2), axis=-1)
            # seg label = binary lake extent from the T2 semantic map
            # ({0=land, 1=water, 2=snow/ice} -> {0, 1})
            mask = (t2_label == 1).astype(np.int64)
            mask[t2_label == 255] = 255
            if self.transforms is not None:
                blob = self.transforms(image=image, mask=mask)
                image, mask = blob["image"], blob["mask"]
            from ._disaster_common import ensure_3d
            image = to_chw(ensure_3d(image))
            return image, np.asarray(mask, dtype=np.int64), sample_id

        # SCD mode: dual-temporal inputs with synced augmentation
        if self.transforms is not None:
            blob = self.transforms(image=t1, image_post=t2, mask=cd_label,
                                   mask_t1=t1_label, mask_cd=cd_label)
            t1 = blob["image"]; t2 = blob["image_post"]
            cd_label = blob["mask"]
            t1_label = blob.get("mask_t1", t1_label)
            # NOTE: mask_cd aliases cd_label inside albumentations; keep both
        pre, post = to_chw(t1), to_chw(t2)
        return (pre, post,
                np.asarray(cd_label, dtype=np.int64),
                np.asarray(t1_label, dtype=np.int64),
                np.asarray(t2_label, dtype=np.int64),
                sample_id)
