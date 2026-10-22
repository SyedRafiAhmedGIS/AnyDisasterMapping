"""Locust outbreak habitat dataset (extended domain: locusts).

Optical (Sentinel-2) locust-habitat / affected-cropland mapping. The FAO
DLIS locust surveys provide presence/absence point records rather than
pixel masks; this adapter therefore expects a *derived* raster label:
habitat suitability / observed affected cropland polygons rasterised from
FAO records + land-cover (see scripts/data_prep/locust/README.md).

Dual-mode adapter (like CAUFloodDataset):

* ``task="seg"`` — habitat / affected-cropland extent at T2 (image stacks
  pre+post channels).
* ``task="cd"``  — binary change of affected cropland between pre/post
  acquisitions (new infestation).

Directory layout expected under *dataset_path*::

    {dataset_path}/PRE/{id}.tif      -- pre-date image (4 ch: B2,B3,B4,B8)
    {dataset_path}/POST/{id}.tif     -- post-date image (4 ch)
    {dataset_path}/GT/{id}.tif       -- seg: 0=background, 1=habitat/affected
                                            cd: 0=no-change, 1=new affected

Split files (one ID per line): train.txt / val.txt / test.txt.

Returns
-------
Seg mode -- 3-tuple : (image, label, sample_id)
    - image    : float32 (8, H, W)  -- pre+post stacked
    - label    : int64   (H, W)     -- 2-class mask, 255 = ignore
    - sample_id: str

CD mode -- 4-tuple : (pre, post, label, sample_id)
    - pre, post: float32 (4, H, W)
    - label    : int64   (H, W)     -- binary change, 255 = ignore
    - sample_id: str
"""

import os

import numpy as np

from ._disaster_common import (
    DisasterDatasetBase,
    ensure_3d,
    find_with_ext,
    match_channels,
    read_image_any,
)


class LocustHabitatDataset(DisasterDatasetBase):
    """Locust habitat / affected-cropland dataset (seg + cd modes)."""

    NUM_SEG_CLASSES = 2   # 0=background, 1=habitat / affected cropland
    NUM_CD_CLASSES = 2    # 0=no-change, 1=new affected cropland
    IN_CHANNELS_PER_DATE = 4

    def __init__(self, dataset_path: str, data_list_path: str,
                 crop_size=None, split: str = "train", transforms=None,
                 task: str = "seg") -> None:
        self.task = task.lower()
        super().__init__(dataset_path, data_list_path, crop_size, split,
                         transforms)

    def _sample_exists(self, sample_id: str) -> bool:
        name = str(sample_id)
        return all([
            find_with_ext(os.path.join(self.dataset_path, "PRE"), name) is not None,
            find_with_ext(os.path.join(self.dataset_path, "POST"), name) is not None,
            find_with_ext(os.path.join(self.dataset_path, "GT"), name) is not None,
        ])

    def __getitem__(self, index: int):
        sample_id = self.data_list[index]
        name = str(sample_id)

        pre_path = find_with_ext(os.path.join(self.dataset_path, "PRE"), name)
        post_path = find_with_ext(os.path.join(self.dataset_path, "POST"), name)
        gt_path = find_with_ext(os.path.join(self.dataset_path, "GT"), name)

        pre, invalid_pre = read_image_any(pre_path, return_invalid=True)
        post, invalid_post = read_image_any(post_path, return_invalid=True)
        mask = read_image_any(gt_path)[0].squeeze()

        invalid_mask = np.zeros(pre.shape[:2], dtype=bool)
        if invalid_pre is not None:
            invalid_mask |= invalid_pre
        if invalid_post is not None:
            invalid_mask |= invalid_post
        if np.any(invalid_mask):
            mask = mask.copy()
            mask[invalid_mask] = 255

        pre, post = match_channels(ensure_3d(pre), ensure_3d(post))
        mask = mask.astype(np.int64)

        if self.task == "seg":
            image = np.concatenate((pre, post), axis=-1)
            image, mask = self._apply_seg_transforms(image, mask)
            return image, mask, sample_id

        pre, post, mask = self._apply_cd_transforms(pre, post, mask)
        return pre, post, mask, sample_id
