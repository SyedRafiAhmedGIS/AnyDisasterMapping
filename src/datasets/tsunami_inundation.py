"""Tsunami inundation dataset (extended domain: tsunamis).

Change-detection adapter for tsunami inundation extent derived from pre/post
event satellite pairs (xBD tsunami events, MMFlood-style SAR inundation).
Reuses the flood-style cd contract: binary inundation change between two
acquisitions.

Directory layout expected under *dataset_path*::

    {dataset_path}/PRE/{id}.tif      -- pre-event image (3 ch RGB / SAR GCPS)
    {dataset_path}/POST/{id}.tif     -- post-event image (3 ch)
    {dataset_path}/GT/{id}.tif       -- 0=no-change, 1=inundated

Split files (one ID per line): train.txt / val.txt / test.txt.

Returns
-------
4-tuple : (pre_img, post_img, label, sample_id)
    - pre_img  : float32 (3, H, W)
    - post_img : float32 (3, H, W)
    - label    : int64   (H, W)  -- binary inundation change, 255 = ignore
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


class TsunamiInundationDataset(DisasterDatasetBase):
    """Tsunami inundation change-detection dataset (xView2/xBD-style pairs)."""

    NUM_CLASSES = 2       # 0=no-change, 1=inundated
    IN_CHANNELS_PER_DATE = 3

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

        pre, post, mask = self._apply_cd_transforms(pre, post, mask)
        return pre, post, mask, sample_id
