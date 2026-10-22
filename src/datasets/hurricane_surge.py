"""Hurricane / cyclone storm-surge dataset (extended domain: hurricanes).

Change-detection adapter for storm-surge inundation from Sentinel-1 pre/post
pairs (Sen1Floods11-style VV/VH, flooded-area labels) plus xBD wind-damage
imagery when optical pairs are preferred. SAR-first: 2 channels (VV, VH).

Directory layout expected under *dataset_path*::

    {dataset_path}/PRE/{id}.tif      -- pre-event SAR (2 ch: VV, VH)
    {dataset_path}/POST/{id}.tif     -- post-event SAR (2 ch: VV, VH)
    {dataset_path}/GT/{id}.tif       -- 0=no-change, 1=surge inundation

Split files (one ID per line): train.txt / val.txt / test.txt.

Returns
-------
4-tuple : (pre_img, post_img, label, sample_id)
    - pre_img  : float32 (2, H, W)
    - post_img : float32 (2, H, W)
    - label    : int64   (H, W)  -- binary surge mask, 255 = ignore
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


class HurricaneSurgeDataset(DisasterDatasetBase):
    """Storm-surge inundation change detection (Sentinel-1 VV/VH)."""

    NUM_CLASSES = 2       # 0=no-change, 1=surge inundation
    IN_CHANNELS_PER_DATE = 2

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
