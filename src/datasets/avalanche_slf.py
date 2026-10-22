"""Avalanche debris dataset (extended domain: avalanches).

Covers optical / SAR avalanche-outlined imagery in the style of the Swiss
(SLF / SwissTopo) open-data avalanche outlines and SPOT-6/7 or Sentinel
acquisitions. Single-date segmentation of avalanche debris outlines.

Directory layout expected under *dataset_path*::

    {dataset_path}/IMG/{id}.tif      -- image (4 ch: R, G, B, NIR or SAR-derived)
    {dataset_path}/GT/{id}.tif       -- 0=background, 1=avalanche debris

Split files (one ID per line): train.txt / val.txt / test.txt.

Returns
-------
3-tuple : (image, label, sample_id)
    - image    : float32 (4, H, W)
    - label    : int64   (H, W)  -- 2-class debris mask, 255 = ignore
    - sample_id: str
"""

import os

import numpy as np

from ._disaster_common import (
    DisasterDatasetBase,
    find_with_ext,
    read_image_any,
    tile_channels,
)


class AvalancheDataset(DisasterDatasetBase):
    """Avalanche debris outline segmentation dataset."""

    NUM_CLASSES = 2       # 0=background, 1=avalanche debris
    IN_CHANNELS = 4

    def _sample_exists(self, sample_id: str) -> bool:
        name = str(sample_id)
        return all([
            find_with_ext(os.path.join(self.dataset_path, "IMG"), name) is not None,
            find_with_ext(os.path.join(self.dataset_path, "GT"), name) is not None,
        ])

    def __getitem__(self, index: int):
        sample_id = self.data_list[index]
        name = str(sample_id)

        img_path = find_with_ext(os.path.join(self.dataset_path, "IMG"), name)
        gt_path = find_with_ext(os.path.join(self.dataset_path, "GT"), name)

        image, invalid = read_image_any(img_path, return_invalid=True)
        mask = read_image_any(gt_path)[0].squeeze()

        if invalid is not None and np.any(invalid):
            mask = mask.copy()
            mask[invalid] = 255

        # Fix channel count to 4 (R, G, B, NIR)
        image = tile_channels(image, self.IN_CHANNELS)
        mask = mask.astype(np.int64)

        image, mask = self._apply_seg_transforms(image, mask)
        return image, mask, sample_id
