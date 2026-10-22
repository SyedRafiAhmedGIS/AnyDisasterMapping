"""Volcanic-eruption dataset (extended domain: volcanic eruptions).

Covers very-high-resolution optical tri-stereo / PlanetScope imagery with
volcanic feature mappings in the style of ESA Phi-Lab PyRawS (lava flows,
vents, fumaroles) and the Sundhnúkur (Iceland) lava-flow mappings.

Directory layout expected under *dataset_path*::

    {dataset_path}/IMG/{id}.tif      -- RGB image (3 ch)
    {dataset_path}/GT/{id}.tif       -- 0=background, 1=lava flow,
                                        2=vent/crater, 3=fumarole

Split files (one ID per line): train.txt / val.txt / test.txt.

Returns
-------
3-tuple : (image, label, sample_id)
    - image    : float32 (3, H, W)
    - label    : int64   (H, W)  -- 4-class volcanic mask, 255 = ignore
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


class VolcanoPyRawsDataset(DisasterDatasetBase):
    """Volcanic feature segmentation dataset (PyRawS-style)."""

    NUM_CLASSES = 4       # 0=background, 1=lava flow, 2=vent, 3=fumarole
    IN_CHANNELS = 3

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

        image = tile_channels(image, self.IN_CHANNELS)
        mask = mask.astype(np.int64)

        image, mask = self._apply_seg_transforms(image, mask)
        return image, mask, sample_id
