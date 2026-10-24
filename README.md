# Natural Disaster and Hazard Pre- and Post-Analysis (SAR Flood Mapping Pipeline)

End-to-end SAR flood mapping pipeline: benchmark data download, tile
extraction, SegFormer (mit-b2) training, inference over all test tiles,
and per-event accuracy reporting for flood events (weihui, nova,
jubba_1, jubba_2) under the UrbanSARFloods 3-class protocol:

- class 0: background
- class 1: flood in open areas
- class 2: flood in urban areas

## Pipeline stages

1. Data download for each event
2. Tiling of SAR images and labels
3. Training (SegFormer mit-b2, UrbanSARFloods protocol)
4. Inference on all test tiles
5. Per-event metrics (OA, mIoU, per-class IoU, F1)
6. Packaging of all artifacts into adm_final_artifacts.tar.gz

## Current status

- Training completed: validation OA 98.69%, mIoU 57.70%
- Checkpoint available for direct inference
- Evaluation and artifact packaging in progress

## Repository notes

- Detailed usage instructions and original framework documentation
  are preserved in docs/UPSTREAM_README.md
- The LICENSE file is retained unchanged

## Author

Syed Rafi Ahmed
GitHub: https://github.com/SyedRafiAhmedGIS

## Credits

This project builds on the open-source AnyDisasterMapping framework.
The original framework README, paper citation, and license information
are preserved in docs/UPSTREAM_README.md and in the LICENSE file.
