# UrbanSARFloods - SegFormer mit-b2 - Test Results

- Generated: 2026-10-24T13:17:33
- Model: segformer (mit-b2), 24.74M params, 8-channel multi-temporal VV/VH input, 256x256 tiles
- Protocol: UrbanSARFloods 3-class (0=background, 1=flood in open areas, 2=flood in urban areas)
- Eval: CPU (torch 2.14.1+cpu), eval_batch_size=4
- Command: WANDB_MODE=disabled python test.py --exp_path results/flood/urbansar/seg_segformer_b2 --test_splits data/flood/UrbanSARFloods/test_weihui.txt data/flood/UrbanSARFloods/test_nova.txt data/flood/UrbanSARFloods/test_jubba_1.txt data/flood/UrbanSARFloods/test_jubba_2.txt --force

| metric | test_weihui | test_nova | test_jubba_1 | test_jubba_2 | average |
|---|---|---|---|---|---|
| OA | 0.9946 | 0.9295 | 0.9877 | 0.9968 | 0.9730 |
| mIoU | 0.5576 | 0.3650 | 0.5084 | 0.5034 | 0.4172 |

- Note: a sampled scan of the testing ground truth found only labels {0, 1} - class 2 (urban flood) is absent from the test GT, so its per-class metrics are 0.0 across all events. This is a property of the benchmark testing set, not an evaluation failure.
- test_weihui / per_class_iou: [0.9945649276343422, 0.6782461565359358, 0.0]
- test_weihui / per_class_f1: [0.9972750086906159, 0.8082796581815, 0.0]
- test_weihui / per_class_precision: [0.9982431473194205, 0.745759308931117, 0.0]
- test_weihui / per_class_recall: [0.9963088459325183, 0.8822420788447877, 0.0]
- test_nova / per_class_iou: [0.9285248352250699, 0.16650858113507838, 0.0]
- test_nova / per_class_f1: [0.9629378580997595, 0.285481934812023, 0.0]
- test_nova / per_class_precision: [0.9949661139547659, 0.17569727296139032, 0.0]
- test_nova / per_class_recall: [0.9329073879194628, 0.7609837499856068, 0.0]
- test_jubba_1 / per_class_iou: [0.9875092279342474, 0.5377704805599539, 0.0]
- test_jubba_1 / per_class_f1: [0.9937153139910603, 0.6994157480598837, 0.0]
- test_jubba_1 / per_class_precision: [0.9998764579570724, 0.5393328783008894, 0.0]
- test_jubba_1 / per_class_recall: [0.9876297325699531, 0.994641977522885, 0.0]
- test_jubba_2 / per_class_iou: [0.996809080313362, 0.5133498909528761, 0.0]
- test_jubba_2 / per_class_f1: [0.998401940596963, 0.6784285110652063, 0.0]
- test_jubba_2 / per_class_precision: [0.9998989422919041, 0.5201911315656823, 0.0]
- test_jubba_2 / per_class_recall: [0.99690951438325, 0.9750211475519558, 0.0]
- average / per_class_iou: [0.9727228541675595, 0.2788957980845937, 0.0]
- average / per_class_f1: [0.9861727944210481, 0.43615089546846464, 0.0]
- average / per_class_precision: [0.9979368087143758, 0.29281943976015246, 0.0]
- average / per_class_recall: [0.9746830026256699, 0.8543395998488081, 0.0]
