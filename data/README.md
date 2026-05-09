# data

This directory is reserved for local datasets and test images.

Do not commit dataset files to Git. The current Phase 1 work expects one of these layouts:

```text
data/
├── imagenet_mini/
│   ├── train/
│   ├── val/
│   └── test/
└── cifar10/
```

Example command paths from `code/`:

```bash
python train.py --method zhang2016 --dataset imagenet_mini --data_dir ../data/imagenet_mini/train
python test.py --method zhang2016 --test_img_dir ../data/imagenet_mini/test
```

