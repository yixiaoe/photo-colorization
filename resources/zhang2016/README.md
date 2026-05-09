# Zhang2016 Resources

This directory stores static resources required to reproduce Zhang et al. 2016, "Colorful Image Colorization".

## Files

- `pts_in_hull.npy`
  - Shape: `(313, 2)`
  - Meaning: 313 in-gamut ab color bin centers in CIE Lab space.
  - Usage: nearest-bin encoding during training and annealed-mean decoding during inference.

- `prior_probs.npy`
  - Shape: `(313,)`
  - Meaning: official color-class prior probabilities from the original Zhang2016 implementation.
  - Usage: class rebalancing for weighted cross entropy.

## Source

These two files were migrated from the local official implementation:

```text
/Users/liubingyi/Documents/CV/official_colorization_2016/resources/pts_in_hull.npy
/Users/liubingyi/Documents/CV/official_colorization_2016/resources/prior_probs.npy
```

The official prior should be used first for Task-05 to Task-07. If we later compute priors from our own dataset, save them separately, for example:

```text
prior_probs_imagenet_mini.npy
prior_counts_imagenet_mini.npy
```

Do not overwrite the official `prior_probs.npy` unless the team explicitly decides to change the baseline.

