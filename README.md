# Notes
- This is a fork of the FixMatch-Pytorch repo from https://github.com/kekmodel/FixMatch-pytorch
- The **master** branch contains the latest **analysis** code.
- The **training** branch contains the latest **SSL training** code. 

## Prelimminary CIFAR10 Top-1
| #Labels per class| 10 | 40 | 400 |
|:---|:---:|:---:|:---:|
| Accuracy | 92.37 | 94.03 | 95.09 |

## Plan
Realisticity Analysis
- Hypothesis 1: strongly and weakly augmented images have different realisiticity and we can measure/distinct this using out-of-distribution methods
- Hypothesis 2: there are variation of realisiticity even among strongly augmented images. e.g. we expect posterization to be more unrealisitic than flip+translate+crop

Improving SSL
- Hypothesis 3: we can improve SSL by training with appropriate/various levels of unrealisticity of strong augmentation

## TODO
- [X] Test run this repo with CIFAR-10

Realisticity Analysis
- [X] Integrate OOD detection for measuring "realisiticity" of augmented images e.g. https://arxiv.org/abs/1912.03263 https://arxiv.org/abs/1905.11001
- [X] Manually check ODD score and qualitative realisticity of sample augmented images. See **save-aug** branch for the extraction sample augmented images (before normalization)
- [X] Generate/save all augmented variation as use in one epoch of FixMatch
- [ ] Analysize the ODD scores of this distribution augmented images to test hypotheses 1 and 2

Improving SSL
- [ ] Train FixMatch with various levels of unrealistic augmentation

=======
---------------------------------------------
*(below is the README from the original repo)*
# FixMatch
This is an unofficial PyTorch implementation of [FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence](https://arxiv.org/abs/2001.07685).
The official Tensorflow implementation is [here](https://github.com/google-research/fixmatch).

This code is only available in FixMatch (RandAugment).
Now only experiments on CIFAR-10 and CIFAR-100 are available.

## Requirements
- Python 3.6+
- PyTorch 1.4
- torchvision 0.5
- tensorboard
- tqdm
- numpy
- apex (optional)

## Usage

### Train
Train the model by 4000 labeled data of CIFAR-10 dataset:

```
python train.py --dataset cifar10 --num-labeled 4000 --arch wideresnet --batch-size 64 --lr 0.03 --seed 5 --out results/cifar10@4000.5
```

Train the model by 10000 labeled data of CIFAR-100 dataset by using DistributedDataParallel:
```
python -m torch.distributed.launch --nproc_per_node 4 ./train.py --dataset cifar100 --num-labeled 10000 --arch wideresnet --batch-size 16 --lr 0.03 --out results/cifar100@10000
```
\* When using DDP, do not use a seed.

### Monitoring training progress
```
tensorboard --logdir=<your out_dir>
```

## Results (Accuracy)

### CIFAR10
| #Labels | 40 | 250 | 4000 |
|:---|:---:|:---:|:---:|
| Paper (RA) | 86.19 ± 3.37 | 94.93 ± 0.65 | 95.74 ± 0.05 |
| This code | 92.92 | 94.13 | 95.33 |
| Acc. curve | [link](https://tensorboard.dev/experiment/YcLQA52kQ1KZIgND8bGijw/) | [link](https://tensorboard.dev/experiment/n3Wd14QRTZWEKXlmbQxfvw/) | [link](https://tensorboard.dev/experiment/MX4pVoLmRMuq7VTQV9M8ww/) |

### CIFAR100
| #Labels | 400 | 2500 | 10000 |
|:---|:---:|:---:|:---:|
|Paper (RA) | 51.15 ± 1.75 | 71.71 ± 0.11 | 77.40 ± 0.12 |
|This code | - | - | - |
| Acc. curve | - | - | - |

## References
- [Unofficial PyTorch implementation of MixMatch: A Holistic Approach to Semi-Supervised Learning](https://github.com/YU1ut/MixMatch-pytorch)
```
@article{sohn2020fixmatch,
    title={FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence},
    author={Kihyuk Sohn and David Berthelot and Chun-Liang Li and Zizhao Zhang and Nicholas Carlini and Ekin D. Cubuk and Alex Kurakin and Han Zhang and Colin Raffel},
    journal={arXiv preprint arXiv:2001.07685},
    year={2020},
}
```
