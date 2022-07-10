# Official Code for paper ``Improving Adversarial Robustness of 3D Point Cloud Classification Models'', which will appear in ECCV 2022.

## Requirements
* Tensorflow>=1.14.0 (not support Tensorflow 2.0)
* Pytorch>=1.2.0
* [PointCutMix-K](https://github.com/cuge1995/PointCutMix)

## Compiling Cuda Operations

Please follow this [repo](https://github.com/charlesq34/pointnet2).

## Dataset

The ModelNet40 can be downloaded from [here](https://modelnet.cs.princeton.edu/).

## Training and Evaluating

Pretrained model can be found in [here](https://entuedu-my.sharepoint.com/:u:/g/personal/guanlin001_e_ntu_edu_sg/Ees8EDm_JyBNirQ_rYmTCdoBOowMzTUMSVFwHh8Y1IP7cA?e=Ij8hAE). Or, you can train your own model from scratch.
```
python train.py
```

## Acknowledgment

Parts of code are from [DGCNN](https://github.com/WangYueFt/dgcnn),
[PointCloud-Saliency-Map](shttps://github.com/tianzheng4/PointCloud-Saliency-Maps) 
and [PointCutMix-K](https://github.com/cuge1995/PointCutMix).
