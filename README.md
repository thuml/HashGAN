HashGAN: Deep Learning to Hash with Pair Conditional Wasserstein GAN
=====================================

Code for CVPR 2018 Paper ["HashGAN: Deep Learning to Hash with Pair Conditional Wasserstein GAN"](http://openaccess.thecvf.com/content_cvpr_2018/papers/Cao_HashGAN_Deep_Learning_CVPR_2018_paper.pdf).


## Prerequisites

- Python3, NumPy, TensorFlow-gpu, SciPy, Matplotlib, OpenCV, easydict, yacs, tqdm
- A recent NVIDIA GPU

We provide a `environment.yaml` for you and you can simplely use `conda env create -f environment.yml` to create the environment.

Or you can create the environment from scratch:
```bash
conda create --no-default-packages -n HashGAN python=3.6 && source activate HashGAN
conda install -y numpy scipy matplotlib  tensorflow-gpu opencv
pip install easydict yacs tqdm pillow
```

## Data Preparation
In `data_list/` folder, we give three examples to show how to prepare image training data. If you want to add other datasets as the input, you need to prepare `train.txt`, `test.txt`, `database.txt` and `database_nolabel.txt` as CIFAR-10 dataset.

You can download the whole cifar10 dataset including the images and data list from [here](https://github.com/thulab/DeepHash/releases/download/v0.1/cifar10.zip), and unzip it to `data/cifar10` folder.

If you need run on NUSWIDE_81 and COCO, we recommend you to follow [here](https://github.com/thuml/HashNet/tree/master/pytorch#datasets) to prepare NUSWIDE_81 and COCO images.

## Pretrained Models

The imagenet pretrained Alexnet model can be downloaded [here](https://github.com/thulab/DeepHash/releases/download/v0.1/reference_pretrain.npy.zip).
You can download the pretrained Generator models in the [release page](https://github.com/thuml/HashGAN/releases) and modify config file to use the pretrained models.

## Training

The training process can be divided into two step:
1. Training a image generator.
2. Fintune Alexnet using original labeled images and generated images.

In `config` folder, we provide some examples to prepare yaml configuration.

```
config
├── cifar_evaluation.yaml
├── cifar_step_1.yaml
├── cifar_step_2.yaml
└── nuswide_step_1.yaml
```

You can run the model using command like the following:

- `python main.py --cfg config/cifar_step_1.yaml --gpus 0`
- `python main.py --cfg config/cifar_step_2.yaml --gpus 0`

You can use tensorboard to monitor the training process such as losses and Mean Average Precision.

## Citation
If you use this code for your research, please consider citing:
```
 @inproceedings{cao2018hashgan,
  title={HashGAN: Deep Learning to Hash with Pair Conditional Wasserstein GAN},
  author={Cao, Yue and Liu, Bin and Long, Mingsheng and Wang, Jianmin and KLiss, MOE},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={1287--1296},
  year={2018}
}
```

## Contact
If you have any problem about our code, feel free to contact 
- liubinthss@gmail.com
- caoyue10@gmail.com
  
or describe your problem in Issues.
