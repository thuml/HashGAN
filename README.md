HashGAN: Deep Learning to Hash with Pair Conditional Wasserstein GAN
=====================================

Code for ["HashGAN: Deep Learning to Hash with Pair Conditional Wasserstein GAN"](http://openaccess.thecvf.com/content_cvpr_2018/papers/Cao_HashGAN_Deep_Learning_CVPR_2018_paper.pdf).


## Prerequisites

- Python3, NumPy, TensorFlow, SciPy, Matplotlib
- A recent NVIDIA GPU

## Data Preparation
In `data_list/` folder, we give three examples to show how to prepare image training data. If you want to add other datasets as the input, you need to prepare `train.txt`, `test.txt`, `database.txt` and `database_nolabel.txt` as CIFAR-10 dataset.

You can download the whole cifar10 dataset including the images and data list from [here](https://github.com/thulab/DeepHash/releases/download/v0.1/cifar10.zip), and unzip it to data/cifar10 folder.

Make sure the tree of `/path/to/project/data/cifar10` looks like this:

```
.
|-- database.txt
|-- database_nolabel.txt
|-- test
|-- test.txt
|-- train
`-- train.txt
```

If you need run on NUSWIDE_81 and COCO, we recommend you to follow https://github.com/thuml/HashNet/tree/master/pytorch#datasets to prepare NUSWIDE_81 and COCO images.

## Models
### TODO

- [ ] Pretrain model of Alexnet
- [ ] pretrained G model
- [ ] eval frequence & eval at last iter
- [ ] refactor all 
  - [ ] use config instead of constant
  - [ ] use no split
  - [ ] evaluate mode
  - [ ] output dir which contains images, models, logs
    - [ ] mkdir automatically
- [ ] training longger
- [ ] resume training
- [ ] rerun all process on a fresh machine

Configuration for th models is specified in a list of constants at the top of
the file, you can use the following command to run it:

- `python main.py`

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
- caoyue10@gmail.com
- liubinthss@gmail.com

or describe your problem in Issues.
