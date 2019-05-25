# MVF-Net: Multi-View 3D Face Morphable Model Regression
Testing code for the paper.
> [MVF-Net: Multi-View 3D Face Morphable Model Regression](https://arxiv.org/abs/1904.04473).   
> Fanzi Wu*, Linchao Bao*, Yajing Chen, Yonggen Ling, Yibing Song, Songnan Li, King Ngi Ngan, Wei Liu. 
> CVPR 2019.

## Installation
1. Python 2.7
2. Pytorch 0.4

## Test
A simple example to test:
```
python test_img.py --image_path ./data/06_01 --save_dir ./result
```
If you are testing the code with your own images, please organize multiview images as:
```
folder.
+--front.jpg.
+--left.jpg.
+--right.jpg.
```
and change `line 15` in `test_img.py` as:
```
crop_opt = True
```
