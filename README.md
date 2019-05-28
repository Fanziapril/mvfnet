# MVF-Net: Multi-View 3D Face Morphable Model Regression
Testing code for the paper.
> [MVF-Net: Multi-View 3D Face Morphable Model Regression](https://arxiv.org/abs/1904.04473).   
> Fanzi Wu*, Linchao Bao*, Yajing Chen, Yonggen Ling, Yibing Song, Songnan Li, King Ngi Ngan, Wei Liu. 
> CVPR 2019.

## Todo
- Code for rendering images.
- Replace image cropping scheme.

## Installation
1. Python 2.7 (Numpy, PIL, scipy)
2. Pytorch 0.4.0, torchvision
3. face-alignment package from [https://github.com/1adrianb/face-alignment](https://github.com/1adrianb/face-alignment). This code is used for face cropping and will be replaced by face detection algorithm in the future.

4. `Model_shape.mat` and `Model_Expression.mat` from [3DDFA](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm).
## Test
You can download the CNN model from [here](https://www.dropbox.com/s/7ds3aesjjmybjh9/net.pth?dl=0) and copy it into `data` folder.
Then you can test the model by:
```
python test_img.py --image_path ./data/imgs --save_dir ./result
```
If you are testing the code with your own images, please organize multiview images as:
```
folder
+--front.jpg
+--left.jpg
+--right.jpg
```
and change `line 15` in `test_img.py` as:
```
crop_opt = True
```
## Citation
If you find this work useful in your research, please cite:
```
@article{wu2019mvf,
  title={MVF-Net: Multi-View 3D Face Morphable Model Regression},
  author={Wu, Fanzi and Bao, Linchao and Chen, Yajing and Ling, Yonggen and Song, Yibing and Li, Songnan and Ngan, King Ngi and Liu, Wei},
  journal={arXiv preprint arXiv:1904.04473},
  year={2019}
}
```
