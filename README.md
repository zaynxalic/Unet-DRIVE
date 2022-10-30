# U-Net(Convolutional Networks for Biomedical Sementic Segmentation)

## Reference 
* [https://github.com/milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)
* [https://github.com/pytorch/vision](https://github.com/pytorch/vision)

## Enviroument
* Python3.6/3.7/3.8
* Pytorch1.10
* Ubuntu Or CentOS(Windows do not support multi-GPU traning)
* Training using GPU
* Enviroument Config`requirements.txt`

## File Structure:
```
  ├── src: Construct U-net
  ├── train_utils: Training, Validation and Multi-GPU training model
  ├── my_dataset.py: Dataset for reading DRIVE dataset(Retinal vascular segmentation)
  ├── train.py: Training in single GPU
  ├── train_multi_GPU.py: Trining in multiple GPU
  ├── predict.py: predict script, using trained weights test the result
  └── compute_mean_std.py: statistic of mean and standard for each channel
```

## Download DRIVE datasets:
* Official: [https://drive.grand-challenge.org/](https://drive.grand-challenge.org/)


## training method
* Make sure to prepare datasets
* If training on single GPU or cpu, using traing.py script
* If using multi-GPU using `torchrun --nproc_per_node=8 train_multi_GPU.py`commend,`nproc_per_node`parameter is the number of GPU
* If want to specify which GPU want to use, add `CUDA_VISIBLE_DEVICES=0,3` to the front of commend(For example, only use first and fourth GPU in computer)
* `CUDA_VISIBLE_DEVICES=0,3 torchrun --nproc_per_node=2 train_multi_GPU.py`
```
/home/ning/anaconda3/envs/ning/bin/python /home/ning/Desktop/Aaron/Unet-DRIVE/train.py
```

## Visualization of Result
* 

## Notification
* When running training script, need to specify `--data-path`to the file where your root folder of your `DRIVE` file**Root Folder**
* When running prediction, need to specify `weights_path` to your own generated weights folder
* When running validation files, make sure your testing and validation datasets must contain each target classes you want，and only need to modify `--num-classes`、`--data-path` and `--weights`, Try do not modify any other codes.

## pre-trained weights using Unet running on DRIVE datasets(Only for testing)
- link: https://pan.baidu.com/s/1BOqkEpgt1XRqziyc941Hcw  password: p50a

## Our Unet use Bilinear interpolation upsampling by default.
![u-net](unet.png)
