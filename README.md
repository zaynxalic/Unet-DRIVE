
# U-Net(Convolutional Networks for Biomedical Sementic Segmentation)

## Group
Group Name: Project Group 15

Group Members:
Xuecheng Zhang u6284513
Junyi Men u7233481
Ke Ning u7175553

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
  ├── configs: history ofparameters used by training, name of config indicate the parameters used by abligation study
  ├── DRIVE: Dataset used.
  ├── src: Construct U-net
  ├── train_utils: Training, Validation and Multi-GPU training model
  ├── my_dataset.py: Dataset for reading DRIVE dataset(Retinal vascular segmentation)
  ├── compute_mean_std.py: Compute the mean and standard for dataset, used by pre-processing.
  ├── drive_dataset.py: load dataset from DRIVE
  ├── train.py: Training in single GPU.
  ├── predict.py: predict the visual result, using all trained weights test the result for all images in dataset.
  ├── predict.py: predict the visual result, using specified weights test the result for single image.
  └── plot.py: Plot the training process and saved to current folder
  └── train.config: Config parameters of traning
  └── train.py: train the model based on parameters
  └── transforms.py: image transforms, resize, crop etc.
```

## Download DRIVE datasets:
* Official: [https://drive.grand-challenge.org/](https://drive.grand-challenge.org/)


## training method
* Make sure to prepare datasets
* Make sure your current folder is in the root folder of UNet-DRIVE, before you run the script.
* If training on single GPU or cpu, using traing.py using script
```
python train.py
```

## Visualization of Result
* After training, the folder will save a new weights in 'save_weights' folder, a new config in 'configs' folder
* If want to predict the result and save segmented images, running script
```
python predict_batch.py
```
* If want to predict the single image, modify the path in the file predict.py, then running script
```
python predict.py
```

## Notification
* When running training script, need to specify `--data-path`to the file where your root folder of your `DRIVE` file**Root Folder**
* When running prediction, need to specify `weights_path` to your own generated weights folder
* When running validation files, make sure your testing and validation datasets must contain each target classes you want，and only need to modify `--num-classes`、`--data-path` and `--weights`, Try do not modify any other codes.

## pre-trained weights using Unet running on DRIVE datasets(Only for testing)
- link: https://pan.baidu.com/s/1BOqkEpgt1XRqziyc941Hcw  password: p50a
