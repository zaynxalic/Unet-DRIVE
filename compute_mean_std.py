import os
from PIL import Image
import numpy as np


def main():
    img_dir = r"./DRIVE/training/images"
    roi_dir = r"./DRIVE/training/mask"

    img_name_list = [i for i in os.listdir(img_dir) if i.endswith(".tif")]

    mean = np.zeros((len(img_name_list), 3))
    std = np.zeros((len(img_name_list), 3))
    
    for idx,img_name in enumerate(img_name_list):
        img_path = os.path.join(img_dir, img_name)
        ori_path = os.path.join(roi_dir, img_name.replace(".tif", "_mask.gif"))
        img = np.array(Image.open(img_path)) / 255.0
        roi_img = np.array(Image.open(ori_path).convert('L'))

        img = img[roi_img == 255]
        mean[idx] = img.mean(axis=0)
        std[idx] += img.std(axis=0)

    avg_mean = np.mean(mean,axis=0)
    avg_std = np.mean(std,axis=0)
    print(f"mean: {avg_mean}")
    print(f"std: {avg_std}")


if __name__ == '__main__':
    main()
