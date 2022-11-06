import torch
from torch import nn
import train_utils.distributed_utils as utils
from src import UNet,Unetpp
import os as os
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import yaml
from os import listdir
from os.path import isfile, join

class EnvVarLoader(yaml.SafeLoader):
    pass

# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
class extract_dict(object):
    """
    The object can be read by call instead of using dictionary
    """
    def __init__(self, d):
        self.__dict__ = d
        
#  /home/ning/anaconda3/envs/ning/bin/python /home/ning/Desktop/Aaron/Unet-DRIVE/demo_batch.py
if __name__ == '__main__':
    configs_path = r"./configs"
    weights_path = r"./save_weights"
    img_path = "./DRIVE/test/images"
    roi_mask_path = "./DRIVE/test/mask"
    
    assert os.path.exists(configs_path), f"weights {configs_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."
    assert os.path.exists(roi_mask_path), f"image {roi_mask_path} not found."
    
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean= (0.709, 0.381, 0.224),
                                                              std= (0.127, 0.079, 0.043))])
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{torch.cuda.device_count()-1}')
    else:
        device = torch.device('cpu')
    
    # iterate to run and save all result image
    configsfiles = [f for f in listdir(configs_path) if isfile(join(configs_path, f))]
    imgfiles = [f for f in listdir(img_path) if isfile(join(img_path, f))]
    maskfiles = [f for f in listdir(roi_mask_path) if isfile(join(roi_mask_path, f))]
    
    for configfile in configsfiles:
        configs = yaml.load(open(configs_path +"/"+ configfile), Loader=EnvVarLoader)
        configs = extract_dict(configs)
        if(configs.mode == 'unet'):
            print(configfile)
            model = UNet(in_channels=3, num_classes=2, base_c=configs.Unetpp_base_c, is_cbam = configs.is_cbam, is_aspp = configs.is_aspp, is_sqex = configs.is_sqex).to(device)
        elif configs.mode == 'unetpp':
            print(configfile)
            model = Unetpp(in_channels=3, num_classes=2,base_c=configs.UNet_base_c, is_cbam = configs.is_cbam, is_aspp = configs.is_aspp, is_sqex = configs.is_sqex).to(device)
        model.load_state_dict(torch.load(weights_path + "/best_model_" + configs.model_id + ".pth", map_location='cpu')['model'])
        model.to(device)
        model.eval()
        for i in range(len(imgfiles)):
            roi_img = Image.open(roi_mask_path+ '/' + maskfiles[i]).convert('L')
            roi_img = np.array(roi_img)
            original_img = Image.open(img_path + '/' + imgfiles[i]).convert('RGB')
            img = data_transform(original_img).unsqueeze(0)
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)
            output = model(img.to(device))

            prediction = output['out'].argmax(1).squeeze(0)
            prediction = prediction.to("cpu").numpy().astype(np.uint8)
            prediction[prediction == 1] = 255
            prediction[roi_img == 0] = 0
            mask = Image.fromarray(prediction)
            # Returns true if the request was successful.
            if not os.path.exists("./result/" + configs.model_id):
                os.mkdir("./result/" + configs.model_id)
            mask.save("./result/" + configs.model_id + "/" + imgfiles[i] + ".png")
        print('finished')
        # plt.imshow(mask)
        # plt.show()
        

           