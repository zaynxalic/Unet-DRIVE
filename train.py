import os
import time
import datetime
import torch
from src import UNet, Unetpp
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from drive_dataset import DriveDataset
import transforms as T
import yaml
from torchvision import transforms as F
from pytorch_ranger import Ranger
import re
import shutil

class EnvVarLoader(yaml.SafeLoader):
    pass

class extract_dict(object):
    """
    The object can be read by call instead of using dictionary
    """
    def __init__(self, d):
        self.__dict__ = d

class Preprocessing:
    def __init__(self, 
        base_size = None, 
        crop_size = None,
        hflip_prob = 0.5,
        vflip_prob = 0.5,
        mean = (0.485, 0.456, 0.406), 
        std = (0.229, 0.224, 0.225), 
        train = True) -> None:
        
        if train:
            assert base_size is not None and crop_size is not None

            min_size :int = int(0.5 * base_size)
            max_size :int = int(1.2 * base_size)

            trans = [T.RandomResize(min_size, max_size)]
            if hflip_prob > 0:
                trans.append(T.RandomHorizontalFlip(hflip_prob))
            if vflip_prob > 0:
                trans.append(T.RandomVerticalFlip(vflip_prob))
            trans.extend([
                T.RandomCrop(crop_size),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])
            self.transforms = T.Compose(trans)

        else: 
            trans = [ T.ToTensor(), T.Normalize(mean=mean, std=std),]
            self.transforms = T.Compose(trans)
        
    def __call__(self, img, target):
        return self.transforms(img, target)

def main(configs):
    if torch.cuda.is_available():
        # device = torch.device(f'cuda:{torch.cuda.device_count()-1}')
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    batch_size = configs.batch_size
    # segmentation nun_classes + background
    num_classes = configs.num_classes

    # using compute_mean_std.py
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # save the weight
    results_file = f"{datetime.datetime.now().strftime('%m%d%H%M%S')}_{configs.model_id}.txt"
    train_dataset = DriveDataset(r"./",
                                 train=True,
                                 transforms=Preprocessing(base_size = 565, crop_size = 480, mean=mean, std=std, train = True))

    val_dataset = DriveDataset(r"./",
                               train=False,
                               transforms=Preprocessing(base_size = 565, crop_size = 480, mean=mean, std=std, train = False))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=configs.num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=configs.num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)
    model = None
    is_cbam = configs.is_cbam 
    is_aspp = configs.is_aspp
    is_sqex = configs.is_sqex
    lossfunc = configs.loss
    UNet_base_c = configs.UNet_base_c
    Unetpp_base_c = configs.Unetpp_base_c
    config_saved = False

    
    print(f'Parameters loss function: {lossfunc}, is_cbam: {is_cbam}, is_aspp: {is_aspp}, is_sqex: {is_sqex}, UNet_base_c: {UNet_base_c}, Unetpp_base_c: {Unetpp_base_c}')
    if(configs.mode == "unet"):
        # 32 16 8
        model = UNet(in_channels=3, num_classes=num_classes, base_c=UNet_base_c, is_cbam = is_cbam, is_aspp = is_aspp, is_sqex = is_sqex).to(device)
    elif(configs.mode == "unetpp"):
        model = Unetpp(in_channels=3, num_classes=num_classes, base_c=Unetpp_base_c, is_cbam = is_cbam, is_aspp = is_aspp, is_sqex = is_sqex).to(device)
    else:
        raise ValueError(f"The variable {configs.mode} is not defined.")
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The total number of trainable parameters are {total_params}")
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    # optimizer = torch.optim.SGD(
    #     params_to_optimize,
    #     lr=configs.lr, momentum=configs.momentum, weight_decay=configs.weight_decay
    # )

    optimizer = Ranger(model.parameters(), lr=configs.lr, weight_decay=configs.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if configs.amp == 1 else None

    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), configs.epochs, warmup=True)

    if configs.resume == 1:
        checkpoint = torch.load(configs.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        configs.start_epoch = checkpoint['epoch'] + 1
        if configs.amp == 1:
            scaler.load_state_dict(checkpoint["scaler"])

    best_dice = 0.
    best_rvd = 100.
    start_time = time.time()
    for epoch in range(configs.start_epoch, configs.epochs):
        mean_loss, lr = train_one_epoch(lossfunc, model, optimizer, train_loader, device, epoch, num_classes,
                                        lr_scheduler=lr_scheduler, print_freq=configs.print_freq, scaler=scaler)

        confmat, dice = evaluate(model, val_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        # TODO: check whether the value is for rvd
        if re.findall(r'rvd : (.*)\n', val_info):
            rvd = float(re.findall(r'rvd : (.*)\n', val_info)[0])
        else:
            rvd = 10000
        print(val_info)
        print(f"rvd: {rvd}")
        print(f"dice coefficient: {dice:.3f}")
        # write into txt
        with open(results_file, "a") as f:
            # record train_loss, lr and validation dataset metrices for each epoch
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n" \
                         f"dice coefficient: {dice:.3f}\n"
            f.write(train_info + val_info + "\n\n")

        if configs.save_best == 1:
            # if best_dice <=  dice and best_rvd > abs(rvd):
            if best_dice <  dice:
                best_dice = dice
                # best_rvd = abs(rvd)
            else:
                continue

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch}
        if configs.amp == 1:
            save_file["scaler"] = scaler.state_dict()

        if configs.save_best == 1:
            torch.save(save_file, "save_weights/best_model_" + configs.model_id + ".pth")
            if(config_saved == False):
                config_saved = True
                src_path = r"/home/ning/Desktop/Aaron/Unet-DRIVE/train.config"
                dst_path = r"/home/ning/Desktop/Aaron/Unet-DRIVE/configs/" + configs.model_id + ".config"
            shutil.copy(src_path, dst_path)
        else:
            torch.save(save_file, "save_weights/model_{}.pth".format(epoch))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


if __name__ == '__main__':  
    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")
    if not os.path.exists("./configs"):
        os.mkdir("./configs")
    
    configs = yaml.load(open('train.config'), Loader=EnvVarLoader)
    configs = extract_dict(configs)
    main(configs = configs)