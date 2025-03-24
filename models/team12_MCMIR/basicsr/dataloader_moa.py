from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
from pathlib import Path
from PIL import Image
import os
import torchvision.transforms.functional as TF
from itertools import permutations
import numpy as np
import cv2
import random   
import json
from torch.utils.data.dataset import Dataset
import torch
from basicsr.data.transforms import augment, paired_random_crop
import augmentation
import math
import torch.nn.functional as F
from degradation import *
from pathlib import Path

class TrainSetLoader_deg(Dataset):
    def __init__(self, cfg):
        super(TrainSetLoader_deg, self).__init__()
        self.dataset_dir = cfg.data_dir +'/HR/train/'

        # TODO file paths
        self.file_list = sorted(self.get_all_image_paths(self.dataset_dir))
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.outputSize = cfg.img_size
        self.degradation_class =  Superresolution(sf = 4,dim_image = self.outputSize,mode="bicubic",device="cpu")
        
    def get_all_image_paths(self, root_dir, extensions=("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff")):
        """
        Recursively finds all image file paths under the root directory, including those in subdirectories.
        """
        return sorted([str(file) for ext in extensions for file in Path(root_dir).rglob(ext)])

    def __getitem__(self, index):
        hr_path = self.file_list[index]
        
        rgb_hr = cv2.imread(hr_path)
        
        hr_rgb_img = np.array(cv2.cvtColor(rgb_hr, cv2.COLOR_BGR2RGB)).astype(np.float32)
        hr_rgb_final = torch.from_numpy(hr_rgb_img.copy()).permute(2, 0, 1).unsqueeze(0)

        _,_,H,W = hr_rgb_final.shape
        spatial_arr = [H,W]
        if min(H,W) < self.outputSize:
            min_index = spatial_arr.index(min(spatial_arr))
            ratio = self.outputSize / spatial_arr[min_index]
            hr_rgb_final = F.interpolate(hr_rgb_final,(math.ceil(H*ratio),math.ceil(W*ratio)),mode='bilinear',align_corners=False)
        
        


        # #crop a patch with scale factor = 2
        # i,j,h,w = transforms.RandomCrop.get_params(lr_raw_final, output_size = (64,64))
        i,j,h,w = transforms.RandomCrop.get_params(hr_rgb_final, output_size = (self.outputSize,self.outputSize))
        high = TF.crop(hr_rgb_final, i, j, h, w)
        # high, low = paired_random_crop(hr_raw_final, lr_raw_final, 64, 2)
        
                
        high = high / 255     
        low = torch.clamp(self.degradation_class.H(high), min=0, max=1)


        # if "RealSR" in opt.dataset:
        #     opt.mix_p = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4] [0.6, 1.0, 1.2, 0.001, 0.7, 0.7, 0.7]
        # img_hr, img_lr = augmentation.apply_augment(high, low, ["rgb", "mixup", "vertical", "horizontal", "none"], [1.0, 1.0, 1.0, 1.0, 1.0], [0.35, 0.35, 0.1, 0.1, 0.1],
        #         1.2, 1.2, None)
        img_hr, img_lr = high, low
        img_hr, img_lr = img_hr.squeeze(0), img_lr.squeeze(0)
        return img_lr,img_hr

    def __len__(self):
        return len(self.file_list)

class ValidSetLoader_deg(Dataset):
    def __init__(self, cfg):
        super(ValidSetLoader_deg, self).__init__()
        self.dataset_dir = cfg.data_dir +'/HR/val/'

        # TODO file paths
        self.file_list = sorted(self.get_all_image_paths(self.dataset_dir))
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.outputSize = cfg.img_size
        self.degradation_class =  Superresolution(sf = 4,dim_image = self.outputSize,mode = "bicubic",device="cpu")
        
    def get_all_image_paths(self, root_dir, extensions=("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff")):
        """
        Recursively finds all image file paths under the root directory, including those in subdirectories.
        """
        return sorted([str(file) for ext in extensions for file in Path(root_dir).rglob(ext)])

    def __getitem__(self, index):
        hr_path = self.file_list[index]
        
        rgb_hr = cv2.imread(hr_path)
        
        hr_rgb_img = np.array(cv2.cvtColor(rgb_hr, cv2.COLOR_BGR2RGB)).astype(np.float32)
        hr_rgb_final = torch.from_numpy(hr_rgb_img.copy()).permute(2, 0, 1).unsqueeze(0)

        _,_,H,W = hr_rgb_final.shape
        spatial_arr = [H,W]
        if min(H,W) < self.outputSize:
            min_index = spatial_arr.index(min(spatial_arr))
            ratio = self.outputSize / spatial_arr[min_index]
            hr_rgb_final = F.interpolate(hr_rgb_final,(math.ceil(H*ratio),math.ceil(W*ratio)),mode='bilinear',align_corners=False)
        
        


        # #crop a patch with scale factor = 2
        # i,j,h,w = transforms.RandomCrop.get_params(lr_raw_final, output_size = (64,64))
        i,j,h,w = transforms.RandomCrop.get_params(hr_rgb_final, output_size = (self.outputSize,self.outputSize))
        high = TF.crop(hr_rgb_final, i, j, h, w)
        # high, low = paired_random_crop(hr_raw_final, lr_raw_final, 64, 2)
        
                
        high = high / 255     
        low = torch.clamp(self.degradation_class.H(high), min=0, max=1)


        # if "RealSR" in opt.dataset:
        #     opt.mix_p = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4] [0.6, 1.0, 1.2, 0.001, 0.7, 0.7, 0.7]
        # img_hr, img_lr = augmentation.apply_augment(high, low, ["rgb", "mixup", "vertical", "horizontal", "none"], [1.0, 1.0, 1.0, 1.0, 1.0], [0.35, 0.35, 0.1, 0.1, 0.1],
        #         1.2, 1.2, None)
        img_hr, img_lr = high, low
        img_hr, img_lr = img_hr.squeeze(0), img_lr.squeeze(0)
        return img_lr,img_hr

    def __len__(self):
        return len(self.file_list)


class TestSetLoader(Dataset):
    def __init__(self, cfg):
        super(TestSetLoader, self).__init__()
        self.dataset_dir = cfg.data_dir

        # TODO file paths
        self.file_list = sorted(self.get_all_image_paths(self.dataset_dir))
        
    def get_all_image_paths(self, root_dir, extensions=("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff")):
        """
        Recursively finds all image file paths under the root directory, including those in subdirectories.
        """
        return sorted([str(file) for ext in extensions for file in Path(root_dir).rglob(ext)])

    def __getitem__(self, index):
        lr_path = self.file_list[index]
        
        rgb_lr = cv2.imread(lr_path)
        
        lr_rgb_img = np.array(cv2.cvtColor(rgb_lr, cv2.COLOR_BGR2RGB)).astype(np.float32)
        low = torch.from_numpy(lr_rgb_img.copy()).permute(2, 0, 1).unsqueeze(0)
        
        

        # #crop a patch with scale factor = 2
        # i,j,h,w = transforms.RandomCrop.get_params(lr_raw_final, output_size = (64,64)
        # high, low = paired_random_crop(hr_raw_final, lr_raw_final, 64, 2)
        
                
        low = low / 255     


        # if "RealSR" in opt.dataset:
        #     opt.mix_p = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4] [0.6, 1.0, 1.2, 0.001, 0.7, 0.7, 0.7]
        # img_hr, img_lr = augmentation.apply_augment(high, low, ["rgb", "mixup", "vertical", "horizontal", "none"], [1.0, 1.0, 1.0, 1.0, 1.0], [0.35, 0.35, 0.1, 0.1, 0.1],
        #         1.2, 1.2, None)
        img_lr =  low
        img_lr =  img_lr.squeeze(0)
        return img_lr.contiguous(), lr_path 

    def __len__(self):
        return len(self.file_list)
    

# # class TestSetLoader(Dataset):
# #     def __init__(self, cfg_test):
# #         self.test_dir = cfg_test.testset_dir + '/hr/'
# #         self.file_list = sorted(os.listdir(self.test_dir))
# #         self.transform = transforms.Compose([transforms.ToTensor()])
# #     def __getitem__(self, index):
# #         lr_path =  str(Path(self.test_dir).resolve().parent) + '/lr/' + self.file_list[index].split('_')[0]
# #         hr_path = self.test_dir + self.file_list[index].split('_')[0]
# #         raw_hr, raw_lr = np.load(hr_path), np.load(lr_path)
# #         hr_raw_img, lr_raw_img = raw_hr["raw"], raw_lr["raw"]
# #         hr_raw_max = raw_hr["max_val"]
# #         hr_raw_final = (hr_raw_img / hr_raw_max).astype(np.float32)
# #         lr_raw_final = (lr_raw_img / hr_raw_max).astype(np.float32)
# #         img_hr_final = self.transform(hr_raw_final)
# #         img_lr_final = self.transform(lr_raw_final)

# #         return img_hr_final, img_lr_final


# #     def __len__(self):
# #         return len(self.file_list)
# class TestSetLoader(Dataset):
#     def __init__(self, cfg_test):
#         self.test_dir = cfg_test.testset_dir + '/hr/'
#         self.file_list = sorted(os.listdir(self.test_dir))
#         self.transform = transforms.Compose([transforms.ToTensor()])
#         # self.kernel_path = np.load(cfg_test.kernel_path, allow_pickle=True)
#         # self.__getitem__(1)
#         pass
#     def __getitem__(self, index):

#         hr_path = self.test_dir + self.file_list[index].split('_')[0]
#         raw_hr = np.load(hr_path)
#         hr_raw_img = raw_hr["raw"]
#         hr_raw_max = raw_hr["max_val"]

#         hr_raw_final = (hr_raw_img / hr_raw_max).astype(np.float32)
        
#         hr_raw_final = torch.from_numpy(hr_raw_final.copy()).permute(2, 0, 1)

#         # lr_raw_img = simple_deg_simulation(hr_raw_final, self.kernel_path)
#         lr_raw_final = downsample_raw(hr_raw_final).permute(2, 0, 1)
#         # print(f"test data =={hr_raw_final.shape},{lr_raw_final.shape}")
#         # lr_raw_final = lr_raw_img
#         return  lr_raw_final,hr_raw_final


#     def __len__(self):
#         return len(self.file_list)
    

class LSDIRLoader(Dataset):
    def __init__(self, cfg):
        super(LSDIRLoader, self).__init__()

        self.data_root = cfg.data_dir  # Base directory
        self.outputSize = cfg.img_size
        self.transform = transforms.Compose([transforms.ToTensor()])

        # Load file paths from JSON
        with open(cfg.json_path, 'r') as f:
            self.file_list = json.load(f)

    def __getitem__(self, index):
        # Get file paths from JSON
        hr_path = os.path.join(self.data_root, self.file_list[index]["path_gt"])
        lr_path = os.path.join(self.data_root, self.file_list[index]["path_lq"])

        # Read images
        rgb_hr = cv2.imread(hr_path)
        rgb_lr = cv2.imread(lr_path)
        
        # Convert BGR to RGB
        hr_rgb_img = cv2.cvtColor(rgb_hr, cv2.COLOR_BGR2RGB).astype(np.float32)
        lr_rgb_img = cv2.cvtColor(rgb_lr, cv2.COLOR_BGR2RGB).astype(np.float32)

        # Convert to torch tensors
        hr_rgb_final = torch.from_numpy(hr_rgb_img.copy()).permute(2, 0, 1).unsqueeze(0)
        lr_rgb_final = torch.from_numpy(lr_rgb_img.copy()).permute(2, 0, 1).unsqueeze(0)

        _, _, H, W = lr_rgb_final.shape
        lr_rgb_final = F.interpolate(lr_rgb_final, (max(self.outputSize, H), max(self.outputSize, W)), mode='bilinear', align_corners=False)
        hr_rgb_final = F.interpolate(hr_rgb_final, (max(self.outputSize, 4 * H), max(self.outputSize, 4 * W)), mode='bilinear', align_corners=False)

        # Normalize images to [0, 1]
        lr_rgb_final = lr_rgb_final / 255.0
        hr_rgb_final = hr_rgb_final / 255.0

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(lr_rgb_final, output_size=(self.outputSize, self.outputSize))
        low = TF.crop(lr_rgb_final, i, j, h, w)
        high = TF.crop(hr_rgb_final, i * 4, j * 4, 4 * h, 4 * w)

        # Apply augmentations
        # img_hr, img_lr = augmentation.apply_augment(
        #     high, low, ["rgb", "mixup", "vertical", "horizontal", "none"],
        #     [1.0, 1.0, 1.0, 1.0, 1.0], [0.35, 0.35, 0.1, 0.1, 0.1], 1.2, 1.2, None
        # )
        img_hr, img_lr = high, low
        img_hr, img_lr = img_hr.squeeze(0), img_lr.squeeze(0)

        return img_lr, img_hr

    def __len__(self):
        return len(self.file_list) 


class ValLSDIRLoader(Dataset):
    def __init__(self, cfg_test):
        super(ValLSDIRLoader, self).__init__()
        self.hr_dataset_dir = os.path.join(cfg_test.data_dir, 'HR', 'val')
        self.lr_dataset_dir = os.path.join(cfg_test.data_dir, 'X4', 'val')
        self.hr_file_list = sorted(os.listdir(self.hr_dataset_dir))
        self.lr_file_list = sorted(os.listdir(self.lr_dataset_dir))
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.outputSize = cfg_test.img_size
    def __getitem__(self, index):
        hr_path = os.path.join(self.hr_dataset_dir, self.hr_file_list[index])
        lr_path = os.path.join(self.lr_dataset_dir, self.lr_file_list[index])
        rgb_hr = cv2.imread(hr_path)
        rgb_lr = cv2.imread(lr_path)
        
        hr_rgb_img = np.array(cv2.cvtColor(rgb_hr, cv2.COLOR_BGR2RGB)) 
        lr_rgb_img = np.array(cv2.cvtColor(rgb_lr, cv2.COLOR_BGR2RGB))
        # TODO check valid 255 rgb range
        hr_rgb_img = (hr_rgb_img).astype(np.float32)
        lr_rgb_img = (lr_rgb_img).astype(np.float32)  
        hr_rgb_final = torch.from_numpy(hr_rgb_img.copy()).permute(2, 0, 1).unsqueeze(0)
        lr_rgb_final = torch.from_numpy(lr_rgb_img.copy()).permute(2, 0, 1).unsqueeze(0)
        _,_,H,W = lr_rgb_final.shape
        lr_rgb_final = F.interpolate(lr_rgb_final,(max(self.outputSize,H),max(self.outputSize,W)),mode='bilinear',align_corners=False)
        hr_rgb_final = F.interpolate(hr_rgb_final,(max(self.outputSize,4*H),max(self.outputSize,4*W)),mode='bilinear',align_corners=False)
        
        lr_rgb_final = lr_rgb_final / 255
        hr_rgb_final = hr_rgb_final / 255
        
        i,j,h,w = transforms.RandomCrop.get_params(lr_rgb_final, output_size = (self.outputSize,self.outputSize))
        low = TF.crop(lr_rgb_final, i, j, h, w)
        high = TF.crop(hr_rgb_final, i*4, j*4, 4*h, 4*w)
        # img_hr, img_lr = augmentation.apply_augment(high, low, ["rgb", "mixup", "vertical", "horizontal", "none"], [1.0, 1.0, 1.0, 1.0, 1.0], [0.35, 0.35, 0.1, 0.1, 0.1],
        #         1.2, 1.2, None)
        img_hr, img_lr = high, low
        img_hr, img_lr = img_hr.squeeze(0), img_lr.squeeze(0)
        return img_lr, img_hr
    
    def __len__(self):
        return min(len(self.hr_file_list),len(self.lr_file_list))
    
class TrainSetLoader(Dataset):
    def __init__(self, cfg):
        super(TrainSetLoader, self).__init__()

        self.hr_dataset_dir = cfg.data_dir +'/train/HR/DIV2K_train_HR/'
        self.lr_dataset_bicubic_dir = cfg.data_dir +f'/train/LR/DIV2K_train_LR_bicubic/'
        self.lr_dataset_mild_dir = cfg.data_dir +f'/train/LR/DIV2K_train_LR_mild/'
        self.lr_dataset_wild_dir = cfg.data_dir +f'/train/LR/DIV2K_train_LR_wild/'
        
        # self.kernel_path = np.load(cfg.kernel_path, allow_pickle=True)
        # self.hr_file_list = sorted(os.listdir(self.hr_dataset_dir))
        self.lr_file_list = [os.path.join(self.lr_dataset_bicubic_dir, file) for file in os.listdir(self.lr_dataset_bicubic_dir)]+[os.path.join(self.lr_dataset_mild_dir, file) for file in os.listdir(self.lr_dataset_mild_dir)]+[os.path.join(self.lr_dataset_wild_dir, file) for file in os.listdir(self.lr_dataset_wild_dir)]
        
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.outputSize = cfg.img_size
        
    def __getitem__(self, index):
        # hr_path = self.hr_dataset_dir + self.hr_file_list[index]
        lr_path = self.lr_file_list[index]
        hr_path = os.path.join(self.hr_dataset_dir,lr_path.split("/")[-1][:4] + '.png')
        # raw_lr, raw_hr = np.load(lr_path), np.load(hr_path)
        rgb_hr = cv2.imread(hr_path)
        rgb_lr = cv2.imread(lr_path)
        
        hr_rgb_img = np.array(cv2.cvtColor(rgb_hr, cv2.COLOR_BGR2RGB)) 
        lr_rgb_img = np.array(cv2.cvtColor(rgb_lr, cv2.COLOR_BGR2RGB))
        # TODO check valid 255 rgb range
        hr_rgb_img = (hr_rgb_img ).astype(np.float32)
        lr_rgb_img = (lr_rgb_img ).astype(np.float32)

        hr_rgb_final = torch.from_numpy(hr_rgb_img.copy()).permute(2, 0, 1).unsqueeze(0)
        lr_rgb_final = torch.from_numpy(lr_rgb_img.copy()).permute(2, 0, 1).unsqueeze(0)
        _,_,H,W = lr_rgb_final.shape
        
        # h = 100 eh = 128, h -> eh, ratio = eh/h  w = 300, w -> 300*ratio
        # lr_rgb_final = F.interpolate(lr_rgb_final,(max(self.outputSize,H),max(self.outputSize,W)),mode='bilinear',align_corners=False)
        # hr_rgb_final = F.interpolate(hr_rgb_final,(max(self.outputSize,4*H),max(self.outputSize,4*W)),mode='bilinear',align_corners=False)
        
        spatial_arr = [H,W]
        if min(H,W) < self.outputSize:
            min_index = spatial_arr.index(min(spatial_arr))
            ratio = self.outputSize / spatial_arr[min_index]
            lr_rgb_final = F.interpolate(lr_rgb_final,(math.ceil(H*ratio),math.ceil(W*ratio)),mode='bilinear',align_corners=False)
            hr_rgb_final = F.interpolate(hr_rgb_final,(math.ceil(4*H*ratio),math.ceil(4*W*ratio)),mode='bilinear',align_corners=False)
        
        
        lr_rgb_final = lr_rgb_final / 255
        hr_rgb_final = hr_rgb_final / 255
        i,j,h,w = transforms.RandomCrop.get_params(lr_rgb_final, output_size = (self.outputSize,self.outputSize))
        low = TF.crop(lr_rgb_final, i, j, h, w)
        high = TF.crop(hr_rgb_final, i*4, j*4, 4*h, 4*w)
        # img_hr, img_lr = augmentation.apply_augment(high, low, ["rgb", "mixup", "vertical", "horizontal", "none"], [1.0, 1.0, 1.0, 1.0, 1.0], [0.35, 0.35, 0.1, 0.1, 0.1],
        #         1.2, 1.2, None)
        img_hr, img_lr = high, low
        img_hr, img_lr = img_hr.squeeze(0), img_lr.squeeze(0)
        return  img_lr, img_hr

    def __len__(self):
        return len(self.lr_file_list)

class ValidSetLoader(Dataset):
    def __init__(self, cfg_test):
        super(ValidSetLoader, self).__init__()

        self.hr_dataset_dir = cfg_test.data_dir +'/val/HR/'
        self.lr_dataset_bicubic_dir = cfg_test.data_dir +f'/val/LR/DIV2K_valid_LR_bicubic/'
        self.lr_dataset_mild_dir = cfg_test.data_dir +f'/val/LR/DIV2K_valid_LR_mild/'
        self.lr_dataset_wild_dir = cfg_test.data_dir +f'/val/LR/DIV2K_valid_LR_wild/'
        
        # self.kernel_path = np.load(cfg.kernel_path, allow_pickle=True)
        # self.hr_file_list = sorted(os.listdir(self.hr_dataset_dir))
        self.lr_file_list = [os.path.join(self.lr_dataset_bicubic_dir, file) for file in os.listdir(self.lr_dataset_bicubic_dir)]+[os.path.join(self.lr_dataset_mild_dir, file) for file in os.listdir(self.lr_dataset_mild_dir)]+[os.path.join(self.lr_dataset_wild_dir, file) for file in os.listdir(self.lr_dataset_wild_dir)]
        
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.outputSize = cfg_test.img_size
        
        
    def __getitem__(self, index):
        # hr_path = self.hr_dataset_dir + self.hr_file_list[index]
        lr_path = self.lr_file_list[index]
        hr_path = os.path.join(self.hr_dataset_dir,lr_path.split("/")[-1][:4] + '.png')
        # raw_lr, raw_hr = np.load(lr_path), np.load(hr_path)
        rgb_hr = cv2.imread(hr_path)
        rgb_lr = cv2.imread(lr_path)
        
        hr_rgb_img = np.array(cv2.cvtColor(rgb_hr, cv2.COLOR_BGR2RGB)) 
        lr_rgb_img = np.array(cv2.cvtColor(rgb_lr, cv2.COLOR_BGR2RGB))
        # TODO check valid 255 rgb range
        hr_rgb_img = (hr_rgb_img).astype(np.float32)
        lr_rgb_img = (lr_rgb_img).astype(np.float32)  
        hr_rgb_final = torch.from_numpy(hr_rgb_img.copy()).permute(2, 0, 1).unsqueeze(0)
        lr_rgb_final = torch.from_numpy(lr_rgb_img.copy()).permute(2, 0, 1).unsqueeze(0)
        _,_,H,W = lr_rgb_final.shape
        
        spatial_arr = [H,W]
        if min(H,W) < self.outputSize:
            min_index = spatial_arr.index(min(spatial_arr))
            ratio = self.outputSize / spatial_arr[min_index]
            lr_rgb_final = F.interpolate(lr_rgb_final,(math.ceil(H*ratio),math.ceil(W*ratio)),mode='bilinear',align_corners=False)
            hr_rgb_final = F.interpolate(hr_rgb_final,(math.ceil(4*H*ratio),math.ceil(4*W*ratio)),mode='bilinear',align_corners=False)
        
        lr_rgb_final = lr_rgb_final / 255
        hr_rgb_final = hr_rgb_final / 255
        
        i,j,h,w = transforms.RandomCrop.get_params(lr_rgb_final, output_size = (self.outputSize,self.outputSize))
        low = TF.crop(lr_rgb_final, i, j, h, w)
        high = TF.crop(hr_rgb_final, i*4, j*4, 4*h, 4*w)
        # img_hr, img_lr = augmentation.apply_augment(high, low, ["rgb", "mixup", "vertical", "horizontal", "none"], [1.0, 1.0, 1.0, 1.0, 1.0], [0.35, 0.35, 0.1, 0.1, 0.1],
        #         1.2, 1.2, None)
        img_hr, img_lr = high, low
        img_hr, img_lr = img_hr.squeeze(0), img_lr.squeeze(0)
        return img_lr, img_hr
    
    def __len__(self):
        return len(self.lr_file_list)
    