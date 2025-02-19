import torchvision.transforms.functional as TF
import os
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
from utils import natsorted


class Fusion_dataset(Dataset):
    def __init__(self, split, ir_path=None, vi_path=None):
        super(Fusion_dataset, self).__init__()
        assert split in ['MSRS', 'val', 'MSRS'], 'split must be "MSRS"|"val"|"MSRS"'

        if split == 'MSRS':
            self.vis_dir = 'MSRS/Visible/MSRS/MSRS/'
            self.ir_dir = 'MSRS/Infrared/train/MSRS/'
            self.label_dir = 'MSRS/Label/aa/MSRS/'
            self.filelist = natsorted(os.listdir(self.vis_dir))
            self.split = split
            self.length = min(len(self.filelist), len(self.filelist))

        elif split == 'val' or split == 'MSRS':
            self.vis_dir = vi_path
            self.ir_dir = ir_path
            self.filelist = natsorted(os.listdir(self.vis_dir))
            self.split = split
            self.length = min(len(self.filelist), len(self.filelist))

    def __getitem__(self, index):
        img_name = self.filelist[index]
        vis_path = os.path.join(self.vis_dir, img_name)
        ir_path = os.path.join(self.ir_dir, img_name)          
        img_vis = self.imread(path=vis_path)
        img_ir = self.imread(path=ir_path, vis_flage=False)            
        if self.split=='MSRS':
            label_path = os.path.join(self.label_dir, img_name)  
            label = self.imread(path=label_path, label=True)
            label = label.type(torch.LongTensor)   
                  
        if self.split=='MSRS':
            return img_vis, img_ir, label, img_name
        else:
            return img_vis, img_ir, img_name

    def __len__(self):
        return self.length
    
    @staticmethod
    def imread(path, label=False, vis_flage=True):
        if label:
            img = Image.open(path)
            im_ts = TF.to_tensor(img) * 255
        else:
            if vis_flage: ## visible images; RGB channel
                img = Image.open(path).convert('RGB')
                im_ts = TF.to_tensor(img)
            else: ## infrared images single channel 
                img = Image.open(path).convert('L') 
                im_ts = TF.to_tensor(img)
        return im_ts



class Fusion_dataset2(Dataset):
    def __init__(self, split, ir_path=None, vi_path=None, mask_path=None):
        super(Fusion_dataset2, self).__init__()
        assert split in ['RoadScene','MSRS', 'val', 'test', 'M3FD'], 'split must be "MSRS"|"val"|"test"'

        if split == 'MSRS':
            self.vis_dir = 'MSRS/Visible/train/'
            self.ir_dir = 'MSRS/Infrared/train/'
            self.label_dir = 'MSRS/Label/train/'
            self.filelist = natsorted(os.listdir(self.vis_dir))
            self.split = split
            self.length = min(len(self.filelist), len(self.filelist))

        elif split == 'RoadScene':
            self.vis_dir = 'RoadScene/Visible/train/'
            self.ir_dir = 'RoadScene/Infrared/train/'
            self.filelist = natsorted(os.listdir(self.vis_dir))
            self.split = split
            self.length = min(len(self.filelist), len(self.filelist))

        elif split == 'M3FD':
            self.vis_dir = 'M3FD/Visible/train/'
            self.ir_dir = 'M3FD/Infrared/train/'
            self.filelist = natsorted(os.listdir(self.vis_dir))
            self.split = split
            self.length = min(len(self.filelist), len(self.filelist))

        elif split == 'val' or split == 'test':
            self.vis_dir = vi_path
            self.mask_dir = mask_path
            self.ir_dir = ir_path
            self.filelist = natsorted(os.listdir(self.vis_dir))
            self.split = split
            self.length = min(len(self.filelist), len(self.filelist))


    def __getitem__(self, index):
        img_name = self.filelist[index]
        vis_path = os.path.join(self.vis_dir, img_name)
        ir_path = os.path.join(self.ir_dir, img_name)
        img_vis = self.imread(path=vis_path)
        img_ir = self.imread(path=ir_path, vis_flage=False)

        if self.split == 'M3FD':
            return img_vis, img_ir, img_name

        if self.split == 'MSRS':
            label_path = os.path.join(self.label_dir, img_name)
            label = self.imread(path=label_path, label=True)
            label = label.type(torch.LongTensor)
        else:
            mask_path = os.path.join(self.mask_dir, img_name)
            img_mask = self.imread(path=mask_path)

        if self.split == 'MSRS':
            return img_vis, img_ir, label, img_name
        else:
            return img_vis, img_ir, img_mask, img_name


    def __len__(self):
        return self.length

    @staticmethod
    def imread(path, label=False, vis_flage=True):
        if label:
            img = Image.open(path)
            im_ts = TF.to_tensor(img) * 255
        else:
            if vis_flage:  ## visible images; RGB channel
                img = Image.open(path).convert('RGB')
                im_ts = TF.to_tensor(img)
            else:  ## infrared images single channel
                img = Image.open(path).convert('L')
                im_ts = TF.to_tensor(img)
        return im_ts