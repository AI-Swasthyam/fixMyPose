import random
import numpy as np

import torch
from torch.utils import data
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms

from param import args
from speaker import Speaker

from data import DiffDataset, TorchDataset
from tok import Tokenizer
from utils import BufferLoader
import copy
from PIL import Image
import json
import random
import os
import numpy as np
import cv2
from speaker import Speaker

# Set the seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Image Transformation
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
img_transform = transforms.Compose([
    transforms.Resize((args.resize, args.resize)),
    transforms.ToTensor(),
    normalize
])


DATA_ROOT = "dataset/"


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def pil_saver(img, path):
    # print(img, path)
    with open(path, 'wb') as f:
        img.save(f)


class DiffDataset:
    def __init__(self, ds_name="fixmypose", split="train"):
        self.ds_name = ds_name
        self.split = split
        self.tok = Tokenizer()
        self.tok.load(os.path.join(DATA_ROOT, self.ds_name, "vocab_NC.txt"))


class TorchDataset(Dataset):
    def __init__(self, dataset, max_length=80, img0_transform=None, img1_transform=None):
        self.dataset = dataset
        self.name = dataset.ds_name+"_"+dataset.split
        self.tok = dataset.tok
        self.max_length = max_length
        self.img0_trans, self.img1_trans = img0_transform, img1_transform
        self.src_img_dir = os.path.join(DATA_ROOT, "src_img")
        self.trg_img_dir = os.path.join(DATA_ROOT, "trg_img")
        self.src_imgs = os.listdir(self.src_img_dir)

    def __len__(self):
        return len(os.listdir(self.src_img_dir))

    def __getitem__(self, index):
        img0 = Image.open(os.path.join(self.src_img_dir,self.src_imgs[index]))
        img1 = Image.open(os.path.join(self.trg_img_dir,self.src_imgs[index]))

        return {
            "src_img":self.img0_trans(img0),
            "trg_img":self.img1_trans(img1)
        }

args.workers = 1

# Loading Dataset


def get_tuple(ds_name, split, shuffle=True, drop_last=True):
    dataset = DiffDataset(ds_name, split)
    torch_ds = TorchDataset(dataset, max_length=args.max_input,
                            img0_transform=img_transform, img1_transform=img_transform
                            )

    print("The size of data split %s is %d" % (split, len(torch_ds)))
    loader = torch.utils.data.DataLoader(
        torch_ds,
        batch_size=1, shuffle=shuffle,
        num_workers=1, pin_memory=True,
        drop_last=drop_last)
    return dataset, torch_ds, loader

if __name__ == "__main__":
    dataset = DiffDataset(split="test")
    torch_ds = TorchDataset(dataset,img0_transform=img_transform,img1_transform=img_transform)
    test_tuple = get_tuple("fixmypose","test")

    speaker = Speaker(dataset)
    
    speaker.evaluate_val_useen(test_tuple,split="test")
    # cv2.imshow("img",np.array(temp["src_img"].permute(1,2,0).detach().cpu()))
    # cv2.waitKey(0)
    