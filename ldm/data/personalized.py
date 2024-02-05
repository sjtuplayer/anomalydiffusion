import os
import numpy as np
import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import json
import random
from .mask_generate import generate_mask
import cv2
from utils import random_transform
import numpy as np
imagenet_templates_smallest = [
    'a photo of a {}',
]

imagenet_templates_small = [
    'a photo of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'a photo of a clean {}',
    'a photo of a dirty {}',
    'a dark photo of the {}',
    'a photo of my {}',
    'a photo of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'a photo of the {}',
    'a good photo of the {}',
    'a photo of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'a photo of the clean {}',
    'a rendition of a {}',
    'a photo of a nice {}',
    'a good photo of a {}',
    'a photo of the nice {}',
    'a photo of the small {}',
    'a photo of the weird {}',
    'a photo of the large {}',
    'a photo of a cool {}',
    'a photo of a small {}',
    'an illustration of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'an illustration of a clean {}',
    'an illustration of a dirty {}',
    'a dark photo of the {}',
    'an illustration of my {}',
    'an illustration of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'an illustration of the {}',
    'a good photo of the {}',
    'an illustration of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'an illustration of the clean {}',
    'a rendition of a {}',
    'an illustration of a nice {}',
    'a good photo of a {}',
    'an illustration of the nice {}',
    'an illustration of the small {}',
    'an illustration of the weird {}',
    'an illustration of the large {}',
    'an illustration of a cool {}',
    'an illustration of a small {}',
    'a depiction of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'a depiction of a clean {}',
    'a depiction of a dirty {}',
    'a dark photo of the {}',
    'a depiction of my {}',
    'a depiction of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'a depiction of the {}',
    'a good photo of the {}',
    'a depiction of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'a depiction of the clean {}',
    'a rendition of a {}',
    'a depiction of a nice {}',
    'a good photo of a {}',
    'a depiction of the nice {}',
    'a depiction of the small {}',
    'a depiction of the weird {}',
    'a depiction of the large {}',
    'a depiction of a cool {}',
    'a depiction of a small {}',
]

imagenet_dual_templates_small = [
    'a photo of a {} with {}',
    'a rendering of a {} with {}',
    'a cropped photo of the {} with {}',
    'the photo of a {} with {}',
    'a photo of a clean {} with {}',
    'a photo of a dirty {} with {}',
    'a dark photo of the {} with {}',
    'a photo of my {} with {}',
    'a photo of the cool {} with {}',
    'a close-up photo of a {} with {}',
    'a bright photo of the {} with {}',
    'a cropped photo of a {} with {}',
    'a photo of the {} with {}',
    'a good photo of the {} with {}',
    'a photo of one {} with {}',
    'a close-up photo of the {} with {}',
    'a rendition of the {} with {}',
    'a photo of the clean {} with {}',
    'a rendition of a {} with {}',
    'a photo of a nice {} with {}',
    'a good photo of a {} with {}',
    'a photo of the nice {} with {}',
    'a photo of the small {} with {}',
    'a photo of the weird {} with {}',
    'a photo of the large {} with {}',
    'a photo of a cool {} with {}',
    'a photo of a small {} with {}',
]

per_img_token_list = [
    'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת',
]
class Personalized_mvtec_encoder(Dataset):
    def __init__(self,
                 mvtec_path,
                 size=None,
                 repeats=1,
                 interpolation="bicubic",
                 flip_p=0.5,
                 set="train",
                 placeholder_token="*",
                 per_image_tokens=False,
                 center_crop=False,
                 mixing_prob=0.25,
                 coarse_class_text=None,
                 data_enhance=False,
                 random_mask=False
                 ):
        self.data_enhance=None
        if data_enhance:
            self.data_enhance=random_transform()
        self.data_root=mvtec_path
        sample_anomaly_pairs=[]
        with open(os.path.join('name-anomaly.txt'),'r') as f:
            data=f.read().split('\n')
            for i in data:
                sample_name,anomaly_name=i.split('+')
                sample_anomaly_pairs.append((sample_name,anomaly_name))
        self.data=[]
        if data_enhance:
            size=512
        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        cnt=0
        for sample_name,anomaly_name in sample_anomaly_pairs:
            print(sample_name,anomaly_name)
            img_path=os.path.join(self.data_root,sample_name,'test',anomaly_name)
            mask_path=os.path.join(self.data_root,sample_name,'ground_truth',anomaly_name)
            img_files=os.listdir(img_path)
            mask_files=os.listdir(mask_path)
            img_files.sort(key=lambda x:int(x[:3]))
            mask_files.sort(key=lambda x: int(x[:3]))
            img_files=[os.path.join(img_path,file_name) for file_name in img_files]
            mask_files=[os.path.join(mask_path,file_name) for file_name in mask_files]
            for idx in range(len(img_files)):
                if set=='train' and idx>len(img_files)//3:
                    break
                if set!='train':
                    if idx<len(img_files)//3:
                        continue
                    elif idx>len(img_files)//3+1:
                        break
                mask_filename = mask_files[idx]
                img_filename = img_files[idx]
                image = Image.open(img_filename)
                mask = Image.open(mask_filename).convert("L")
                if not image.mode == "RGB":
                    image = image.convert("RGB")
                img = np.array(image).astype(np.uint8)
                mas = np.array(mask).astype(np.float32)
                image = Image.fromarray(img)
                mask = Image.fromarray(mas)
                image = image.resize((size, size), resample=self.interpolation)
                mask = mask.resize((size, size), resample=self.interpolation)
                image = np.array(image).astype(np.uint8)
                mask = np.array(mask).astype(np.float32)
                image= (image / 127.5 - 1.0).astype(np.float32)
                mask = mask / 255.0
                mask[mask < 0.5] = 0
                mask[mask >= 0.5] = 1
                self.data.append((image,mask,sample_name+'+'+anomaly_name))

        self.num_images = len(self.data)
        self._length = self.num_images

        self.placeholder_token = placeholder_token

        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.mixing_prob = mixing_prob

        self.coarse_class_text = coarse_class_text

        if per_image_tokens:
            assert self.num_images < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == "train":
            self._length = self.num_images * repeats
        else:
            self._length = 4
        self.random_mask=random_mask



    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        idx=idx%self.num_images
        example = {}


        placeholder_string = self.placeholder_token
        if self.coarse_class_text:
            placeholder_string = f"{self.coarse_class_text} {placeholder_string}"

        if self.per_image_tokens and np.random.uniform() < self.mixing_prob:
            text = random.choice(imagenet_dual_templates_small).format(placeholder_string, per_img_token_list[i % self.num_images])
        else:
            text = random.choice(imagenet_templates_small).format(placeholder_string)
        image=self.data[idx][0]
        if self.random_mask:
            mask=generate_mask(256)
        else:
            mask=self.data[idx][1]
        example["caption"] = text
        example["image"] = image
        example["mask"] = mask
        example["name"]=self.data[idx][2]
        return example