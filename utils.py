from torchvision.transforms import functional as tF
import random
import torch
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target=None):
        for i in range(image.size(0)):
            if random.random() < self.flip_prob:
                image[i] = tF.hflip(image[i])
                if target is not None:
                    target[i] = tF.hflip(target[i])
        return image, target
class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target=None):
        for i in range(image.size(0)):
            if random.random() < self.flip_prob:
                image[i] = tF.vflip(image[i])
                if target is not None:
                    target[i] = tF.vflip(target[i])
        return image, target
class RandomRotation(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target=None):
        for i in range(image.size(0)):
            random_angle = random.randint(0, self.flip_prob)
            image[i] = tF.rotate(image[i],random_angle)
            if target is not None:
                target[i] = tF.rotate(target[i],random_angle)
        return image, target

class Compose(object):
    def __init__(self):
        self.transforms = [
                    RandomVerticalFlip(0.5),
                    RandomHorizontalFlip(0.5),
                    RandomRotation(20)
                ]

    def __call__(self, image, mask=None):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image,mask
class random_transform(object):
    def __init__(self):
        self.transform=Compose()
        self.resize_512=transforms.Resize((512,512))
        self.resize_256=transforms.Resize((256,256))
    def __call__(self, image, mask):
        image=self.resize_512(image)
        mask=self.resize_512(mask)
        image, mask = self.transform(image, mask)
        mask2 = (mask > 0.5).int()
        x = torch.nonzero(mask2)
        x1, x2 = x[:, 2].min(), x[:, 2].max()
        y1, y2 = x[:, 3].min(), x[:, 3].max()
        crop_x1 = random.randint(0, x1)
        crop_y1 = random.randint(0, y1)
        crop_x2 = random.randint(x2, image.size(-1))
        crop_y2 = random.randint(y2, image.size(-1))
        image = self.resize_256(image[:, :, crop_x1:crop_x2, crop_y1:crop_y2])
        mask = self.resize_256(mask[:, :, crop_x1:crop_x2, crop_y1:crop_y2])
        return image,mask.float()


class Erosion2d(nn.Module):

    def __init__(self, m=1):
        super(Erosion2d, self).__init__()
        self.m = m
        self.pad = [m, m, m, m]
        self.unfold = nn.Unfold(2 * m + 1, padding=0, stride=1)

    def forward(self, x):
        batch_size, c, h, w = x.shape
        x_pad = F.pad(x, pad=self.pad, mode='constant', value=1e9)
        channel = self.unfold(x_pad).view(batch_size, c, -1, h, w)
        result = torch.min(channel, dim=2)[0]
        return result


def erosion(x, m=1):
    b, c, h, w = x.shape
    x_pad = F.pad(x, pad=[m, m, m, m], mode='constant', value=1e9)
    channel = nn.functional.unfold(x_pad, 2 * m + 1, padding=0, stride=1).view(b, c, -1, h, w)
    result = torch.min(channel, dim=2)[0]
    return result

class Dilation2d(nn.Module):

    def __init__(self, m=1):
        super(Dilation2d, self).__init__()
        self.m = m
        self.pad = [m, m, m, m]
        self.unfold = nn.Unfold(2 * m + 1, padding=0, stride=1)

    def forward(self, x):
        batch_size, c, h, w = x.shape
        x_pad = F.pad(x, pad=self.pad, mode='constant', value=-1e9)
        channel = self.unfold(x_pad).view(batch_size, c, -1, h, w)
        result = torch.max(channel, dim=2)[0]
        return result


def dilation(x, m=1):
    b, c, h, w = x.shape
    x_pad = F.pad(x, pad=[m, m, m, m], mode='constant', value=-1e9)
    channel = nn.functional.unfold(x_pad, 2 * m + 1, padding=0, stride=1).view(b, c, -1, h, w)
    result = torch.max(channel, dim=2)[0]
    return result