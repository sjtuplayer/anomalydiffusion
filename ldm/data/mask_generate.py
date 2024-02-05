import torch
import math
from torchvision.utils import save_image
import imgaug.augmenters as iaa
from torchvision import transforms
def rand_perlin_2d(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1])), dim=-1) % 1
    angles = 2 * math.pi * torch.rand(res[0] + 1, res[1] + 1)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0],
                                                                                                              0).repeat_interleave(
        d[1], 1)
    dot = lambda grad, shift: (
                torch.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                            dim=-1) * grad[:shape[0], :shape[1]]).sum(dim=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])

    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])
def generate_mask_batch(batch_size=1,size=256):
    resize=transforms.Resize((size,size))
    perlin_scale = 6
    min_perlin_scale = 0
    perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,))[0])
    perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,))[0])
    outs=[]
    for i in range(batch_size):
        perlin_noise = rand_perlin_2d((512,512), (perlin_scalex, perlin_scaley))
        threshold = 0.5
        perlin_thr = torch.where(perlin_noise > threshold, torch.ones_like(perlin_noise), torch.zeros_like(perlin_noise))
        outs.append(perlin_thr.unsqueeze(0))
    outs=torch.stack(outs,dim=0)
    outs = resize(outs)
    outs = (outs > 0.5).float()
    return outs
def generate_mask(size=256):
    resize=transforms.Resize((size,size))
    perlin_scale = 6
    min_perlin_scale = 0
    perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,))[0])
    perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,))[0])
    perlin_noise = rand_perlin_2d((size,size), (perlin_scalex, perlin_scaley))
    threshold = 0.5
    perlin_thr = torch.where(perlin_noise > threshold, torch.ones_like(perlin_noise), torch.zeros_like(perlin_noise)).unsqueeze(0)
    # perlin_thr=resize(perlin_thr)
    # perlin_thr = (perlin_thr > 0.5).float()
    return perlin_thr
if __name__=='__main__':
    x=generate_mask(256)
    print(x.shape)
    print(x.min(),x.max())
    save_image(x, 'tmp.jpg')