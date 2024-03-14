import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from ldm.util import instantiate_from_config

from ldm.models.diffusion.ddim import DDIMSampler
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from ldm.data.personalized import Positive_sample_with_generated_mask
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    config.model.params.ckpt_path = ckpt
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    return model

def make_batch(image, mask, device):
    image = np.array(Image.open(image).convert("RGB"))
    image = image.astype(np.float32)/255.0
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image).to(device)*2-1
    mask = np.array(Image.open(mask).convert("L"))
    mask = mask.astype(np.float32)/255.0
    mask = mask[None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask).to(device)
    print(image.shape, mask.shape)
    batch = {"image": image, "mask": mask,}
    return batch
def log_local( images,masked_img, cnt,sample_name,sample_name2,anomaly_name,ori_img=None,sub_dir=None):
    root='test-results/%s'%sub_dir
    for k in images:
        N = images[k].shape[0]
        images[k] = images[k][:N]
        if isinstance(images[k], torch.Tensor):
            images[k] = images[k].detach().cpu()
            images[k] = torch.clamp(images[k], -1., 1.)
    resize = transforms.Resize(images['samples_inpainting'].size(-1))
    for k in images:
        continue
        if k in ['samples_inpainting','mask']:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}-{}-{}-{:02}-2out-{}.jpg".format(sample_name,sample_name2,anomaly_name,cnt,k[:4])
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)
    #masked_img=resize(masked_img)
    # filename = "{}-{}-{}-{:02}-0mask.jpg".format(sample_name,sample_name2,anomaly_name,cnt)
    # path = os.path.join(root, filename)
    # save_image(masked_img,path,nrow=masked_img.size(0))
    if ori_img is not None:
        ori_img=torch.cat([ori_img,images['samples_inpainting']],dim=0)
        ori_img = resize(ori_img)
        filename = "{}-{}-{}-{:02}-1ori.jpg".format(sample_name, sample_name2, anomaly_name, cnt)
        path = os.path.join(root, filename)
        save_image((ori_img+1)/2, path, nrow=masked_img.size(0))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        required=True,
    )
    parser.add_argument(
        "--sample_name",
        default='capsule',
    )
    parser.add_argument(
        "--anomaly_name",
        default='crack',
    )
    parser.add_argument(
        "--adaptive_mask",
        action="store_true", default=True,
        help='whether use adaptive attention reweighting',
    )

    # setup_seed(42)
    opt = parser.parse_args()
    config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-finetune-encoder+embedding.yaml")
    actual_resume = './models/ldm/text2img-large/model.ckpt'
    model = load_model_from_config(config, actual_resume)
    sample_name=opt.sample_name
    anomaly_name=opt.anomaly_name
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)
    model.prepare_spatial_encoder(optimze_together=True)
    ckpt = torch.load('logs/anomaly-checkpoints/checkpoints/spatial_encoder.pt')
    model.embedding_manager.spatial_encoder_model.load_state_dict(ckpt)
    model.embedding_manager.load('logs/anomaly-checkpoints/checkpoints/embeddings.pt')
    dataset = Positive_sample_with_generated_mask(opt.data_root,sample_name, anomaly_name, repeats=1, size=256, set='train',
                                                      per_image_tokens=False)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, drop_last=True)
    save_dir = 'generated_dataset/%s/%s' % (sample_name, anomaly_name)
    os.makedirs(save_dir,exist_ok=True)
    os.makedirs(os.path.join(save_dir,'image'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'mask'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'image-mask'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'ori'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'recon'), exist_ok=True)
    cnt=0
    with torch.no_grad():
        for epoch in range(1000):
            for idx, batch in enumerate(dataloader):
                with model.ema_scope():
                    mask=batch['mask'].cpu()
                    ori_images=batch['image'].permute(0,3,1,2)
                    images=model.log_images(batch,sample=False,inpaint=True,unconditional_only=True,adaptive_mask=opt.adaptive_mask)
                    imgs=images['samples_inpainting'].cpu()
                    recon_image=images['reconstruction']
                    for i in range(len(imgs)):
                        save_image((imgs[i] + 1) / 2, os.path.join(save_dir, 'image', '%d.jpg' % cnt), normalize=False)
                        save_image((ori_images[i] + 1) / 2, os.path.join(save_dir, 'ori', '%d.jpg' % cnt),
                                   normalize=False)
                        save_image((recon_image[i]+1) / 2, os.path.join(save_dir, 'recon', '%d.jpg' % cnt),
                                   normalize=False)
                        save_image(mask[i], os.path.join(save_dir, 'mask','%d.jpg' % cnt))
                        save_image(torch.stack([(imgs[i]+1)/2,mask[i].repeat(3,1,1)],dim=0), os.path.join(save_dir, 'image-mask', '%d.jpg' % cnt))
                        cnt+=1

#python generate_with_mask.py --sample_name=screw --anomaly_name=thread_side --adaptive_mask
#python generate_with_mask.py --sample_name=wood --anomaly_name=color --adaptive_mask
