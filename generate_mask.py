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
from ldm.data.personalized import Personalized_mvtec_mask
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
def check_mask(mask):
    w=mask.size(-1)
    if torch.count_nonzero(mask[:,0,:])>w/3:
        return False
    if torch.count_nonzero(mask[:, w-1, :])>w/3:
        return False
    if torch.count_nonzero(mask[:, :, 0])>w/3:
        return False
    if torch.count_nonzero(mask[:, :, w-1])>w/3:
        return False
    return True
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample_name",
        default='capsule',
        help="whether use ht encoder",
    )
    parser.add_argument(
        "--anomaly_name",
        default='crack',
        help="whether use ht encoder",
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="whether use ht encoder",
    )

    opt = parser.parse_args()
    config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-finetune.yaml")
    actual_resume = './models/ldm/text2img-large/model.ckpt'
    model = load_model_from_config(config, actual_resume)
    sample_name=opt.sample_name
    anomaly_name=opt.anomaly_name
    model.embedding_manager.load('logs/mask-checkpoints/%s-%s/checkpoints/embeddings.pt'%(sample_name,anomaly_name))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)
    cnt=0
    dataset = Personalized_mvtec_mask(opt.data_root, sample_name, anomaly_name,repeats=10000)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, drop_last=True)
    save_dir='generated_mask/%s/%s'%(sample_name,anomaly_name)
    os.makedirs(save_dir,exist_ok=True)
    with torch.no_grad():
        for i in range(1000):
            for idx, batch in enumerate(dataloader):
                if cnt>500:
                    exit()
                with model.ema_scope():
                    images=model.log_images(batch,sample=True,inpaint=False,unconditional_only=True,ddim_steps=100)
                    masks=images['samples_scaled']
                    for idx2,mask in enumerate(masks):
                        mask=mask.mean(0).unsqueeze(0)
                        mask=(mask>0.8).float()
                        flag=check_mask(mask)
                        if flag:
                            save_image(mask,os.path.join(save_dir,'%d.jpg'%cnt))
                            cnt+=1
