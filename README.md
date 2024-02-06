# AnomalyDiffusion: Few-Shot Anomaly Image Generation with Diffusion Model (AAAI 2024)


<!-- <br> -->
[Teng Hu<sup>1#</sup>](https://sjtuplayer.github.io/), [Jiangning Zhang<sup>2#</sup>](https://zhangzjn.github.io/),  [Ran Yi<sup>1*</sup>](https://yiranran.github.io/), [Yuzhen Du<sup>1</sup>](https://github.com/YuzhenD),  [Xu Chen<sup>2</sup>](https://scholar.google.com/citations?hl=zh-CN&user=1621dVIAAAAJ), [Liang Liu<sup>2</sup>](https://scholar.google.com/citations?hl=zh-CN&user=Kkg3IPMAAAAJ), [Yabiao Wang<sup>2</sup>](https://scholar.google.com/citations?hl=zh-CN&user=xiK4nFUAAAAJ), and [Chengjie Wang<sup>1,2</sup>](https://scholar.google.com/citations?hl=zh-CN&user=fqte5H4AAAAJ).
<!-- <br> -->

(#Equal contribution,*Corresponding author)

[<sup>1</sup>Shanghai Jiao Tong University](https://www.sjtu.edu.cn/), 
[<sup>2</sup>Youtu Lab, Tencent](https://open.youtu.qq.com/#/open)

[![arXiv](https://img.shields.io/badge/arXiv-2312.05767-b31b1b.svg)](https://arxiv.org/abs/2312.05767)

[Project Page](https://sjtuplayer.github.io/anomalydiffusion-page/)



## Todo (Latest update: 2024/02/05)
- [x] **Release the training code
- [ ] **Release the inference code
- [ ] **Release the data


## Prepare

### Prepare the environment
```
conda env create -f environment.yaml
conda activate Anomalydiffusion
```


### Checkpoint

Download the official checkpoint of the latent diffusion model:
```
mkdir -p models/ldm/text2img-large/
wget -O models/ldm/text2img-large/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt
```

## Model Training

Train the model by:

```
CUDA_VISIBLE_DEVICES=$gpu_id python main.py --spatial_encoder_embedding --data_enhance
 --base configs/latent-diffusion/txt2img-1p4B-finetune-encoder+embedding.yaml -t 
 --actual_resume models/ldm/text2img-large/model.ckpt -n test --gpus 0, 
  --init_word anomaly  --mvtec_path=$path_to_mvtec_dataset

```

## Citation

If you make use of our work, please cite our paper:

```
@inproceedings{hu2023anomalydiffusion,
  title={AnomalyDiffusion: Few-Shot Anomaly Image Generation with Diffusion Model},
  author={Hu, Teng and Zhang, Jiangning and Yi, Ran and Du, Yuzhen and Chen, Xu and Liu, Liang and Wang, Yabiao and Wang, Chengjie},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2024}
}
```
