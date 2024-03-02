# AnomalyDiffusion: Few-Shot Anomaly Image Generation with Diffusion Model (AAAI 2024)


<!-- <br> -->
[Teng Hu<sup>1#</sup>](https://sjtuplayer.github.io/), [Jiangning Zhang<sup>2#</sup>](https://zhangzjn.github.io/),  [Ran Yi<sup>1*</sup>](https://yiranran.github.io/), [Yuzhen Du<sup>1</sup>](https://github.com/YuzhenD),  [Xu Chen<sup>2</sup>](https://scholar.google.com/citations?hl=zh-CN&user=1621dVIAAAAJ), [Liang Liu<sup>2</sup>](https://scholar.google.com/citations?hl=zh-CN&user=Kkg3IPMAAAAJ), [Yabiao Wang<sup>2</sup>](https://scholar.google.com/citations?hl=zh-CN&user=xiK4nFUAAAAJ), and [Chengjie Wang<sup>1,2</sup>](https://scholar.google.com/citations?hl=zh-CN&user=fqte5H4AAAAJ).
<!-- <br> -->

(#Equal contribution,*Corresponding author)

[<sup>1</sup>Shanghai Jiao Tong University](https://www.sjtu.edu.cn/), 
[<sup>2</sup>Youtu Lab, Tencent](https://open.youtu.qq.com/#/open)

[![arXiv](https://img.shields.io/badge/arXiv-2312.05767-b31b1b.svg)](https://arxiv.org/abs/2312.05767)

[Project Page](https://sjtuplayer.github.io/anomalydiffusion-page/)



## Todo (Latest update: 2024/02/22)
- [x] **Release the training code
- [x] **Release the UNet checkpoints for testing anomaly detection accuracy
- [x] **Release the data
- [ ] **Release checkpoints for anomalydiffusion.
- [ ] **Release the inference code



## Data and checkpoints

The generated anomaly data of MvTec can be downloaded from [Google Drive](https://drive.google.com/file/d/1yzsZdW_xS-v4GprE2KQmQ1EbIWyGyFcG/view?usp=sharing)
or [百度网盘](https://pan.baidu.com/s/12gKMfc64sy3JDx5FAR-ytQ) (提取码: 0306).
(Note that we have filtered out some data with poor generation effects. Therefore, some classes
have relatively fewer samples.)

You can download the checkpoints for the UNet models trained on the generated data from 
[Google Drive](https://drive.google.com/drive/folders/1kcOdfQrvWeJyliGTYJ4HXKU5ccfn7t96?usp=sharing)
or [百度网盘](https://pan.baidu.com/s/1Xoe__ODeq_YrVc9lA-7B_A) (提取码: 0306).

## Prepare


### Prepare the environment
```
Ubuntu
python 3.8
cuda==11.8
gcc==7.5.0
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

## Infernece

You can download the checkpoints for the UNet models trained on the generated data from 
[Google Drive](https://drive.google.com/drive/folders/1kcOdfQrvWeJyliGTYJ4HXKU5ccfn7t96?usp=sharing)
or [百度网盘](https://pan.baidu.com/s/1Xoe__ODeq_YrVc9lA-7B_A) (提取码: 0306). 

After Downloading the checkpoints, you can test the anomaly detection accuracy:

```
python test-unet.py --data_path $path_to_mvtec --checkpoint_path $path_to_ckpt --sample_name=all
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
