import os
import argparse
parser=argparse.ArgumentParser()
parser.add_argument(
        "--data_path",
        required=True,
        type=str,
        help="whether use ht encoder",
    )
parser.add_argument(
        "--adaptive_mask",
        action="store_true",
        help='whether use adaptive attention reweighting',
    )
parser.add_argument(
        "--gpu_id",
        type=int, default=0,
        help="whether use ht encoder",
    )
opt=parser.parse_args()
root_dir=opt.data_path
cnt=0
flag=False
# dirs1= ['bottle','cable','capsule','carpet','grid','hazelnut','leather','metal_nut','pill','screw','tile','transistor','toothbrush','wood','zipper']
dirs1=os.listdir(root_dir)
for dir1 in dirs1:
    if os.path.isdir(os.path.join(root_dir,dir1)):
        dirs=os.listdir(os.path.join(root_dir,dir1,'test'))
        for dir2 in dirs:
            if dir2!='good':
                os.system('CUDA_VISIBLE_DEVICES=%d python train_mask.py --mvtec_path=%s --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume ./models/ldm/text2img-large/model.ckpt  -n test --gpus 0, --init_word crack --sample_name=%s --anomaly_name=%s'%(opt.gpu_id,opt.data_path,dir1,dir2))
                os.system(
                    'CUDA_VISIBLE_DEVICES=%d python generate_mask.py --data_root=%s --sample_name=%s --anomaly_name=%s --data_root=%s'%(opt.gpu_id,opt.data_path,dir1,dir2,root_dir))
                if opt.adaptive_mask:
                    os.system(
                        'CUDA_VISIBLE_DEVICES=%d python generate_with_mask.py --data_root=%s --sample_name=%s --anomaly_name=%s --adaptive_mask'%(opt.gpu_id,opt.data_path,dir1,dir2))
                else:
                    os.system(
                        'CUDA_VISIBLE_DEVICES=%d python generate_with_mask.py --data_root=%s --sample_name=%s --anomaly_name=%s' % (
                        opt.gpu_id, opt.data_path, dir1, dir2))

#For generating texture anomaly (e.g., color in wood), set 'adaptive_mask' to be True to use Adaptive attention reweighting by '--adaptive_mask' in generate_with_mask.py
#For generating structual anomaly (e.g., squeeze in capsule), set 'adaptive_mask' to be False by removing '--adaptive_mask' in generate_with_mask.py