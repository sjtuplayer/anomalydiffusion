
import torch
from torch import optim
from unet_utils.tensorboard_visualizer import TensorboardVisualizer
from unet_utils.loss import FocalLoss, SSIM
import os
from unet_utils.data_loader import MVTec_Anomaly_Detection,MVTecDRAEMTestDataset_partial
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from unet_utils.model_unet import DiscriminativeSubNetwork
import os
from unet_utils.au_pro_util import calculate_au_pro
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
def test(args,obj_name, model_seg):
    mvtec_path = args.mvtec_path
    obj_ap_pixel_list = []
    obj_auroc_pixel_list = []
    obj_ap_image_list = []
    obj_auroc_image_list = []
    img_dim = 256
    model_seg.eval()
    dataset = MVTecDRAEMTestDataset_partial(mvtec_path +'/'+ obj_name + "/test/", resize_shape=[img_dim, img_dim])
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0)

    total_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
    total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
    mask_cnt = 0

    anomaly_score_gt = []
    anomaly_score_prediction = []

    gt_masks=[]
    predicted_masks=[]

    for i_batch, sample_batched in enumerate(dataloader):

        gray_batch = sample_batched["image"].cuda()
        gray_batch=gray_batch[:,[2,1,0],:,:]

        is_normal = sample_batched["has_anomaly"].detach().numpy()[0 ,0]
        anomaly_score_gt.append(is_normal)
        true_mask = sample_batched["mask"]
        true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))
        out_mask = model_seg(gray_batch)
        out_mask_sm = torch.softmax(out_mask, dim=1)

        out_mask_cv = out_mask_sm[0 ,1 ,: ,:].detach().cpu().numpy()
        out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[: ,1: ,: ,:], 21, stride=1,
                                                           padding=21 // 2).cpu().detach().numpy()
        image_score = np.max(out_mask_averaged)
        anomaly_score_prediction.append(image_score)

        flat_true_mask = true_mask_cv.flatten()
        flat_out_mask = out_mask_cv.flatten()
        gt_masks.append(true_mask_cv.squeeze())
        predicted_masks.append(out_mask_cv.squeeze())

        total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask
        total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask
        mask_cnt += 1

    anomaly_score_prediction = np.array(anomaly_score_prediction)
    anomaly_score_gt = np.array(anomaly_score_gt)
    auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)
    ap = average_precision_score(anomaly_score_gt, anomaly_score_prediction)

    total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
    total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_cnt]
    total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]
    auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
    ap_pixel = average_precision_score(total_gt_pixel_scores, total_pixel_scores)
    pro_pixel, _ = calculate_au_pro(gt_masks, predicted_masks)
    obj_ap_pixel_list.append(ap_pixel)
    obj_auroc_pixel_list.append(auroc_pixel)
    obj_auroc_image_list.append(auroc)
    obj_ap_image_list.append(ap)
    print(obj_name)
    print("AUC Image:  " +str(auroc))
    print("AP Image:  " +str(ap))
    print("AUC Pixel:  " +str(auroc_pixel))
    #print("AUC Pixel:  " +str(auroc_pixel))
    print("AP Pixel:  " +str(ap_pixel))
    print('PRO Pixel:' +str(pro_pixel))
    print("==============================")
    return float(auroc),float(auroc_pixel),float(ap_pixel),float(pro_pixel)


def train_on_device(obj_names, args):

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    for obj_name in obj_names:

        run_name = obj_name

        model_seg = DiscriminativeSubNetwork(in_channels=3, out_channels=2)
        model_seg.cuda()
        model_seg.apply(weights_init)

        optimizer = torch.optim.Adam([
                                      {"params": model_seg.parameters(), "lr": args.lr}])

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[args.epochs*0.8,args.epochs*0.9],gamma=0.2, last_epoch=-1)

        loss_focal = FocalLoss()

        dataset = MVTec_Anomaly_Detection(args,obj_name,length=500)
        dataloader = DataLoader(dataset, batch_size=args.bs,
                                shuffle=True, num_workers=16)

        n_iter = 0
        last_sum=0
        for epoch in range(args.epochs):
            model_seg.train()
            print("Epoch: "+str(epoch))
            for i_batch, sample_batched in enumerate(dataloader):
                aug_gray_batch = sample_batched["image"].cuda()
                anomaly_mask = sample_batched["mask"].cuda()
                out_mask = model_seg(aug_gray_batch)
                out_mask_sm = torch.softmax(out_mask, dim=1)
                segment_loss = loss_focal(out_mask_sm, anomaly_mask)
                loss = segment_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                n_iter +=1
            scheduler.step()

            auroc,auroc_px,ap_px,pro_px=test(args,obj_name, model_seg)
            sum_metric=auroc+auroc_px+ap_px+pro_px
            if sum_metric>last_sum:
                torch.save(model_seg.state_dict(), os.path.join(args.save_path, run_name + ".pckl"))
                last_sum=sum_metric

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_name', type=str, default='all')
    parser.add_argument('--generated_data_path', action='store', type=str, required=True)
    parser.add_argument('--save_path', default='checkpoints/localization', type=str)
    parser.add_argument('--mvtec_path', action='store', type=str, required=True)
    parser.add_argument('--bs', action='store', type=int,default=8, required=False)
    parser.add_argument('--lr', action='store', type=float,default=0.0001, required=False)
    parser.add_argument('--epochs', action='store', type=int,default=200, required=False)
    parser.add_argument('--gpu_id', action='store', type=int, default=0, required=False)
    parser.add_argument('--log_path', action='store', type=str,default='./logs/ ', required=False)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--test_separately', action='store_true',default=False)
    parser.add_argument('--reverse', action='store_true',default=False)
    parser.add_argument('--data_name',type=str, default='text_inversion')
    args = parser.parse_args()

    obj_batch =  [
                    'bottle',
                    'capsule',
                     'carpet',
                     'leather',
                     'pill',
                     'transistor',
                     'tile',
                     'cable',
                     'zipper',
                     'toothbrush',
                     'metal_nut',
                     'hazelnut',
                     'screw',
                     'grid',
                     'wood'
                     ]
    if args.reverse:
        obj_batch=reversed(obj_batch)
    if args.sample_name!='all':
        obj_list=[args.sample_name]
        picked_classes = obj_list
    else:
        picked_classes = obj_batch

    with torch.cuda.device(args.gpu_id):
        train_on_device(picked_classes, args)
#python train-unet.py --data_path $path_to_the_generated_data  --save_path ./ --mvtec_path=$path_to_mvtec --sample_name=capsule



