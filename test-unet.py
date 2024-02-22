import torch
from unet_utils.data_loader import MVTecDRAEMTestDataset_partial
from torch.utils.data import DataLoader
import numpy as np
from unet_utils.model_unet import  DiscriminativeSubNetwork
import os
from unet_utils.au_pro_util import calculate_au_pro
import csv
from sklearn.metrics import auc, roc_auc_score, average_precision_score, precision_recall_curve


def test(obj_names, mvtec_path, checkpoint_path):
    obj_ap_pixel_list = []
    obj_auroc_pixel_list = []
    obj_ap_image_list = []
    obj_auroc_image_list = []
    obj_pro_image_list = []
    from torchvision import transforms
    resize_224=transforms.Resize([224,224])
    crop_224=transforms.CenterCrop([224,224])
    with open("result.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Object', 'AUROC-I', 'AP-I', 'f1_max-I', 'AUROC-P', 'AP-P', 'f1_max-P', 'PRO-P'])
    for obj_name in obj_names:
        img_dim = 256
        run_name = obj_name
        model_seg = DiscriminativeSubNetwork(in_channels=3, out_channels=2)
        if not os.path.exists(os.path.join(checkpoint_path, run_name+".pckl")):
            print(os.path.join(checkpoint_path, run_name+".pckl"), 'not exists')
            continue
        model_seg.load_state_dict(torch.load(os.path.join(checkpoint_path, run_name+".pckl"), map_location='cuda:0'))
        model_seg.cuda()
        model_seg.eval()
        dataset = MVTecDRAEMTestDataset_partial(mvtec_path + obj_name + "/test/", resize_shape=[img_dim, img_dim])
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
            img_dim=out_mask.size(-1)
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

        precisions, recalls, thresholds = precision_recall_curve(anomaly_score_gt, anomaly_score_prediction)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls)
        best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
        best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])
        best_threshold = thresholds[best_f1_score_index]
        f1_max= best_f1_score

        total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
        total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_cnt]
        total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]
        auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
        ap_pixel = average_precision_score(total_gt_pixel_scores, total_pixel_scores)
        #print(total_gt_pixel_scores.shape,total_pixel_scores.shape)
        pro_pixel ,_ = calculate_au_pro(gt_masks, predicted_masks)

        score_l, score_h, score_step = 0.0, 1.0, 0.05
        gt = total_gt_pixel_scores.astype(np.bool_)
        max_f1_px=-99
        eps=1e-8
        pr_px_norm=total_pixel_scores
        for score in np.arange(score_l, score_h + 1e-3, score_step):
            pr = pr_px_norm > score
            total_area_intersect = np.logical_and(gt, pr).sum()
            total_area_union = np.logical_or(gt, pr).sum()
            total_area_pred_label = pr.sum()
            total_area_label = gt.sum()
            precision = total_area_intersect / (total_area_pred_label + eps)
            recall = total_area_intersect / (total_area_label + eps)
            f1_px = 2 * precision * recall / (
                     precision + recall + eps)
            max_f1_px=max(max_f1_px,f1_px)



        obj_ap_pixel_list.append(ap_pixel)
        obj_auroc_pixel_list.append(auroc_pixel)
        obj_auroc_image_list.append(auroc)
        obj_ap_image_list.append(ap)
        obj_pro_image_list.append(pro_pixel)
        print(obj_name)
        print("AUC Image:  " +str(auroc))
        print("AP Image:  " +str(ap))
        print("f1_max   "+str(f1_max))
        print("AUC Pixel:  " +str(auroc_pixel))
        print("AP Pixel:  " +str(ap_pixel))
        print("max_f1 Pixel:  " + str(ap_pixel))
        print("Pro Pixel:  " + str(pro_pixel))
        print("==============================")
        datas=[auroc,ap,f1_max,auroc_pixel,ap_pixel,max_f1_px,pro_pixel]
        for i in range(len(datas)):
            if datas[i]==1:
                datas[i]='100'
            else:
                datas[i]=str(round(100*datas[i],1))
        with open("result.csv", "a") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([obj_name,datas[0],datas[1], datas[2],datas[3], datas[4],datas[5],datas[6]])
    print("AUC Image mean:  " + str(np.mean(obj_auroc_image_list)))
    print("AP Image mean:  " + str(np.mean(obj_ap_image_list)))
    print("AUC Pixel mean:  " + str(np.mean(obj_auroc_pixel_list)))
    print("AP Pixel mean:  " + str(np.mean(obj_ap_pixel_list)))
    print("Pro Pixel mean:  " + str(np.mean(obj_pro_image_list)))


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id',default=0, type=int)
    parser.add_argument('--sample_name',type=str, required=True)
    parser.add_argument('--data_path', action='store', type=str, required=True)
    parser.add_argument('--checkpoint_path', action='store', type=str, required=True)

    args = parser.parse_args()
    if args.sample_name=='all':
        obj_list = ['capsule',
                    'bottle',
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
                    'grid',
                    'wood'
                    ]
    else:
        obj_list = [args.sample_name]


    with torch.cuda.device(args.gpu_id):
        test(obj_list,args.data_path, args.checkpoint_path)
#测试我们的增广方法训练的unet  ******************
#python test-unet.py --data_path /home/huteng/dataset/mvtec_anomaly_detection/ --checkpoint_path /home/huteng/anomaly_detection/DRAEM-main/checkpoints/github-opensourced/ --sample_name=all

