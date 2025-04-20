import torch
from torchmetrics import ROC, AUROC, F1Score, AveragePrecision
import os
from torchvision.transforms import transforms
import numpy as np
import pandas as pd
from skimage import measure
from statistics import mean

from sklearn.metrics import auc
from sklearn import metrics

def metric(labels_list, predictions, anomaly_map_list, GT_list, config):
    labels_list = torch.tensor(labels_list)
    predictions = torch.tensor(predictions)
    pro = compute_pro(GT_list, anomaly_map_list, num_th = 200)
        
    resutls_embeddings = anomaly_map_list[0]
    
    for feature in anomaly_map_list[1:]:
        resutls_embeddings = torch.cat((resutls_embeddings, feature), 0)
    resutls_embeddings =  ((resutls_embeddings - resutls_embeddings.min())/ (resutls_embeddings.max() - resutls_embeddings.min())) 
    
    GT_embeddings = GT_list[0]
    
    for feature in GT_list[1:]:
        GT_embeddings = torch.cat((GT_embeddings, feature), 0)

    resutls_embeddings = resutls_embeddings.clone().detach().requires_grad_(False)
   
    GT_embeddings = GT_embeddings.clone().detach().requires_grad_(False)

 
    roc = ROC(task="binary")
    auroc = AUROC(task="binary")

    fpr, tpr, thresholds = roc(predictions, labels_list)
    auroc_score = auroc(predictions, labels_list)

    GT_embeddings = torch.flatten(GT_embeddings).type(torch.bool).cpu().detach()
    resutls_embeddings = torch.flatten(resutls_embeddings).cpu().detach()

    auroc_pixel = auroc(resutls_embeddings, GT_embeddings)
    thresholdOpt_index = torch.argmax(tpr - fpr)
    thresholdOpt = thresholds[thresholdOpt_index]

    f1 = F1Score(task="binary")
    ap = AveragePrecision(task="binary")
    ap_image = ap(predictions, labels_list)
    ap_pixel = ap(resutls_embeddings, GT_embeddings)
    
    predictions0_1 = (predictions > thresholdOpt).int()
    for i,(l,p) in enumerate(zip(labels_list, predictions0_1)):
        print('sample : ', i, ' prediction is: ',p.item() ,' label is: ',l.item() , 'prediction is : ', predictions[i].item() ,'\n' ) if l != p else None

    f1_score = f1(predictions0_1, labels_list)


    if config.metrics.image_level_AUROC:
        print(f'AUROC: {auroc_score}')
    if config.metrics.pixel_level_AUROC:
        print(f"AUROC pixel level: {auroc_pixel} ")
    if config.metrics.image_level_F1Score:
        print(f'F1SCORE: {f1_score}')
    if config.metrics.pro:
        print(f'PRO: {pro}')#
        print(f"AP-I:{ap_image}")
        print(f"AP-P:{ap_pixel}") 

    
    with open('readme.txt', 'a') as f:
        f.write(
            f"{config.data.category} \n")
        f.write(
            f"AUROC: {auroc_score}       |    auroc_pixel: {auroc_pixel}    |     F1SCORE: {f1_score}   |     PRO_AUROC: {pro}   \n")
    roc = roc.reset()
    auroc = auroc.reset()
    f1 = f1.reset()
    return thresholdOpt

#https://github.com/hq-deng/RD4AD/blob/main/test.py#L337
def compute_pro(masks, amaps, num_th = 200):
    resutls_embeddings = amaps[0]
    for feature in amaps[1:]:
        resutls_embeddings = torch.cat((resutls_embeddings, feature), 0)
    amaps =  ((resutls_embeddings - resutls_embeddings.min())/ (resutls_embeddings.max() - resutls_embeddings.min())) 
    amaps = amaps.squeeze(1)
    amaps = amaps.cpu().detach().numpy()
    gt_embeddings = masks[0]
    for feature in masks[1:]:
        gt_embeddings = torch.cat((gt_embeddings, feature), 0)
    masks = gt_embeddings.squeeze(1).cpu().detach().numpy()
    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks , binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
     

        df = pd.concat([df, pd.DataFrame({"pro": mean(pros), "fpr": fpr, "threshold": th}, index=[0])], ignore_index=True)
      

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc
