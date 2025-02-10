import torch 
import torch.nn as nn 
from medpy.metric import binary
from monai.metrics.utils import get_mask_edges, get_surface_distance
import numpy as np 
class LabelHd(nn.Module):
    pass 



def hd(pred,gt):
    #labelPred=sitk.GetImageFromArray(lP.astype(np.float32), isVector=False)
    #labelTrue=sitk.GetImageFromArray(lT.astype(np.float32), isVector=False)
    #hausdorffcomputer=sitk.HausdorffDistanceImageFilter()
    #hausdorffcomputer.Execute(labelTrue>0.5,labelPred>0.5)
    #return hausdorffcomputer.GetAverageHausdorffDistance()
    pred = pred.cpu().detach().numpy()
    gt   =   gt.cpu().detach().numpy()
    res = 0 
    for i in range(pred.shape[0]):
        res = res + compute_hausdorff_monai(pred[0], gt[0], max_dist=np.sqrt(pred.shape[-1] ** 2 + pred.shape[-2] ** 2))
    return res / pred.shape[0]
    # if pred.sum() > 0 and gt.sum()>0:
    #     hd95 = binary.hd95(pred, gt)
    #     return  hd95
    # else:
    #     return 0

def compute_hausdorff_monai(pred, gt, max_dist):
    if np.all(pred == gt):
        return 0.0
    (edges_pred, edges_gt) = get_mask_edges(pred, gt)
    surface_distance = get_surface_distance(edges_pred, edges_gt, distance_metric="euclidean")
    if surface_distance.shape == (0,):
        return 0.0
    dist = surface_distance.max()
    if dist > max_dist:
        return 1.0
    return dist / max_dist 

def hausdorff_distance(prediction, target):
    # Flatten prediction and target tensors
    prediction = prediction.contiguous().view(-1)
    target = target.contiguous().view(-1)

    # Find non-zero indices
    prediction_indices = np.nonzero(prediction).float()
    target_indices = np.nonzero(target).float()

    # Compute pairwise distances between non-zero indices
    distances_pred_to_target = np.cdist(prediction_indices, target_indices)
    distances_target_to_pred = np.cdist(target_indices, prediction_indices)

    # Compute Hausdorff distance
    hd_distance = np.max(np.max(np.min(distances_pred_to_target, axis=1)[0]), 
                            np.max(np.min(distances_target_to_pred, axis=1)[0]))

    return hd_distance.item()

if __name__ == "__main__":
    # pred = np.rand(32, 5, 128, 128)
    # pred   = torch.rand(size = (32, 5, 128, 128))
    gt   = torch.randint(0, 5, (32, 5, 128, 128))
    pred   = torch.randint(0, 5, (32, 5, 128, 128))
    # print(hausdorff_distance(pred, gt))
    image_shape = (8, 5, 128, 128)
    max_distance = np.sqrt(image_shape[-1] ** 2 + image_shape[-2] ** 2)

    pred = torch.randint(0, 2, size=image_shape)
    label = torch.randint(0,2, size=image_shape)
    # label = np.abs(np.random.random(size=image_shape))
    # label = np.random.randint( size=image_shape)
    print(hd(pred, label))
    