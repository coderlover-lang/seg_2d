from PIL import Image 
from matplotlib import pyplot as plt 
import matplotlib as mpl 


mpl.rcParams['font.sans-serif'] = ['Times New Roman']
mpl.rcParams['axes.unicode_minus'] = False

import sys 
sys.path.append('..')
from train import TrainModel 
import torch 
from custom_dataset.dataset import predict_by_name_processing
import numpy as np 
import cv2 
import os 

# figure_name = "34_18.png" # 图片名称
mode = "test"
target_files = []
source_files = []
start = 0 if mode == "train" else 300
end = 300 if mode == "train" else -1
for image_dir in ["/home/kathy/dataset/submit_a/G_image1/", "/home/kathy/dataset/submit_a/R_image1/"]:
    for file in os.listdir(image_dir)[start : end]:
        source_files.append(image_dir + file)
for image_dir in ["/home/kathy/dataset/submit_a/G_target1/", "/home/kathy/dataset/submit_a/R_target1/"]:
    for file in os.listdir(image_dir)[start : end]:
        target_files.append(image_dir + file)
# 定义颜色
colors = [
    (0, 0, 0),    # 背景 - 黑色
    (0, 255, 0),  # 器官1 - 绿色
    (255, 0, 0),  # 器官2 - 蓝色
    (0, 0, 255),   # 器官3 - 红色
    (255, 255, 0)   # 器官4 - cyan
]
device = torch.device("cuda:0")
model = TrainModel.load_from_checkpoint(
    "/home/kathy/projects/project_guo/test/best_model/version_7/checkpoints/epoch=499-step=30500.ckpt", 
    map_location=device
)



## Figure background 
img = plt.imread(source_files[0])
img = img[:256, :256]
background = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


img_model, masks_model = predict_by_name_processing(figure_name)
img_model, masks_model = img_model.to(model.device), masks_model.to(model.device)
output = model(img_model)
pred = output.argmax(dim = 1)


pred_mask = torch.concat(
    (
        torch.concat((pred[0, ...], pred[3, ...])), 
        torch.concat((pred[1, ...], pred[2, ...])), 
    ) , dim = 1
) 

true_mask = torch.concat(
    (
        torch.concat((pred[0, ...], pred[3, ...])), 
        torch.concat((pred[1, ...], pred[2, ...])), 
    ) , dim = 1
) 




# Figure foreground


for n, (mask_type_name, mask) in enumerate(zip(["true", "pred"], [true_mask, pred_mask])):
    fig , axes = plt.subplots(1, 1)
    fig.set_size_inches(6, 6)
    overlay = np.ones(shape = (256, 256, 3))
    mask = mask.detach().cpu().numpy()

    for i in range(1, 5):  # 假设类别从1到4
        overlay[mask == i] = colors[i]

    result = cv2.addWeighted(background, 0.8, overlay.astype("float32"), 0.2, 0)
    # print(result)
    # matrix = (result>1)
    # print(result[matrix])
    # # plt.imsave(f"./output/{mask_type_name}_{figure_name}",result)
    axes.imshow(result)
    axes.set_xticks([])
    axes.set_yticks([])
    # axes.flatten()[n].set_title(mask_type_name)
    # axes.flatten()[i].([])
    plt.savefig(f"./output/{mask_type_name}_{figure_name}", transparent=True, bbox_inches='tight')