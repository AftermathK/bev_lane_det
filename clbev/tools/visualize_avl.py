import os
import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms, utils
import sys
sys.path.append('/home/dfpazr/Documents/CogRob/avl/DSM/network_estimation/temp_cl_bev_det/cl_bev_lane_det/clbev')
import argparse
from models.util.load_model import load_checkpoint, resume_training
from models.util.save_model import save_model_dp
from models.loss import IoULoss, NDPushPullLoss
from models.util.load_model import load_model
from models.util.cluster import embedding_post
from models.util.post_process import bev_instance2points_with_offset_z
from models.util.post_process import bev_instance2points
from models.util.post_process import mean_col_by_row 
from utils.config_util import load_config_module
from sklearn.metrics import f1_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time
from models.util.cluster import colors
from sklearn.cluster import DBSCAN

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
# model_path = '/home/dfpazr/Documents/CogRob/avl/DSM/network_estimation/bev_lane_det/breadcrumb_checkpoints/latest.pth'
model_path = '/home/dfpazr/Documents/CogRob/avl/DSM/network_estimation/semantics_clbev/cl_bev_lane_det/sampling_checkpoints/latest.pth'

''' parameter from config '''
config_file = './clbev/tools/breadcrumbs_config.py'
configs = load_config_module(config_file)
x_range = configs.x_range
y_range = configs.y_range
input_shape = configs.input_shape
meter_per_pixel = configs.meter_per_pixel
img_tf = A.Compose([
                A.Resize(height=input_shape[0], width=input_shape[1]),
                A.Normalize(),
                ToTensorV2()
                ])

# thresholds
post_conf = 0.1 # Minimum confidence on the BEV segmentation map for clustering
post_img_conf = -1.5 # Minimum confidence on the image segmentation map for clustering 
post_emb_margin = 6.0 # embeding margin of different clusters
img_emb_margin = 5.5 # embeding margin of different clusters
post_min_cluster_size = 15 # The minimum number of points in a cluster


def run_inference(config_file, image_path, output_path, gpu_id):
    print('use gpu ids is '+','.join([str(i) for i in gpu_id]))
    configs = load_config_module(config_file)

    ''' models and optimizer '''
    model = configs.model()
    # model = Eval_Model(model)
    model = load_model(model,
                       model_path)
    if torch.cuda.is_available():
        model = model.cuda()
    model = torch.nn.DataParallel(model)
    model.eval()

    idx = 0
    output_img_path = os.path.join(output_path, "img")
    output_bev_path = os.path.join(output_path, "bev")
    if not os.path.exists(output_img_path):
        os.mkdir(output_img_path)
    if not os.path.exists(output_bev_path):
        os.mkdir(output_bev_path)
    

    for infile in sorted(glob.glob(image_path + '*')):
        if 'png' in infile:
            # read image
            image = cv2.imread(infile)
            orig_img = np.copy(image)
            input_tf = img_tf(image=image)['image'].unsqueeze(0).cuda()
            t_start = time.time()
            pred = model(input_tf)
            t_fp = time.time()
            bev_pred, img_pred = pred[0], pred[1]
            
            idx = infile.split('/')[-1].split('.')[-2]
            bev_seg = bev_pred[0].detach().cpu().numpy()
            bev_embedding = bev_pred[1].detach().cpu().numpy()
            bev_offset_y = torch.sigmoid(bev_pred[2]).detach().cpu().numpy()
            bev_z_pred = bev_pred[3].detach().cpu().numpy()

            img_seg = img_pred[0].detach().cpu().numpy()
            img_embedding = img_pred[1].detach().cpu().numpy()

            img_seg_thresh = (img_seg.squeeze() > post_img_conf) 
            seg_idx = np.where(img_seg_thresh > 0)
            orig_img = cv2.resize(orig_img, (input_shape[1],input_shape[0]), interpolation=cv2.INTER_NEAREST)


            
            bev_prediction = (bev_seg, bev_embedding) 
            img_prediction = (img_seg, img_embedding) 

            bev_canvas, ids = embedding_post(bev_prediction, post_conf, emb_margin=post_emb_margin, min_cluster_size=post_min_cluster_size, canvas_color=False)
            offset_y = bev_offset_y[0][0]
            z_pred = bev_z_pred[0][0]
            # bev_lines = bev_instance2points_with_offset_z(bev_canvas, max_x=x_range[1],
            #                         meter_per_pixal=(meter_per_pixel, meter_per_pixel),offset_y=offset_y,Z=z_pred)         

            # img_canvas, img_ids = embedding_post(img_prediction, post_img_conf, emb_margin=post_emb_margin, min_cluster_size=post_min_cluster_size, canvas_color=True)
            img_canvas, img_ids = embedding_post(img_prediction, post_img_conf, emb_margin=img_emb_margin, min_cluster_size=post_min_cluster_size, canvas_color=False)
            img_cls = mean_col_by_row(img_canvas)
            # img_cls, splines = bev_instance2points(img_canvas)

            t_process = time.time()
            print("t_fp: {} | fps: {}".format(t_fp - t_start, 1/(t_fp - t_start)))
            print("t_pp: {} | fps: {}".format(t_process - t_fp, 1/(t_process - t_fp)))
            print("t_tt: {} | fps: {}".format(t_process - t_start, 1/(t_process - t_start)))

        
            x = seg_idx[1]
            y = seg_idx[0]
            X =  np.stack((x,y))

            # ------- DBSCAN
            # clustering = DBSCAN(eps=6, min_samples=2).fit(X.T)
            # cluster_idx = clustering.labels_

            # for i in range(len(seg_idx[0])):
            #     color_i = colors[cluster_idx[i]]
            #     orig_img = cv2.circle(orig_img, (4*seg_idx[1][i], 4*seg_idx[0][i]), 2, color_i, -1)

            # ------- OG
            # for i in range(len(seg_idx[0])):
            #     orig_img = cv2.circle(orig_img, (4*seg_idx[1][i], 4*seg_idx[0][i]), 2, (0, 0, 255), -1)

            # -------  requires canvas color to true && and no bev_instance2points()
            # for i in range(len(img_ids)):
            #     color_i = img_canvas[img_ids[i][0], img_ids[i][1]]
            #     orig_img = cv2.circle(orig_img, (4*img_ids[i][1], 4*img_ids[i][0]), 2, color_i.tolist(), -1)

            # -------
            # ok
            for cl in img_cls:
                for i in range(len(cl[0])):
                    # color_i = img_canvas[img_ids[i][0], img_ids[i][1]]
                    color_i = colors[cl[2]] 
                    orig_img = cv2.circle(orig_img, (int(cl[0][i]), int(cl[1][i])), 4, color_i, -1)

            cv2.imwrite(os.path.join(output_img_path, str(idx) + ".png"), orig_img)
            # plt.imshow(bev_canvas)
            # plt.savefig(os.path.join(output_bev_path, str(idx) + ".png"))


if __name__ == "__main__":

    indx = 0
    image_path = "/home/dfpazr/Desktop/livox-pcd/1/"
    output_path = "/home/dfpazr/Desktop/livox-pcd/1-pred-random-sampling/"
    gpu_id=[0,1]


    run_inference(config_file, image_path, output_path, gpu_id)

    #for i, data in enumerate(loader):
    
    #    inputs, class_ids, kp_2d, kp_3d, camera_params, image_paths = data
    #    inputs = inputs.to(device)
    #    camera_params = [[c_p[0].to(device), c_p[1], c_p[2], c_p[3]] for c_p in camera_params]
    #    _, confidence_gt, _, _, _ = loss_m.gen_groundtruth_tensors(class_ids, kp_2d, kp_3d)
    #    y, attentions = model(inputs, camera_params, confidence_gt) # use confidence_gt only in training
    #    indx = visualize_keypoint_predictions(indx, image_paths, class_ids, kp_2d, y)
