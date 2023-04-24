import os
import glob
import datetime
import argparse

import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms, utils
import cv2

import rospy
import time
import std_msgs.msg
from geometry_msgs.msg import Point
# import sensor_msgs.point_cloud2 as pcl2
# from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
br = CvBridge()
from clbev.models.util.load_model import load_model
from clbev.models.util.cluster import embedding_post
from clbev.models.util.post_process import bev_instance2points_with_offset_z
from clbev.utils.config_util import load_config_module

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms


''' parameter from config '''
model_path = '/home/dfpazr/Documents/CogRob/avl/DSM/network_estimation/bev_lane_det/breadcrumb_checkpoints/latest.pth'
config_file = './bev_lanedet/tools/breadcrumbs_config.py'
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
post_min_cluster_size = 15 # The minimum number of points in a cluster

configs = load_config_module(config_file)
model = configs.model()
model = load_model(model,
                   model_path)
if torch.cuda.is_available():
    model = model.cuda()
model = torch.nn.DataParallel(model)
model.eval()

def cam1_callback(msg):

    image = br.compressed_imgmsg_to_cv2(msg)
    orig_img = np.copy(image)
    input_tf = img_tf(image=image)['image'].unsqueeze(0).cuda()
    print("----")

    t_start = time.time()
    pred = model(input_tf)
    t_fp = time.time()
    bev_pred, img_pred = pred[0], pred[1]
    
    bev_seg = bev_pred[0].detach().cpu().numpy()
    bev_embedding = bev_pred[1].detach().cpu().numpy()
    bev_offset_y = torch.sigmoid(bev_pred[2]).detach().cpu().numpy()
    bev_z_pred = bev_pred[3].detach().cpu().numpy()

    img_seg = img_pred[0]
    img_embedding = img_pred[1]

    img_seg_thresh = (img_seg > post_img_conf).detach().cpu().numpy().squeeze() 
    seg_idx = np.where(img_seg_thresh > 0)
    orig_img = cv2.resize(orig_img, (input_shape[1],input_shape[0]), interpolation=cv2.INTER_NEAREST)

    prediction = (bev_seg, bev_embedding) 
    canvas, ids = embedding_post(prediction, post_conf, emb_margin=post_emb_margin, min_cluster_size=post_min_cluster_size, canvas_color=False)
    offset_y = bev_offset_y[0][0]
    z_pred = bev_z_pred[0][0]
    lines = bev_instance2points_with_offset_z(canvas, max_x=x_range[1],
                            meter_per_pixal=(meter_per_pixel, meter_per_pixel),offset_y=offset_y,Z=z_pred)         

    t_process = time.time()
    print("t_fp: {} | fps: {}".format(t_fp - t_start, 1/(t_fp - t_start)))
    print("t_pp: {} | fps: {}".format(t_process - t_fp, 1/(t_process - t_fp)))
    print("t_tt: {} | fps: {}".format(t_process - t_start, 1/(t_process - t_start)))

    for i in range(len(seg_idx[0])):
        orig_img = cv2.circle(orig_img, (4*seg_idx[1][i], 4*seg_idx[0][i]), 2, (0, 0, 255), -1)

    vis_msg = br.cv2_to_compressed_imgmsg(orig_img)
    vis_msg.header = msg.header
    cl1_pub.publish(vis_msg)
    # cv2.imwrite(os.path.join(output_img_path, str(idx) + ".png"), orig_img)
    # plt.imshow(canvas)
    # plt.savefig(os.path.join(output_bev_path, str(idx) + ".png"))

    return


if __name__ == "__main__":
    rospy.init_node("cldet_bld", anonymous=True)
    cam1_sub = rospy.Subscriber(
                        "/avt_cameras/camera1/image_rect_color/compressed", CompressedImage, queue_size=10, callback=cam1_callback)
    cl1_pub = rospy.Publisher(
                        "/cl_detections/camera1/compressed", CompressedImage, queue_size=10) 
    
    rospy.spin()