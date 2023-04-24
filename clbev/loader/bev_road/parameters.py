#############################################################################################################
##
##  Parameters
##
#############################################################################################################
import numpy as np

class Parameters():
    n_epoch = 500
    l_rate = 0.00001
    weight_decay=1e-5
    batch_size = 6
    x_size = 512
    y_size = 256

    kp_x_size = 512
    kp_y_size = 256

    resize_ratio = 8
    grid_x = x_size//resize_ratio  #64
    grid_y = y_size//resize_ratio #32

    # center image crop + resize
    center_crop_min = 750
    center_crop_max = 1750

    
    feature_size = 4
    regression_size = 110
    mode = 3
    threshold_point = 0.05 #0.5 #0.57 #0.64 #0.35
    threshold_instance = 0.08
    use_lidar = True 
    predict_depth = True 

    #loss function parameter
    K1 = 1.0                     #  ####################################
    K2 = 2.0
    constant_offset = 0.2
    constant_exist = 1.0 #2.0#1.0    #8
    constant_nonexist = 1.0#3.0
    constant_angle = 1.0
    constant_similarity = 1.0
    constant_attention = 0.1
    constant_alpha = 0.5 #in SGPN paper, they increase this factor by 2 every 5 epochs
    constant_beta = 0.5
    constant_l = 1.0
    constant_lane_loss = 1.0  #10  ######################################
    constant_instance_loss = 1.0

    #data loader parameter
    flip_ratio=0.6
    translation_ratio=0.6
    rotate_ratio=0.6
    noise_ratio=0.6
    intensity_ratio=0.6
    shadow_ratio=0.6
    scaling_ratio=0.2

    #Need to keep an eye out for this
    flip_indices=[(0,34),(1,35),(2,36),(3,37),(4,38),(5,39),(6,40),(7,41),(8,42),(9,43),(10,44),(11,45),(12,46),(13,47),(14,48),(15,49),(16,50),(17,51)
                    ,(18,52),(19,53),(20,54),(21,55),(22,56),(23,57),(24,58),(25,59),(26,60),(27,61),(28,62),(29,63),(30,64),(31,65)
                    ,(32,66),(33,67),(68,68),(69,69),(70,72),(71,73)]

    #Data roots 
    sensor_root_url="/mnt/00B0A680755C4DFA/DevSpace/DSM/datasets/argoverse2/test"

    #.txt datapaths
    # labels_data_path = "/mnt/00B0A680755C4DFA/DevSpace/DSM/datasets/argoverse2/labels/sample-labels/val/04994d08-156c-3018-9717-ba0e29be8153"
    # labels_data_path = "/mnt/00B0A680755C4DFA/DevSpace/DSM/datasets/argoverse2/labels/sample-labels/val/0bae3b5e-417d-3b03-abaa-806b433233b8"
    labels_data_path = "/mnt/00B0A680755C4DFA/DevSpace/DSM/datasets/argoverse2/labels/labels-v3/test"

    # test weights
    weights_path = ""
    

    # Cutoff coefficients
    # test parameter
    color = [(0,0,0), (255,0,0), (0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(255,255,255),(100,255,0),(100,0,255),(255,100,0),(0,100,255),(255,0,100),(0,255,100)]
    grid_location = np.zeros((grid_y, grid_x, 2))
    for y in range(grid_y):
        for x in range(grid_x):
            grid_location[y][x][0] = x
            grid_location[y][x][1] = y
    num_iter = 30
    threshold_RANSAC = 0.1
    ratio_inliers = 0.1