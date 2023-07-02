import warnings
warnings.filterwarnings("ignore")
import numpy as np
import os
from torchvision import utils, transforms
from matplotlib import pyplot as plt
from pathlib import Path
import itertools

# AV2 requirements
import av2.geometry.interpolate as interp_utils
import av2.utils.depth_map_utils as depth_map_utils
import av2.rendering.video as video_utils
import av2.utils.io as io_utils
import av2.utils.raster as raster_utils
from av2.geometry.camera.pinhole_camera import PinholeCamera
from av2.geometry.camera.pinhole_camera import Intrinsics
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.datasets.sensor.constants import RingCameras
from av2.map.map_api import ArgoverseStaticMap
from av2.rendering.color import BLUE_BGR
from av2.rendering.map import EgoViewMapRenderer
from av2.utils.typing import NDArrayByte
from av2.map.lane_segment import LaneMarkType, LaneSegment
from av2.utils.typing import NDArrayBool, NDArrayByte, NDArrayFloat, NDArrayInt
from collections import namedtuple
from copy import deepcopy
import pickle

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for you approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

# labels = [
#     #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
#     Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
#     Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
#     Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
#     Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
#     Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
#     Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
#     Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
#     Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
#     Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
#     Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
#     Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
#     Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
#     Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
#     Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
#     Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
#     Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
#     Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
#     Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
#     Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
#     Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
#     Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
#     Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
#     Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
#     Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
#     Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
#     Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
#     Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
#     Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
#     Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
#     Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
# ]

labels = [
    Label(name='Bird', id=0, trainId=255, category='Building', categoryId=0, hasInstances=False, ignoreInEval=False, color=(165, 42, 42)), 
    Label(name='Ground_Animal', id=1, trainId=255, category='Building', categoryId=0, hasInstances=False, ignoreInEval=False, color=(0, 192, 0)), 
    Label(name='Curb', id=2, trainId=255, category='Building', categoryId=0, hasInstances=False, ignoreInEval=False, color=(196, 196, 196)), 
    Label(name='Fence', id=3, trainId=255, category='Building', categoryId=0, hasInstances=False, ignoreInEval=False, color=(190, 153, 153)), 
    Label(name='Guard_Rail', id=4, trainId=255, category='Building', categoryId=0, hasInstances=False, ignoreInEval=False, color=(180, 165, 180)), 
    Label(name='Barrier', id=5, trainId=255, category='Building', categoryId=0, hasInstances=False, ignoreInEval=False, color=(90, 120, 150)), 
    Label(name='Wall', id=6, trainId=255, category='Building', categoryId=0, hasInstances=False, ignoreInEval=False, color=(102, 102, 156)), 
    Label(name='Bike_Lane', id=7, trainId=255, category='Building', categoryId=0, hasInstances=False, ignoreInEval=False, color=(128, 64, 255)), 
    Label(name='Crosswalk_-_Plain', id=8, trainId=255, category='Building', categoryId=0, hasInstances=False, ignoreInEval=False, color=(140, 140, 200)), 
    Label(name='Curb_Cut', id=9, trainId=255, category='Building', categoryId=0, hasInstances=False, ignoreInEval=False, color=(170, 170, 170)), 
    Label(name='Parking', id=10, trainId=255, category='Building', categoryId=0, hasInstances=False, ignoreInEval=False, color=(250, 170, 160)), 
    Label(name='Pedestrian_Area', id=11, trainId=255, category='void', categoryId=0, hasInstances=False, ignoreInEval=False, color=(96, 96, 96)), 
    Label(name='Rail_Track', id=12, trainId=255, category='Building', categoryId=0, hasInstances=False, ignoreInEval=False, color=(230, 150, 140)), 
    Label(name='Road', id=13, trainId=255, category='Road', categoryId=0, hasInstances=False, ignoreInEval=False, color=(128, 64, 128)), 
    Label(name='Service_Lane', id=14, trainId=255, category='Building', categoryId=0, hasInstances=False, ignoreInEval=False, color=(110, 110, 110)), 
    Label(name='Sidewalk', id=15, trainId=255, category='Building', categoryId=0, hasInstances=False, ignoreInEval=False, color=(244, 35, 232)), 
    Label(name='Bridge', id=16, trainId=255, category='Building', categoryId=0, hasInstances=False, ignoreInEval=False, color=(150, 100, 100)), 
    Label(name='Building', id=17, trainId=255, category='Building', categoryId=0, hasInstances=False, ignoreInEval=False, color=(70, 70, 70)), 
    Label(name='Tunnel', id=18, trainId=255, category='Building', categoryId=0, hasInstances=False, ignoreInEval=False, color=(150, 120, 90)), 
    Label(name='Person', id=19, trainId=255, category='Vehicle', categoryId=0, hasInstances=False, ignoreInEval=False, color=(220, 20, 60)), 
    Label(name='Bicyclist', id=20, trainId=255, category='Vehicle', categoryId=0, hasInstances=False, ignoreInEval=False, color=(255, 0, 0)), 
    Label(name='Motorcyclist', id=21, trainId=255, category='Vehicle', categoryId=0, hasInstances=False, ignoreInEval=False, color=(255, 0, 100)), 
    Label(name='Other_Rider', id=22, trainId=255, category='Vehicle', categoryId=0, hasInstances=False, ignoreInEval=False, color=(255, 0, 200)), 
    Label(name='Lane_Marking_-_Crosswalk', id=23, trainId=255, category='void', categoryId=0, hasInstances=False, ignoreInEval=False, color=(200, 128, 128)), 
    Label(name='Lane_Marking_-_General', id=24, trainId=255, category='void', categoryId=0, hasInstances=False, ignoreInEval=False, color=(255, 255, 255)), 
    Label(name='Mountain', id=25, trainId=255, category='Building', categoryId=0, hasInstances=False, ignoreInEval=False, color=(64, 170, 64)), 
    Label(name='Sand', id=26, trainId=255, category='Building', categoryId=0, hasInstances=False, ignoreInEval=False, color=(230, 160, 50)), 
    Label(name='Sky', id=27, trainId=255, category='Building', categoryId=0, hasInstances=False, ignoreInEval=False, color=(70, 130, 180)), 
    Label(name='Snow', id=28, trainId=255, category='Building', categoryId=0, hasInstances=False, ignoreInEval=False, color=(190, 255, 255)), 
    Label(name='Terrain', id=29, trainId=255, category='Building', categoryId=0, hasInstances=False, ignoreInEval=False, color=(152, 251, 152)), 
    Label(name='Vegetation', id=30, trainId=255, category='Building', categoryId=0, hasInstances=False, ignoreInEval=False, color=(107, 142, 35)), 
    Label(name='Water', id=31, trainId=255, category='Building', categoryId=0, hasInstances=False, ignoreInEval=False, color=(0, 170, 30)), 
    Label(name='Banner', id=32, trainId=255, category='Building', categoryId=0, hasInstances=False, ignoreInEval=False, color=(255, 255, 128)), 
    Label(name='Bench', id=33, trainId=255, category='Building', categoryId=0, hasInstances=False, ignoreInEval=False, color=(250, 0, 30)), 
    Label(name='Bike_Rack', id=34, trainId=255, category='Building', categoryId=0, hasInstances=False, ignoreInEval=False, color=(100, 140, 180)), 
    Label(name='Billboard', id=35, trainId=255, category='Building', categoryId=0, hasInstances=False, ignoreInEval=False, color=(220, 220, 220)), 
    Label(name='Catch_Basin', id=36, trainId=255, category='Building', categoryId=0, hasInstances=False, ignoreInEval=False, color=(220, 128, 128)), 
    Label(name='CCTV_Camera', id=37, trainId=255, category='Building', categoryId=0, hasInstances=False, ignoreInEval=False, color=(222, 40, 40)), 
    Label(name='Fire_Hydrant', id=38, trainId=255, category='Building', categoryId=0, hasInstances=False, ignoreInEval=False, color=(100, 170, 30)), 
    Label(name='Junction_Box', id=39, trainId=255, category='Building', categoryId=0, hasInstances=False, ignoreInEval=False, color=(40, 40, 40)), 
    Label(name='Mailbox', id=40, trainId=255, category='Building', categoryId=0, hasInstances=False, ignoreInEval=False, color=(33, 33, 33)), 
    Label(name='Manhole', id=41, trainId=255, category='Road', categoryId=0, hasInstances=False, ignoreInEval=False, color=(100, 128, 160)), 
    Label(name='Phone_Booth', id=42, trainId=255, category='Building', categoryId=0, hasInstances=False, ignoreInEval=False, color=(142, 0, 0)), 
    Label(name='Pothole', id=43, trainId=255, category='Road', categoryId=0, hasInstances=False, ignoreInEval=False, color=(70, 100, 150)), 
    Label(name='Street_Light', id=44, trainId=255, category='Vehicle', categoryId=0, hasInstances=False, ignoreInEval=False, color=(210, 170, 100)), 
    Label(name='Pole', id=45, trainId=255, category='Vehicle', categoryId=0, hasInstances=False, ignoreInEval=False, color=(153, 153, 153)), 
    Label(name='Traffic_Sign_Frame', id=46, trainId=255, category='Vehicle', categoryId=0, hasInstances=False, ignoreInEval=False, color=(128, 128, 128)), 
    Label(name='Utility_Pole', id=47, trainId=255, category='Vehicle', categoryId=0, hasInstances=False, ignoreInEval=False, color=(0, 0, 80)), 
    Label(name='Traffic_Light', id=48, trainId=255, category='Vehicle', categoryId=0, hasInstances=False, ignoreInEval=False, color=(250, 170, 30)), 
    Label(name='Traffic_Sign_(Back)', id=49, trainId=255, category='Vehicle', categoryId=0, hasInstances=False, ignoreInEval=False, color=(192, 192, 192)), 
    Label(name='Traffic_Sign_(Front)', id=50, trainId=255, category='Vehicle', categoryId=0, hasInstances=False, ignoreInEval=False, color=(220, 220, 0)), 
    Label(name='Trash_Can', id=51, trainId=255, category='Building', categoryId=0, hasInstances=False, ignoreInEval=False, color=(140, 140, 20)), 
    Label(name='Bicycle', id=52, trainId=255, category='Vehicle', categoryId=0, hasInstances=False, ignoreInEval=False, color=(119, 11, 32)), 
    Label(name='Boat', id=53, trainId=255, category='Vehicle', categoryId=0, hasInstances=False, ignoreInEval=False, color=(150, 0, 255)), 
    Label(name='Bus', id=54, trainId=255, category='Vehicle', categoryId=0, hasInstances=False, ignoreInEval=False, color=(0, 60, 100)), 
    Label(name='Car', id=55, trainId=255, category='Vehicle', categoryId=0, hasInstances=False, ignoreInEval=False, color=(0, 0, 142)), 
    Label(name='Caravan', id=56, trainId=255, category='Vehicle', categoryId=0, hasInstances=False, ignoreInEval=False, color=(0, 0, 90)), 
    Label(name='Motorcycle', id=57, trainId=255, category='Vehicle', categoryId=0, hasInstances=False, ignoreInEval=False, color=(0, 0, 230)), 
    Label(name='On_Rails', id=58, trainId=255, category='void', categoryId=0, hasInstances=False, ignoreInEval=False, color=(0, 80, 100)), 
    Label(name='Other_Vehicle', id=59, trainId=255, category='Vehicle', categoryId=0, hasInstances=False, ignoreInEval=False, color=(128, 64, 64)), 
    Label(name='Trailer', id=60, trainId=255, category='Vehicle', categoryId=0, hasInstances=False, ignoreInEval=False, color=(0, 0, 110)), 
    Label(name='Truck', id=61, trainId=255, category='Vehicle', categoryId=0, hasInstances=False, ignoreInEval=False, color=(0, 0, 70)), 
    Label(name='Wheeled_Slow', id=62, trainId=255, category='Vehicle', categoryId=0, hasInstances=False, ignoreInEval=False, color=(0, 0, 192)), 
    Label(name='Car_Mount', id=63, trainId=255, category='Vehicle', categoryId=0, hasInstances=False, ignoreInEval=False, color=(32, 32, 32)), 
    Label(name='Ego_Vehicle', id=64, trainId=255, category='Vehicle', categoryId=0, hasInstances=False, ignoreInEval=False, color=(120, 10, 10)), 
    Label(name='Unlabeled', id=65, trainId=255, category='Building', categoryId=0, hasInstances=False, ignoreInEval=False, color=(0, 0, 0))
]

def random_pick_generic(a, W):
    L = W*(len(a)//W)
    b = a[:L].reshape(-1,W)
    idx = np.random.randint(0,b.shape[1], len(b))
    out = b[np.arange(len(idx)), idx]
    if len(a[L:])>0:
        out = np.hstack([out, np.array([np.random.choice(a[L:])])])
    return out

import sys
import pdb
class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


class KeypointLoader():
    def __init__(self, labels_path, sensor_path, split, semantics_config, skip=10):
        self.dset_path = labels_path 
        self.sensor_path = sensor_path
        self.data_skip = skip
        # self.transforms = transforms
        self.split = split
        self.semantics_config = semantics_config
        # self.semantics_config["classes_to_ignore"] = ['construction', 'object', 'sky', 'nature', 'void']
        # self.semantics_config["taint_classes"] = ['ground', 'sidewalk', 'parking', 'rail track']
        # self.semantics_config["taint_categories"] = ['vehicle', 'human']
        # self.semantic_threshold = 0.7

        # self.train = "train/" if train else "val/" #Validation set's sensor folder should be reorganized
        #Create a list of dataset files
        self.full_dset_files = None # self.get_dataset_files(self.dset_path)
        print("Loading pickle")
        # pdb_temp = ForkedPdb()
        # pdb_temp.set_trace()
        with open('av2-cli-dataset.pkl', 'rb') as f:
            self.full_dset_files = pickle.load(f)
        print("Loaded Pickle")
        
        # new_dset_files = []
        # for path in self.dset_files:
        #     new_dset_files.append(path.replace("/media/Data/argoverse2/labels/labels-v5-semantics", "/cogrob-avl-dataset/argoverse2/sensor/labels/labels-v3"))
        
        # self.dset_files = new_dset_files
        
        print(self.full_dset_files[0])
        
        # with open('av2-cli-dataset.pkl', 'wb') as f:
        #     pickle.dump(self.dset_files, f)
        self.dset_files = random_pick_generic(np.array(self.full_dset_files), skip) # self.dset_files[::skip] #Adding skip to avoid consecuitve frame of data
        self.cam_params = {} # dictionary of camera parameters
        self.image_shape = (512, 256) #Sticking to PiNet

        # for extracting additional AV2 data
        # self.av_loader = AV2SensorDataLoader(data_dir=Path(sensor_path), labels_dir=Path(sensor_path))
        with open('av-loader.pkl', 'rb') as f:
            self.av_loader = pickle.load(f)

        print("Dataset size: {}".format(len(self.dset_files)))
    
    def reset_dset_files(self):
        self.dset_files = random_pick_generic(np.array(self.full_dset_files), self.data_skip)


    def __len__(self):
        return len(self.dset_files)

    def read_and_set_file(self, index):
        filename = self.dset_files[index]
        with open(filename, "r") as f:
            self.f_content = f.readlines()

        _, log_name, cam_name = self.get_intrinsic_req(index)

        # update camera parameters dictionary
        if log_name not in self.cam_params.keys():
            pinhole_cam = self.av_loader.get_log_pinhole_camera(log_name, cam_name)
            self.cam_params[log_name] = {cam_name : pinhole_cam.intrinsics.K}

        else:
            # add log and camera if not present
            if cam_name not in self.cam_params[log_name].keys():
                pinhole_cam = self.av_loader.get_log_pinhole_camera(log_name, cam_name)
                self.cam_params[log_name][cam_name] = pinhole_cam.intrinsics.K

        
        return filename

    def filter_twos(self, keypoints):
        filtered_kp = [np.array([[lane[i], lane[i+1]] for i in range(0, len(lane), 3) if lane[i] != -2.0]) for lane in keypoints]
        kp_classes = [[lane[i] for i in range(2, len(lane), 3) if lane[i] != -2.0] for lane in keypoints]
        return filtered_kp, kp_classes #Returing the orignal unraveled tuple

    def get_keypoints_from_file(self):
        keypoints = []
        class_ids = []
        i = 4
        while True:
            if "cam_label" in self.f_content[i]:
                break
            line_split = self.f_content[i].split("[")
            line_kp = line_split[1][:-3] #-3 is to get rid of the last comma
            class_id = line_split[0].split(",")[4]
            kp_list = [float(i) for i in line_kp.split(",")]
            
            if int(class_id) != 0:
                i += 1
                continue

            # if len(kp_list) < 10 and int(class_id) != 4:
            #     i+=1
            #     continue
            keypoints.append(kp_list)
            class_ids.append(class_id)
            i+=1
        return keypoints, class_ids
        
        # keypoints_lines = f_content[]

    def get_image_name(self, index):
        # filename = self.dset_files[index]
        # root = filename.split("argoverse2")[0]
        # root = os.path.join(root, "argoverse2", self.train)

        image_name = self.f_content[1].split(",")[0]

        return os.path.join(self.sensor_path, image_name[1:])


    # def get_data_root(self):
    #     filename = self.dset_files[0]
    #     root = filename.split("argoverse2")[0]
    #     root = os.path.join(root, "argoverse2", self.train)

    #     return root

    def semantics_filter(self, keypoints, keypoints_classes, class_ids):
        filtered_kp = []
        filtered_classes = []
        # classes_to_ignore = ['construction', 'object', 'sky', 'nature', 'void']
        # taint_classes = ['ground', 'sidewalk', 'parking', 'rail track']
        # taint_categories = ['vehicle', 'human']
        lanes_mask_dict = {"lanes_mask": [], "intra_lane_masks": []}
        lanes_mask = []
        intra_lane_masks = []
        for i in range(len(keypoints)):
            lane = keypoints[i]
            classes = keypoints_classes[i]
            init_lane_length = len(lane)
            tainted_points = 0
            filtered_lane = []
            intra_lane_mask = []
            for j in range(len(classes)):
                current_label = labels[int(classes[j])]
                if current_label.category in self.semantics_config["classes_to_ignore"]:
                    tainted_points += 1
                    continue
                elif current_label.name in self.semantics_config["taint_classes"] or current_label.category in self.semantics_config["taint_categories"]:
                    tainted_points += 1
                filtered_lane.append(lane[j])
                intra_lane_mask.append(j)

            
            if tainted_points / init_lane_length < self.semantics_config["semantic_threshold"]:
                filtered_kp.append(np.array(filtered_lane))
                filtered_classes.append(class_ids[i])
                lanes_mask.append(i)
                intra_lane_masks.append(intra_lane_mask)
        

        lanes_mask_dict["lanes_mask"] = lanes_mask
        lanes_mask_dict["intra_lane_masks"] = intra_lane_masks
        return filtered_kp, filtered_classes, lanes_mask_dict


    def get_tusimple_kps(self):
        keypoints, class_ids = self.get_keypoints_from_file()
        keypoints, keypoints_classes = self.filter_twos(keypoints)
        keypoints, class_ids, mask = self.semantics_filter(keypoints, keypoints_classes, class_ids)
        #Not converting to TuSimple format as it expects the keypoints to be equally spaced
        # return np.array(keypoints), np.array(class_ids)
        return keypoints, class_ids, mask

    def get_3d_kps(self, mask=None):
        i = self.f_content.index("cam_label\n") + 2
        keypoints = []
        while True:
            if "lidar_label" in self.f_content[i]:
                break
            line_split = self.f_content[i].split("[")
            line_kp = line_split[1][:-3] #-3 is to get rid of the last comma
            class_id = line_split[0].split(",")[4]
            kp_list = [float(i) for i in line_kp.split(",")]
            if int(class_id) != 0:
                i += 1
                continue
            # if len(kp_list) < 10 and int(class_id) != 4: # class_id 4 is CL anchor
            #     i+=1
            #     continue
            keypoints.append(kp_list)
            i+=1
        filtered_kp = [np.array([[lane[i], lane[i+1], lane[i+2]] for i in range(0, len(lane), 3) if lane[i] != -2.0 or lane[i+1] != -2.0 or lane[i+2] != -2.0 ]) for lane in keypoints]
        filtered_kp_semantic = []
        if mask is not None:
            for i in range(len(mask["lanes_mask"])):
                current_lane = filtered_kp[mask["lanes_mask"][i]]
                # for j in range(len(mask["intra_lane_masks"][i])):
                filtered_kp_semantic.append(current_lane[mask["intra_lane_masks"][i]])
        return filtered_kp_semantic

    def get_dataset_files(self, path):
        file_list = []
        camera_params = {}

        i=0

        for root, _, files in os.walk(path):
            for file in files:
                if ".txt" in file:
                    i+=1
                    file_path = os.path.join(root, file)
                    with open(file_path, "r") as f:
                        # for now ignore content without labels 
                        num_labels = f.readlines()[1].split(",")[-1][:-1]

                        # must have at least one label with valid keypoints
                        # if (int(num_labels) > 0) and ("ring_front_center" in file_path):
                        if int(num_labels) > 0:
                            file_list.append(file_path)
        return file_list

    def convert_to_simple(self, keypoints):
        '''
        Method to convert to TuSimple format
        '''
        #Filter out the invisible points
        max_len = 0
        for lane in keypoints:
            if len(lane) > max_len:
                max_len = len(lane)
        for lane in keypoints:
            if len(lane) != max_len:
                for i in range(max_len - len(lane)):
                    lane.append(0.0)
        vertical = np.array(keypoints)[:, ::3]
        horizontal = np.array(keypoints)[:, 1::3]
        return vertical, horizontal

    def get_intrinsic_req(self, index):
        line_info = self.f_content[1].split(",")[0]
        split = line_info.split("/")
        log_name = split[1]
        cam_name = split[4]
        path = self.get_image_name(index).split("//")[0]
        # print(path)
        # return {"path": path, "log_name":log_name, "cam_name": cam_name}
        return path, log_name, cam_name
    
    def get_intrinsic(self, index):
        _, log_name, cam_name = self.get_intrinsic_req(index)
        pinhole_cam = self.av_loader.get_log_pinhole_camera(log_name, cam_name)
        cached_k = self.cam_params[log_name][cam_name]
        assert np.sum(cached_k - pinhole_cam.intrinsics.K) == 0 # temp  
        #Fetch from AV2
        
        # return pinhole_cam.intrinsics.K 
        return cached_k 

    def get_ego_SE3_cam(self, index):
        _, log_name, cam_name = self.get_intrinsic_req(index)
        pinhole_cam = self.av_loader.get_log_pinhole_camera(log_name, cam_name)
        
        return pinhole_cam.ego_SE3_cam

        



if __name__ == "__main__":
    dset_path = "/media/Data/argoverse2/labels/sample-val/data"
    keypoint_dset = KeypointLoader(dset_path, train=False)
    index = np.random.randint(0, len(keypoint_dset))
    keypoint_dset.__getitem__(index)
    