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


class KeypointLoader():
    def __init__(self, labels_path, sensor_path, split, skip=10):
        self.dset_path = labels_path 
        self.sensor_path = sensor_path
        self.data_skip = skip
        self.transforms = transforms
        self.split = split

        # self.train = "train/" if train else "val/" #Validation set's sensor folder should be reorganized
        #Create a list of dataset files
        self.dset_files = self.get_dataset_files(self.dset_path)
        self.dset_files = self.dset_files[::skip] #Adding skip to avoid consecuitve frame of data
        self.cam_params = {} # dictionary of camera parameters
        self.image_shape = (512, 256) #Sticking to PiNet

        # for extracting additional AV2 data
        self.av_loader = AV2SensorDataLoader(data_dir=Path(sensor_path), labels_dir=Path(sensor_path))

        print("Dataset size: {}".format(len(self.dset_files)))

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
        return filtered_kp #Returing the orignal unraveled tuple

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

    def get_tusimple_kps(self):
        keypoints, class_ids = self.get_keypoints_from_file()
        keypoints = self.filter_twos(keypoints)
        #Not converting to TuSimple format as it expects the keypoints to be equally spaced
        # return np.array(keypoints), np.array(class_ids)
        return keypoints, class_ids

    def get_3d_kps(self):
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
        filtered_kp = [np.array([[lane[i], lane[i+1], lane[i+2]] for i in range(0, len(lane), 3) if lane[i] != -2.0]) for lane in keypoints]
        return filtered_kp

    def get_dataset_files(self, path):
        file_list = []
        camera_params = {}

        for root, _, files in os.walk(path):
            for file in files:
                if ".txt" in file:
                    file_path = os.path.join(root, file)
                    with open(file_path, "r") as f:
                        # for now ignore content without labels 
                        num_labels = f.readlines()[1].split(",")[-1][:-1]

                        # must have at least one label with valid keypoints
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
    