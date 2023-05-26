import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms, utils
import os
import cv2
import torch
from scipy.interpolate import interp1d
from torch.utils.data import Dataset
from utils.coord_util import ego2image,IPM2ego_matrix
from utils.standard_camera_cpu import Standard_camera
# from parameters import Parameters
from loader.bev_road.textfileloader import KeypointLoader
import matplotlib.pyplot as plt

class Generator(Dataset):
    def __init__(self,
                 txt_file_path,
                 dataset_base_dir,
                 x_range,
                 y_range,
                 meter_per_pixel,
                 data_trans,
                 output_2d_shape,
                 virtual_camera_config,
                 vertical_center_cam_crop,
                 split="train",
                 sample_skip=10):
        super(Generator, self).__init__()

        # data params
        self.txt_file_path = txt_file_path
        self.sensor_root_dir = dataset_base_dir

        # model params 
        self.x_range = x_range
        self.y_range = y_range
        self.center_crop_min, self.center_crop_max = vertical_center_cam_crop
        self.meter_per_pixel = meter_per_pixel
        self.lane3d_thick = 1
        self.lane2d_thick = 3
        self.dataset_base_dir = dataset_base_dir

        ''' virtual camera paramter'''
        self.use_virtual_camera = virtual_camera_config['use_virtual_camera']
        self.vc_intrinsic = virtual_camera_config['vc_intrinsic']
        self.vc_extrinsics = virtual_camera_config['vc_extrinsics']
        self.vc_image_shape = virtual_camera_config['vc_image_shape']

        ''' transform loader '''
        self.output2d_size = output_2d_shape
        self.trans_image = data_trans
        self.ipm_h, self.ipm_w = int((self.x_range[1] - self.x_range[0]) / self.meter_per_pixel), int(
            (self.y_range[1] - self.y_range[0]) / self.meter_per_pixel)

        # orig params
        # self.p = Parameters()
        self.image_transform = data_trans 
        self.txt_file_path = txt_file_path 
        # self.sensor_root_dir = self.p.sensor_root_url
        # self.center_crop_min = self.p.center_crop_min
        # self.center_crop_max = self.p.center_crop_max
        self.sample_data_loader = KeypointLoader(self.txt_file_path, self.sensor_root_dir, split, skip=sample_skip)
        self.num_classes = 4



    def get_y_offset_and_z(self,res_d):
        def caculate_distance(base_points, lane_points, lane_z, lane_points_set):
            condition = np.where(
                (lane_points_set[0] == int(base_points[0])) & (lane_points_set[1] == int(base_points[1])))
            if len(condition[0]) == 0:
                return None,None
            lane_points_selected = lane_points.T[condition]  # 找到bin
            lane_z_selected = lane_z.T[condition]
            offset_y = np.mean(lane_points_selected[:, 1]) - base_points[1]
            z = np.mean(lane_z_selected[:, 1])
            return offset_y, z  # @distances.argmin(),distances[min_idx] #1#lane_points_selected[distances.argmin()],distances.min()

        # mask
        res_lane_points = {}
        res_lane_points_z = {}
        res_lane_points_bin = {}
        res_lane_points_set = {}
        for idx in res_d:
            ipm_points_ = np.array(res_d[idx])
            ipm_points = ipm_points_.T[np.where((ipm_points_[1] >= 0) & (ipm_points_[1] < self.ipm_h))].T  # 进行筛选
            if len(ipm_points[0]) <= 1:
                continue
            x, y, z = ipm_points[1], ipm_points[0], ipm_points[2]
            base_points = np.linspace(x.min(), x.max(),
                                      int((x.max() - x.min()) // 0.05))  # 画 offset 用得 画的非常细 一个格子里面20个点
            base_points_bin = np.linspace(int(x.min()), int(x.max()),
                                          int(int(x.max()) - int(x.min()))+1)  # .astype(np.int)
            # print(len(x),len(y),len(y))
            if len(x) == len(set(x)):
                if len(x) <= 1:
                    continue
                elif len(x) <= 2:
                    function1 = interp1d(x, y, kind='linear',
                                         fill_value="extrapolate")  # 线性插值 #三次样条插值 kind='quadratic' linear cubic
                    function2 = interp1d(x, z, kind='linear')
                elif len(x) <= 3:
                    function1 = interp1d(x, y, kind='quadratic', fill_value="extrapolate")
                    function2 = interp1d(x, z, kind='quadratic')
                else:
                    function1 = interp1d(x, y, kind='cubic', fill_value="extrapolate")
                    function2 = interp1d(x, z, kind='cubic')
            else:
                sorted_index = np.argsort(x)[::-1] # 从大到小
                x_,y_,z_ = [],[],[]
                for x_index in range(len(sorted_index)): # 越来越小
                    if x[sorted_index[x_index]] >= x[sorted_index[x_index-1]] and x_index !=0:
                        continue
                    else:
                        x_.append(x[sorted_index[x_index]])
                        y_.append(y[sorted_index[x_index]])
                        z_.append(z[sorted_index[x_index]])
                x,y,z = np.array(x_),np.array(y_),np.array(z_)
                if len(x) <= 1:
                    continue
                elif len(x) <= 2:
                    function1 = interp1d(x, y, kind='linear',
                                         fill_value="extrapolate")  # 线性插值 #三次样条插值 kind='quadratic' linear cubic
                    function2 = interp1d(x, z, kind='linear')
                elif len(x) <= 3:
                    function1 = interp1d(x, y, kind='quadratic', fill_value="extrapolate")
                    function2 = interp1d(x, z, kind='quadratic')
                else:
                    function1 = interp1d(x, y, kind='cubic', fill_value="extrapolate")
                    function2 = interp1d(x, z, kind='cubic')

            y_points = function1(base_points)
            y_points_bin = function1(base_points_bin)
            z_points = function2(base_points)
            # cv2.polylines(instance_seg, [ipm_points.T.astype(np.int)], False, idx+1, 1)
            res_lane_points[idx] = np.array([base_points, y_points])  # 
            res_lane_points_z[idx] = np.array([base_points, z_points])
            res_lane_points_bin[idx] = np.array([base_points_bin, y_points_bin]).astype(np.int)  # 画bin用的
            res_lane_points_set[idx] = np.array([base_points, y_points]).astype(
                np.int)  
        offset_map = np.zeros((self.ipm_h, self.ipm_w))
        z_map = np.zeros((self.ipm_h, self.ipm_w))
        ipm_image = np.zeros((self.ipm_h, self.ipm_w))
        for idx in res_lane_points_bin:
            lane_bin = res_lane_points_bin[idx].T
            for point in lane_bin:
                row, col = point[0], point[1]
                if not ( 0 < row < self.ipm_h and 0 < col < self.ipm_w): # 没有在视野内部的去除掉
                    continue
                ipm_image[row, col] = idx
                center = np.array([row, col])
                offset_y, z = caculate_distance(center, res_lane_points[idx], res_lane_points_z[idx],
                                                res_lane_points_set[idx])  # 根据距离选idex
                if offset_y is None: #
                    ipm_image[row,col] = 0
                    continue
                if offset_y > 1:
                    print('haha')
                    offset_y = 1
                if offset_y < 0:
                    print('hahahahahha')
                    offset_y = 0
                offset_map[row][col] = offset_y
                z_map[row][col] = z

        return ipm_image,offset_map,z_map

    def get_camera_matrix(self,cam_pitch,cam_height):
        proj_g2c = np.array([[1,                             0,                              0,          0],
                            [0, np.cos(np.pi / 2 + cam_pitch), -np.sin(np.pi / 2 + cam_pitch), cam_height],
                            [0, np.sin(np.pi / 2 + cam_pitch),  np.cos(np.pi / 2 + cam_pitch),          0],
                            [0,                             0,                              0,          1]])


        camera_K = np.array([[2015., 0., 960.],
                        [0., 2015., 540.],
                        [0., 0., 1.]])


        return proj_g2c,camera_K
              
    def update_intrinsics(self, intrinsics, l_u=0.0, l_v=0.0, r_u=1.0, r_v=1.0):
        """_summary_
        Args:
            intrinsics (ndarray): (3, 3) camera parameters 
            l_u (int, optional): image crop along u axis starting from the left. Defaults to 0.
            l_v (int, optional): image crop along v axis starting from the top. Defaults to 0.
            r_u (int, optional): image resize factor along u axis. Defaults to 1.
            r_v (int, optional): image resize factor along v axis. Defaults to 1.
        Returns:
            new_intrinsics: (3, 3) ndarray for updated camera parameters assuming resize and crop factors. 
        """

        crop_matrix = np.array([[1.0, 0, -l_u], [0.0, 1.0, -l_v], [0.0, 0.0, 1.0]])
        resize_matrix = np.array([[r_u, 0.0, 0.0], [0.0, r_v, 0.0], [0.0, 0.0, 1.0]])
        # new_intrinsics = resize_matrix @ crop_matrix @ intrinsics
        new_intrinsics = crop_matrix @ resize_matrix @ intrinsics

        return new_intrinsics

    def remove_invalid_points(self, uv_points, width, height):
        is_valid_x = np.logical_and(0 <= uv_points[:, 0], uv_points[:, 0] < width - 1)
        is_valid_y = np.logical_and(0 <= uv_points[:, 1], uv_points[:, 1] < height - 1)
        idx = np.logical_and.reduce([is_valid_x, is_valid_y])
        return idx
    def get_seg_offset(self,idx):
        # ------ TODO -------
        # info_dict = self.cnt_list[idx]
        # name_list = info_dict['raw_file'].split('/')
        # image_path = os.path.join(self.dataset_base_dir, 'images', name_list[-2], name_list[-1])
        # image = cv2.imread(image_path)

        tensor_convert = ToTensor()
        file_name = self.sample_data_loader.read_and_set_file(idx)
        image_name = self.sample_data_loader.get_image_name(idx) 
        image = cv2.imread(image_name)
        h, w, _ = image.shape
        camera_k = self.sample_data_loader.get_intrinsic(idx)

        # if "ring_front_center" in image_name:
        #     image = image[self.center_crop_min:self.center_crop_max, :]
        #     camera_k = self.update_intrinsics(camera_k, l_u=0, l_v=self.center_crop_min) # crop only
        #     h = h - self.center_crop_min - (h - self.center_crop_max)

        ego2cam = self.sample_data_loader.get_ego_SE3_cam(idx)

        keypoints, class_ids, mask = self.sample_data_loader.get_tusimple_kps()
        keypoints_3d = self.sample_data_loader.get_3d_kps(mask)


        # ------ TODO -------
        # caculate camera parameter
        cam_height = 1.7860000133514404
        cam_pitch = 0.01768055371940136
        project_g2c, _ = self.get_camera_matrix(cam_pitch, cam_height)
        project_c2g = np.linalg.inv(project_g2c)

        # caculate point
        image_gt = np.zeros(image.shape[:2], dtype=np.uint8)
        matrix_IPM2ego = IPM2ego_matrix(
            ipm_center=(int(self.x_range[1] / self.meter_per_pixel), int(self.y_range[1] / self.meter_per_pixel)),
            m_per_pixel=self.meter_per_pixel)
        res_points_d = {}
        for lane_idx in range(len(keypoints_3d)):
            lane_camera = keypoints_3d[lane_idx]
            lane_camera = np.concatenate([lane_camera, np.ones([lane_camera.shape[0], 1])], axis=1).T # (4, lane_size)

            # get image gt
            lane_ground = np.matmul(project_c2g, lane_camera)
            lane_image = camera_k @ lane_camera[:3]
            lane_image = lane_image / lane_image[2]
            lane_uv = lane_image[:2].T

            cv2.polylines(image_gt, [lane_uv.astype(np.int)], False, lane_idx + 1, 3)
            x, y, z = lane_ground[1], -1 * lane_ground[0], lane_ground[2]
            ground_points = np.array([x, y])
            ipm_points = np.linalg.inv(matrix_IPM2ego[:, :2]) @ (
                        ground_points[:2] - matrix_IPM2ego[:, 2].reshape(2, 1))
            ipm_points_ = np.zeros_like(ipm_points)
            ipm_points_[0] = ipm_points[1]
            ipm_points_[1] = ipm_points[0]
            res_points = np.concatenate([ipm_points_, np.array([z])], axis=0)
            res_points_d[lane_idx+1] = res_points

        plt.imshow(image_gt, 'gray')
        plt.show()

        bev_gt,offset_y_map,z_map = self.get_y_offset_and_z(res_points_d)

        if "ring_front_center" in image_name:
            image = image[self.center_crop_min:self.center_crop_max, :]
            image_gt = image_gt[self.center_crop_min:self.center_crop_max]
            
        return image,image_gt,bev_gt,offset_y_map,z_map,project_c2g,camera_k

    def __len__(self):
        return len(self.sample_data_loader)

    def __getitem__(self, idx):
        '''
        :param idx:
        :return:
        '''
        image, image_gt, bev_gt, offset_y_map, z_map, cam_extrinsics, cam_intrinsic = self.get_seg_offset(idx)
        print(self.sample_data_loader.get_image_name(idx))
        orig_image = np.copy(image)
        transformed = self.trans_image(image=image)
        image = transformed["image"]

        ''' 2d gt'''
        image_gt = self.gt_resize(image_gt) # cv2.resize(image_gt, (self.output2d_size[1],self.output2d_size[0]), interpolation=cv2.INTER_NEAREST)
        image_gt_instance = torch.tensor(image_gt).unsqueeze(0)  # h, w, c
        image_gt_segment = torch.clone(image_gt_instance)
        image_gt_segment[image_gt_segment > 0] = 1

        ''' 3d gt'''
        bev_gt_instance = torch.tensor(bev_gt).unsqueeze(0)  # h, w, c0
        bev_gt_offset = torch.tensor(offset_y_map).unsqueeze(0)
        bev_gt_z = torch.tensor(z_map).unsqueeze(0)
        bev_gt_segment = torch.clone(bev_gt_instance)
        bev_gt_segment[bev_gt_segment > 0] = 1
        # orig_image = np.expand_dims(orig_image, 0)

        return orig_image, image, bev_gt_segment.float(), bev_gt_instance.float(),bev_gt_offset.float(),bev_gt_z.float(),image_gt_segment.float(),image_gt_instance.float()

    def old_get_item(self, i):
        tensor_convert = ToTensor()
        file_name = self.sample_data_loader.read_and_set_file(i)
        image_name = self.sample_data_loader.get_image_name(i) 
        image = cv2.imread(image_name)
        ratio_w = self.p.x_size/image.shape[1]
        ratio_h = self.p.y_size/image.shape[0]
        kp_ratio_w = self.p.kp_x_size / image.shape[1]
        kp_ratio_h = self.p.kp_y_size / image.shape[0]
        keypoints, class_ids = self.sample_data_loader.get_tusimple_kps()
        keypoints_3d = self.sample_data_loader.get_3d_kps()
        
        intrinsic_params = self.sample_data_loader.get_intrinsic(i)        
        # crop_factor = [0, 0] # [u (left), v (top)]
        resize_factor = [(image.shape[0]/self.p.x_size) * self.p.resize_ratio, 
                         (image.shape[1]/self.p.y_size) * self.p.resize_ratio] # from original to (x_size, y_size) [u, v]

        # number of lane objects in this sample should match 
        assert len(keypoints) == len(class_ids) == len(keypoints_3d)

        # crop_factor = [0, 0] if "center" in image_name else [0, self.p.center_crop_min] 
        crop_factor = [0, 0]
        image_dims = image.shape
        # crop center image
        if "center" in image_name:
            crop_factor = [0, self.p.center_crop_min]
            image = image[self.center_crop_min:self.center_crop_max, :]
            ratio_w = self.p.x_size/image.shape[1]
            ratio_h = self.p.y_size/image.shape[0]
            kp_ratio_w = self.p.kp_x_size / image.shape[1]
            kp_ratio_h = self.p.kp_y_size / image.shape[0]
                
        image = cv2.resize(image, (self.p.x_size,self.p.y_size))

        max_keypoint_length = 0

        # crop and resize keypoints
        for i in range(len(keypoints)):

            # eliminates keypoints outside cropped region 
            if "center" in image_name:
                mask_i = keypoints[i][:, 1] <= self.center_crop_max
                # image keypoints
                keypoints[i] = keypoints[i][mask_i]
                keypoints[i][:, 1] -= self.center_crop_min 

                # match 3d keypoints
                keypoints_3d[i] = keypoints_3d[i][mask_i]

            # resize keypoints    
            keypoints[i][:, 0] = keypoints[i][:, 0] * ratio_w 
            keypoints[i][:, 1] = keypoints[i][:, 1] * ratio_h

            mask_i = keypoints[i][:, 0] < self.p.x_size
            keypoints[i] = keypoints[i][mask_i]
            keypoints_3d[i] = keypoints_3d[i][mask_i]

            mask_i = keypoints[i][:, 1] < self.p.y_size
            keypoints[i] = keypoints[i][mask_i]
            keypoints_3d[i] = keypoints_3d[i][mask_i]

            if len(keypoints[i]) > max_keypoint_length:
                max_keypoint_length = len(keypoints[i])
        
        # pad labels
        for i in range(len(keypoints)):
            pad_size = max_keypoint_length - len(keypoints[i])

            if len(keypoints[i]) != len(keypoints_3d[i]):
                print("Mismatch in data")

            if pad_size > 0:
                keypoints[i] = np.vstack((keypoints[i], -2.0 * np.ones((pad_size, 2))))
                keypoints_3d[i] = np.vstack((keypoints_3d[i], -2.0 * np.ones((pad_size, 3))))
        
        image = image[:, :, ::-1]    
        
        save = False
        if save:
            #Plot the data for testing
            plt.imshow(image)
            for i in range(len(keypoints)):
                plt.scatter(keypoints[i][:, 0], keypoints[i][:, 1], marker="x", color="red", s=1)
            plt.show()
            plt.savefig("test_image.png")
        
        # keypoints_tensor = self.get_keypoint_tensor(keypoints, class_ids, image.shape)

        image = torch.Tensor(image.copy()).permute(2, 0, 1)
        # image = self.img_tf(image)
        # ret_dict = {"image": image, "labels":keypoints_tensor}
        cam_params = {"K": torch.Tensor(intrinsic_params),
                      "crop_factor": crop_factor,
                      "resize_factor": resize_factor,
                      "image_dims": image_dims}

        ret_dict = {"image": image, 
                    "image_path": image_name,
                    "class_ids": class_ids, 
                    "keypoints_2d": torch.Tensor(keypoints), 
                    "keypoints_3d": torch.Tensor(keypoints_3d),
                    "camera_params": cam_params}
        
        # input_data: [8, 3, 576, 1024]
        # gt_seg_data: [8, 1, 200, 48] (bev)
        # gt_emb_data: [8, 1, 200, 48] (bev)
        # offset_y_data: [8, 1, 200, 48] (bev)
        # z_data: [8, 1, 200, 48] (bev)
        # image_gt_segment: [8, 1, 144, 256] (image)
        # image_gt_instance: [8, 1, 144, 256] (image)

        # return image, bev_gt_segment.float(), bev_gt_instance.float(),bev_gt_offset.float(),bev_gt_z.float(),image_gt_segment.float(),image_gt_instance.float()

        return (ret_dict)

    def gt_resize(self, image_gt):
        image_gt_resize = np.zeros((self.output2d_size[0],self.output2d_size[1]), dtype=np.uint8)
        indices = np.argwhere(image_gt>0)
        original_image_center = np.array([[(image_gt.shape[0]-1)/2, (image_gt.shape[1]-1)/2]])
        scaled_image_center = np.array([[(self.output2d_size[0]-1)/2, (self.output2d_size[1]-1)/2]])
        scale = np.array([[self.output2d_size[0]/image_gt.shape[0], self.output2d_size[1]/image_gt.shape[1]]])
        mapped_indices = ((indices - original_image_center)*scale + scaled_image_center).astype(int)
        image_gt_resize[mapped_indices[:,0], mapped_indices[:,1]] = image_gt[indices[:,0], indices[:,1]]
        return image_gt_resize

    def get_keypoint_tensor(self, keypoints, class_ids, im_shape):
        # im_shape = (int(im_shape[0] / 8), int(im_shape[1]/8))
        kp_tensor = np.zeros((self.num_classes, *im_shape[:2]))
        for i, id in enumerate(class_ids):
            kp = np.floor(keypoints[i]).astype(np.int)
            for k in kp:
                kp_tensor[int(id), k[1], k[0]] = 1 
        return torch.Tensor(kp_tensor)
        
class ToTensor(object):
    def __call__(self, nparray):
        image = nparray["image"]
        image = image.transpose((2, 0, 1))
        return {"image": image, "labels": nparray["labels"]}

if __name__ == "__main__":
    gen = Generator(train=False)
    random_index = np.random.randint(0, len(gen))
    print(random_index)
    #Assert test
    image = gen.__getitem__(random_index)
    
    print("No failures!")
    