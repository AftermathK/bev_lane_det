import enum
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from loader.bev_road.breadcrumbs_loader import Generator
from models.model.single_camera_bev import BEV_LaneDet
import torch

def get_camera_matrix(cam_pitch,cam_height):
    proj_g2c = np.array([[1,                             0,                              0,          0],
                        [0, np.cos(np.pi / 2 + cam_pitch), -np.sin(np.pi / 2 + cam_pitch), cam_height],
                        [0, np.sin(np.pi / 2 + cam_pitch),  np.cos(np.pi / 2 + cam_pitch),          0],
                        [0,                             0,                              0,          1]])


    camera_K = np.array([[2015., 0., 960.],
                    [0., 2015., 540.],
                    [0., 0., 1.]])


    return proj_g2c,camera_K

def custom_collate(batch):
    # max_keypoint_length = 0
    orig_images = []
    input_images = []
    gt_seg_list = []
    gt_emb_list = []
    offset_y_list = []
    z_list = []
    image_gt_list = []
    image_emb_list = []

    for i, sample in enumerate(batch):
        orig_images.append(np.expand_dims(sample[0], 0))
        input_images.append(sample[1])
        gt_seg_list.append(sample[2])
        gt_emb_list.append(sample[3])
        offset_y_list.append(sample[4])
        z_list.append(sample[5])
        image_gt_list.append(sample[6])
        image_emb_list.append(sample[7])



        
    return orig_images, torch.stack(input_images, 0), torch.stack(gt_seg_list, 0), torch.stack(gt_emb_list, 0), torch.stack(offset_y_list, 0), torch.stack(z_list, 0), torch.stack(image_gt_list), torch.stack(image_emb_list, 0) 

''' data split '''
# train_txt_paths = '/mnt/00B0A680755C4DFA/DevSpace/DSM/datasets/argoverse2/labels/sample-labels/val/04994d08-156c-3018-9717-ba0e29be8153'
# test_txt_paths = '/mnt/00B0Aj680755C4DFA/DevSpace/DSM/datasets/argoverse2/labels/sample-labels/val/04994d08-156c-3018-9717-ba0e29be8153'
# data_base_path = '/mnt/00B0A680755C4DFA/DevSpace/DSM/datasets/argoverse2/val'
train_txt_paths = '/mnt/00B0A680755C4DFA/DevSpace/DSM/datasets/argoverse2/labels/labels-v4/test'
test_txt_paths = '/mnt/00B0A680755C4DFA/DevSpace/DSM/datasets/argoverse2/labels/labels-v4/test'
data_base_path = '/mnt/00B0A680755C4DFA/DevSpace/DSM/datasets/argoverse2/test'
model_save_path = "/home/dfpazr/Documents/CogRob/avl/DSM/network_estimation/CenterlineNetworks/clbev/checkpoints"

input_shape = (576,1024)
output_2d_shape = (144,256)
vertical_center_cam_crop = (750, 1750) # crop along rows
frame_skip = 20 # skip 20 frames (2 seconds)

''' BEV range '''
x_range = (3, 103)
y_range = (-12, 12)
meter_per_pixel = 0.5 # grid size
bev_shape = (int((x_range[1] - x_range[0]) / meter_per_pixel),int((y_range[1] - y_range[0]) / meter_per_pixel))

loader_args = dict(
    batch_size=10,
    num_workers=12,
    shuffle=True,
    collate_fn=custom_collate
)

''' virtual camera config '''
camera_ext_virtual, camera_K_virtual = get_camera_matrix(0.04325083977888603, 1.7860000133514404) # a random parameter
vc_config = {}
vc_config['use_virtual_camera'] = True
vc_config['vc_intrinsic'] = camera_K_virtual
vc_config['vc_extrinsics'] = np.linalg.inv(camera_ext_virtual)
vc_config['vc_image_shape'] = (1920, 1080)


''' model '''
def model():
    return BEV_LaneDet(bev_shape=bev_shape, output_2d_shape=output_2d_shape,train=True)


''' optimizer '''
epochs = 70
optimizer = AdamW
optimizer_params = dict(
    lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
    weight_decay=1e-2, amsgrad=False
)
scheduler = CosineAnnealingLR


def train_dataset():
    train_trans = A.Compose([
                    A.Resize(height=input_shape[0], width=input_shape[1]),
                    A.MotionBlur(p=0.2),
                    A.RandomBrightnessContrast(),
                    A.ColorJitter(p=0.1),
                    A.Normalize(),
                    ToTensorV2()
                    ])
    train_data = Generator(train_txt_paths, data_base_path, 
                           x_range, y_range, meter_per_pixel, 
                           train_trans, output_2d_shape, vc_config, vertical_center_cam_crop, split="train", sample_skip=frame_skip)

    return train_data

def test_dataset():
    # test_trans = A.Compose([
    #                 A.Resize(height=input_shape[0], width=input_shape[1]),
    #                 A.MotionBlur(p=0.2),
    #                 A.RandomBrightnessContrast(),
    #                 A.ColorJitter(p=0.1),
    #                 A.Normalize(),
    #                 ToTensorV2()
    #                 ])
    test_trans = A.Compose([
                    A.Resize(height=input_shape[0], width=input_shape[1]),
                    A.Normalize(),
                    ToTensorV2()
                    ])
    test_data = Generator(test_txt_paths, data_base_path, 
                           x_range, y_range, meter_per_pixel, 
                           test_trans, output_2d_shape, vc_config, vertical_center_cam_crop, split="train", sample_skip=frame_skip)

    return test_data 

# def val_dataset():
#     trans_image = A.Compose([
#         A.Resize(height=input_shape[0], width=input_shape[1]),
#         A.Normalize(),
#         ToTensorV2()])
#     val_data = Apollo_dataset_with_offset_val(test_json_paths,data_base_path,
#                                                 trans_image,vc_config)
#     return val_data



