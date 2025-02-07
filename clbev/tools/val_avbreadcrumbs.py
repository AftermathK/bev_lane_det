import os
import sys
sys.path.append('/home/dfpazr/Documents/CogRob/avl/DSM/network_estimation/bev_lane_det')
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision as tv
from models.util.load_model import load_checkpoint, resume_training
from models.util.save_model import save_model_dp
from models.loss import IoULoss, NDPushPullLoss
from models.util.load_model import load_model
from models.util.cluster import embedding_post
from models.util.post_process import bev_instance2points_with_offset_z
from utils.config_util import load_config_module
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt
import cv2

model_path = '/home/dfpazr/Documents/CogRob/avl/DSM/network_estimation/bev_lane_det/breadcrumb_checkpoints/latest.pth'

''' parameter from config '''
config_file = './tools/breadcrumbs_config.py'
configs = load_config_module(config_file)
x_range = configs.x_range
y_range = configs.y_range
input_shape = configs.input_shape
meter_per_pixel = configs.meter_per_pixel

'''Post-processing parameters '''
post_conf = 0.1 # Minimum confidence on the BEV segmentation map for clustering
post_img_conf = -1.5 # Minimum confidence on the image segmentation map for clustering 
post_emb_margin = 6.0 # embeding margin of different clusters
post_min_cluster_size = 15 # The minimum number of points in a cluster

class Eval_Model(torch.nn.Module):
    def __init__(self, model):
        super(Eval_Model, self).__init__()
        self.model = model
        self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]))
        self.iou_loss = IoULoss()
        self.poopoo = NDPushPullLoss(1.0, 1., 1.0, 5.0, 200)
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, gt_seg=None, gt_instance=None, gt_offset_y=None, gt_z=None, image_gt_segment=None,
                image_gt_instance=None, train=False):
        res = self.model(inputs)
        pred, emb, offset_y, z = res[0]
        pred_2d, emb_2d = res[1]
        if train:
            ## 3d
            loss_seg = self.bce(pred, gt_seg) + self.iou_loss(torch.sigmoid(pred), gt_seg)
            loss_emb = self.poopoo(emb, gt_instance)
            loss_offset = self.bce_loss(gt_seg * torch.sigmoid(offset_y), gt_offset_y)
            loss_z = self.mse_loss(gt_seg * z, gt_z)
            loss_total = 3 * loss_seg + 0.5 * loss_emb
            loss_total = loss_total.unsqueeze(0)
            loss_offset = 60 * loss_offset.unsqueeze(0)
            loss_z = 30 * loss_z.unsqueeze(0)
            ## 2d
            loss_seg_2d = self.bce(pred_2d, image_gt_segment) + self.iou_loss(torch.sigmoid(pred_2d), image_gt_segment)
            loss_emb_2d = self.poopoo(emb_2d, image_gt_instance)
            loss_total_2d = 3 * loss_seg_2d + 0.5 * loss_emb_2d
            loss_total_2d = loss_total_2d.unsqueeze(0)
            return pred, loss_total, loss_total_2d, loss_offset, loss_z
        else:
            return pred


def test_epoch(model, dataset, optimizer, configs, epoch):

    # Last iter as mean loss of whole epoch
    model.eval()

    idx = 0

    if not os.path.exists('vis-img-test'):
        os.mkdir('vis-img-test')

    if not os.path.exists('vis-bev-test'):
        os.mkdir('vis-bev-test')

    '''image,image_gt_segment,image_gt_instance,ipm_gt_segment,ipm_gt_instance'''
    for idx, (
    orig_img, input_data, gt_seg_data, gt_emb_data, offset_y_data, z_data, image_gt_segment, image_gt_instance) in enumerate(
            dataset):

        # loss_back, loss_iter = forward_on_cuda(gpu, gt_data, input_data, loss, models)
        input_data = input_data.cuda()
        gt_seg_data = gt_seg_data.cuda()
        gt_emb_data = gt_emb_data.cuda()
        offset_y_data = offset_y_data.cuda()
        z_data = z_data.cuda()
        image_gt_segment = image_gt_segment.cuda()
        image_gt_instance = image_gt_instance.cuda()
        

        # input_data: [8, 3, 576, 1024]
        # gt_seg_data: [8, 1, 200, 48] (bev)
        # gt_emb_data: [8, 1, 200, 48] (bev)
        # offset_y_data: [8, 1, 200, 48] (bev)
        # z_data: [8, 1, 200, 48] (bev)
        # image_gt_segment: [8, 1, 144, 256] (image)
        # image_gt_instance: [8, 1, 144, 256] (image)
        pred = model(input_data)
        bev_pred, img_pred = pred[0], pred[1]

        bev_seg = bev_pred[0].detach().cpu().numpy()
        bev_embedding = bev_pred[1].detach().cpu().numpy()
        bev_offset_y = torch.sigmoid(bev_pred[2]).detach().cpu().numpy()
        bev_z_pred = bev_pred[3].detach().cpu().numpy()

        img_seg = img_pred[0]
        img_embedding = img_pred[1]

        img_seg_thresh = (img_seg > post_img_conf).detach().cpu().numpy().squeeze() 
        # img_seg_thresh = np.resize(img_seg_thresh, (144*4, 256*4))
        seg_idx = np.where(img_seg_thresh > 0)
        # orig_img = 255*input_data[0].squeeze().permute(1,2,0) 
        r, g, b = cv2.split(orig_img.squeeze().numpy())
        orig_img = cv2.merge([r,g,b])
        orig_img = cv2.resize(orig_img, (input_shape[1],input_shape[0]), interpolation=cv2.INTER_NEAREST)


        for i in range(len(seg_idx[0])):
            # orig_img[4*seg_idx[0][i], 4*seg_idx[1][i]] = 1
            orig_img = cv2.circle(orig_img, (4*seg_idx[1][i], 4*seg_idx[0][i]), 2, (0, 0, 255), -1)
        

            

        prediction = (bev_seg, bev_embedding) 
        canvas, ids = embedding_post(prediction, post_conf, emb_margin=post_emb_margin, min_cluster_size=post_min_cluster_size, canvas_color=False)
        offset_y = bev_offset_y[0][0]
        z_pred = bev_z_pred[0][0]
        lines = bev_instance2points_with_offset_z(canvas, max_x=x_range[1],
                                meter_per_pixal=(meter_per_pixel, meter_per_pixel),offset_y=offset_y,Z=z_pred)

        cv2.imwrite("./vis-img-test/" + str(idx) + ".png", orig_img)
        plt.imshow(canvas)
        plt.savefig("./vis-bev-test/" + str(idx) + ".png")
        

        idx += 1
        # prediction, loss_total_bev, loss_total_2d, loss_offset, loss_z = model(input_data,
        #                                                                         gt_seg_data,
        #                                                                         gt_emb_data,
        #                                                                         offset_y_data, z_data,
        #                                                                         image_gt_segment,
        #                                                                         image_gt_instance)
        # if idx % 300 == 0:
        #     target = gt_seg_data.detach().cpu().numpy().ravel()
        #     pred = torch.sigmoid(prediction).detach().cpu().numpy().ravel()
        #     f1_bev_seg = f1_score((target > 0.5).astype(np.int64), (pred > 0.5).astype(np.int64), zero_division=1)


def worker_function(config_file, gpu_id, checkpoint_path=None):
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
    optimizer = configs.optimizer(filter(lambda p: p.requires_grad, model.parameters()), **configs.optimizer_params)
    scheduler = getattr(configs, "scheduler", CosineAnnealingLR)(optimizer, configs.epochs)
    if checkpoint_path:
        if getattr(configs, "load_optimizer", True):
            resume_training(checkpoint_path, model.module, optimizer, scheduler)
        else:
            load_checkpoint(checkpoint_path, model.module, None)

    ''' dataset '''
    # test_ds = getattr(configs, "test_dataset", None)
    test_ds = configs.test_dataset()
    # if Dataset is None:
    #     Dataset = configs.training_dataset
    # test_loader = DataLoader(Dataset(), **configs.loader_args, pin_memory=True)
    test_loader = DataLoader(dataset=test_ds,
                             batch_size=1,
                             num_workers=8,
                             shuffle=False)


    ''' get validation '''
    # if configs.with_validation:
    #     val_dataset = Dataset(**configs.val_dataset_args)
    #     val_loader = DataLoader(val_dataset, **configs.val_loader_args, pin_memory=True)
    #     val_loss = getattr(configs, "val_loss", loss)
    #     if eval_only:
    #         loss_mean = val_dp(model, val_loader, val_loss)
    #         print(loss_mean)
    #         return

    for epoch in range(configs.epochs):
        print('*' * 100, epoch)
        test_epoch(model, test_loader, optimizer, configs, epoch)


# TODO template config file.
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    worker_function('./tools/breadcrumbs_config.py', gpu_id=[0,1])
