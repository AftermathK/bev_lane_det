import sys
sys.path.append('/naruarjun-central/cl_bev_lane_det/clbev')
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torch.nn as nn
from models.util.load_model import load_checkpoint, resume_training
from models.util.save_model import save_model_dp
from models.loss import IoULoss, NDPushPullLoss
from utils.config_util import load_config_module
from sklearn.metrics import f1_score
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
import os
from tools.breadcrumbs_config import *

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()



writer = SummaryWriter()
class Combine_Model_and_Loss(torch.nn.Module):
    def __init__(self, model):
        super(Combine_Model_and_Loss, self).__init__()
        self.model = model
        self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]))
        self.iou_loss = IoULoss()
        self.poopoo = NDPushPullLoss(1.0, 1., 1.0, 5.0, 200)
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, gt_seg=None, gt_instance=None, gt_offset_y=None, gt_z=None, image_gt_segment=None,
                image_gt_instance=None, train=True):
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


def train_epoch(model, dataset, optimizer, epoch, rank):

    # Last iter as mean loss of whole epoch
    model.train()
    losses_avg = {}
    '''image,image_gt_segment,image_gt_instance,ipm_gt_segment,ipm_gt_instance'''
    for idx, (
    orig_img, input_data, gt_seg_data, gt_emb_data, offset_y_data, z_data, image_gt_segment, image_gt_instance) in enumerate(
            dataset):

        # loss_back, loss_iter = forward_on_cuda(gpu, gt_data, input_data, loss, models)
        input_data = input_data.cuda(rank) # .cuda()
        gt_seg_data = gt_seg_data.cuda(rank)
        gt_emb_data = gt_emb_data.cuda(rank)
        offset_y_data = offset_y_data.cuda(rank)
        z_data = z_data.cuda(rank)
        image_gt_segment = image_gt_segment.cuda(rank)
        image_gt_instance = image_gt_instance.cuda(rank)
        
        # input_data: [8, 3, 576, 1024]
        # gt_seg_data: [8, 1, 200, 48] (bev)
        # gt_emb_data: [8, 1, 200, 48] (bev)
        # offset_y_data: [8, 1, 200, 48] (bev)
        # z_data: [8, 1, 200, 48] (bev)
        # image_gt_segment: [8, 1, 144, 256] (image)
        # image_gt_instance: [8, 1, 144, 256] (image)
        prediction, loss_total_bev, loss_total_2d, loss_offset, loss_z = model(input_data,
                                                                                gt_seg_data,
                                                                                gt_emb_data,
                                                                                offset_y_data, z_data,
                                                                                image_gt_segment,
                                                                                image_gt_instance)
        loss_back_bev = loss_total_bev.mean()
        loss_back_2d = loss_total_2d.mean()
        loss_offset = loss_offset.mean()
        loss_z = loss_z.mean()
        loss_back_total = loss_back_bev + 0.5 * loss_back_2d + loss_offset + loss_z
        ''' calcute loss '''


        optimizer.zero_grad()
        loss_back_total.backward()
        optimizer.step()
        writer.add_scalar("Loss/bev", loss_back_bev, epoch)
        writer.add_scalar("Loss/2d", loss_back_2d, epoch)
        writer.add_scalar("Loss/offset", loss_offset, epoch)
        writer.add_scalar("Loss/z", loss_z, epoch)
        writer.add_scalar("Loss/overall", loss_back_total, epoch)

        if idx % 50 == 0:
            print(idx, loss_back_bev.item(), '*' * 10)
        if idx % 300 == 0:
            target = gt_seg_data.detach().cpu().numpy().ravel()
            pred = torch.sigmoid(prediction).detach().cpu().numpy().ravel()
            f1_bev_seg = f1_score((target > 0.5).astype(np.int64), (pred > 0.5).astype(np.int64), zero_division=1)
            loss_iter = {"BEV Loss": loss_back_bev.item(), 'offset loss': loss_offset.item(), 'z loss': loss_z.item(),
                            "F1_BEV_seg": f1_bev_seg}
            # losses_show = loss_iter
            print(idx, loss_iter)
        if multiGPU == True:
            dist.barrier()
    dataset.dataset.sample_data_loader.reset_dset_files()
    print("Dataset Reset")


def worker_function(rank, gpu_id, checkpoint_path=None):
    print('use gpu ids is '+','.join([str(i) for i in gpu_id]))
    # name = config_file.split(".py")[0].replace("/", ".")
    # configs = load_config_module(name, config_file)
    setup(rank, torch.cuda.device_count())
    ''' models and optimizer '''
    model = br_model()
    model = Combine_Model_and_Loss(model)
    print("Model Loaded")
    # if torch.cuda.is_available():
    #     model = model.cuda()
    model.cuda(rank)
    print("Model moved")
    model.train()
    if multiGPU == True:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    else:
        model = torch.nn.DataParallel(model)
    print("Multi-GPU done")
    optimizer = br_optimizer(filter(lambda p: p.requires_grad, model.parameters()), **optimizer_params)
    scheduler = CosineAnnealingLR(optimizer, epochs)
    # if checkpoint_path:
    #     if getattr(configs, "load_optimizer", True):
    #         resume_training(checkpoint_path, model.module, optimizer, scheduler)
    #     else:
    #         load_checkpoint(checkpoint_path, model.module, None)

    ''' dataset '''
    Dataset = train_dataset # getattr(configs, "train_dataset", None)
    # if Dataset is None:
    #     Dataset = configs.training_dataset
    dataset = Dataset()
    train_loader = DataLoader(dataset, **loader_args, sampler = torch.utils.data.distributed.DistributedSampler(dataset), pin_memory=True)

    ''' get validation '''
    # if configs.with_validation:
    #     val_dataset = Dataset(**configs.val_dataset_args)
    #     val_loader = DataLoader(val_dataset, **configs.val_loader_args, pin_memory=True)
    #     val_loss = getattr(configs, "val_loss", loss)
    #     if eval_only:
    #         loss_mean = val_dp(model, val_loader, val_loss)
    #         print(loss_mean)
    #         return

    for epoch in range(epochs):
        print('*' * 100, epoch)
        train_epoch(model, train_loader, optimizer, epoch, rank)
        scheduler.step()
        save_model_dp(model, optimizer, model_save_path, 'ep%03d.pth' % epoch)
        save_model_dp(model, None, model_save_path, 'latest.pth')


# TODO template config file.
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    config_file = './clbev/tools/breadcrumbs_config.py'
    configs = load_config_module(config_file)
    # import pickle
    # with open("config_save", "wb") as f:
    #     pickle.dump(configs, f)
    mp.spawn(worker_function,
            args=([0,1],),
            nprocs=torch.cuda.device_count(),
            join=True)
    # cleanup()
    # worker_function('./clbev/tools/breadcrumbs_config.py', gpu_id=[0,1])
