from cfg import constants
from utils.multiview import *
from utils.conversion import *
from utils.pose_utils import *

import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

DEFAULT_DTYPE = torch.float

def reconstruction_error(S1, S2, reduction='mean'):
    """Do Procrustes alignment and compute reconstruction error."""
    S1_hat = compute_similarity_transform_batch(S1, S2)
    re = np.sqrt( ((S1_hat - S2)** 2).sum(axis=-1)).mean(axis=-1)
    if reduction == 'mean':
        re = re.mean()
    elif reduction == 'sum':
        re = re.sum()
    return re


class Evaluator():
    def __init__(self, J_regressor, device, num_joints=17, **kwargs):

        self.J_regressor = torch.from_numpy(np.load(J_regressor)).float()
        self.device = device
        self.num_joints = num_joints

    def __call__(self, pred_output, keypoints_3d_gt, ):
        num_joints_ = self.num_joints
        self.num_joints = 14

        J_regressor = self.J_regressor[None, :].expand(keypoints_3d_gt.shape[0], -1, -1).to(self.device)
        keypoints_3d_pred = torch.matmul(J_regressor, pred_output.vertices)
        keypoints_3d_pred = keypoints_3d_pred[:, constants.H36M_TO_J17, :]
        keypoints_3d_gt = keypoints_3d_gt / 1e3

        mpjpe, _ = self.keypoints_3d_error_in_batch(keypoints_3d_gt, keypoints_3d_pred)
        recone = self.reconstruction_error_in_batch(keypoints_3d_gt, keypoints_3d_pred)

        self.num_joints = num_joints_

        return mpjpe.mean().item() * 1e3, recone.mean() * 1e3

    def centering_joints(self, joints):
        sacrum_center = joints[:, 14]
        joints_ = joints - sacrum_center.unsqueeze(1)

        return joints_

    def align_two_joints(self, gt_joints, pred_joints, opt_joints=None):

        gt_joints = self.centering_joints(gt_joints)
        pred_joints = self.centering_joints(pred_joints)

        if opt_joints is not None:
            opt_joints = self.centering_joints(opt_joints)

        return gt_joints, pred_joints, opt_joints


    def keypoints_3d_error_in_batch(self, keypoints_3d_gt, keypoints_3d_pred, keypoints_3d_opt=None):
        # if self.data_type == 'human36m':

        # keypoints_3d_pred = 1000 * keypoints_3d_pred[:, 25:][:, constants.SMPL_TO_H36].clone().detach()
        # if keypoints_3d_opt is not None:
        #     keypoints_3d_opt = 1000 * keypoints_3d_opt[:, 25:][:, constants.SMPL_TO_H36].clone().detach()

        if keypoints_3d_gt.size(-1) == 4:
            keypoints_3d_gt = keypoints_3d_gt[:, :, :-1].detach()

        keypoints_3d_gt, keypoints_3d_pred, keypoints_3d_opt =\
            self.align_two_joints(keypoints_3d_gt, keypoints_3d_pred, keypoints_3d_opt)
        keypoints_3d_gt = keypoints_3d_gt[:, :self.num_joints]
        keypoints_3d_pred = keypoints_3d_pred[:, :self.num_joints]
        pred_error_in_batch = torch.sqrt(((keypoints_3d_pred - keypoints_3d_gt)**2).sum(-1)).mean(1)

        if keypoints_3d_opt is not None:
            keypoints_3d_opt = keypoints_3d_opt[:, :self.num_joints]
            opt_error_in_batch = torch.sqrt(((keypoints_3d_opt - keypoints_3d_gt)**2).sum(-1)).mean(1)
            return pred_error_in_batch.detach(), opt_error_in_batch.detach()
        else:
            return pred_error_in_batch.detach(), None


    def reconstruction_error_in_batch(self, keypoints_3d_gt, keypoints_3d_pred):
        # keypoints_3d_pred = keypoints_3d_pred[:, 25:][:, constants.SMPL_TO_H36].clone().detach()

        if keypoints_3d_gt.size(-1) == 4:
            conf = keypoints_3d_gt[:, :, -1:]
            keypoints_3d_pred = keypoints_3d_pred * conf
            keypoints_3d_gt = keypoints_3d_gt[:, :, :-1].clone().detach()

        keypoints_3d_gt, keypoints_3d_pred, _ = self.align_two_joints(keypoints_3d_gt, keypoints_3d_pred)

        gt = keypoints_3d_gt[:, :self.num_joints].cpu().detach().numpy()
        pred = keypoints_3d_pred[:, :self.num_joints].cpu().detach().numpy()

        pred_hat = np.zeros_like(pred)
        batch_size = keypoints_3d_gt.shape[0]
        for b in range(batch_size):
            pred_hat[b] = compute_similarity_transform(pred[b], gt[b])

        error = np.sqrt( ((pred_hat - gt)**2).sum(axis=-1)).mean(axis=-1)

        return error.mean()


def build_evaluator(args, device):
    return Evaluator(args.regressor, device)
