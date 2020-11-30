from models.smplify.prior import *
from cfg import constants

import torch
from torch import nn
import numpy as np


class ShapeFittingLoss(nn.Module):
    def __init__(self,
                 body_segments_length,
                 data_type='human36m',
                 rho=100,
                 body_segments_idx=None,
                 shape_prior=None,
                 joint_dist_weight=0.0,
                 shape_prior_weight=0.0,
                 dtype=torch.float32,
                 device='cuda:0',
                 **kwargs):

        self.body_segments_length = body_segments_length
        self.data_type = data_type
        self.rho = rho
        self.body_segments_idx = body_segments_idx

        self.shape_prior = shape_prior
        self.joint_dist_weight = joint_dist_weight
        self.shape_prior_weight = shape_prior_weight

        self.dtype = dtype
        self.device = device
        
        self.register_buffer('joint_dist_weight',
                             torch.tensor(joint_dist_weight, dtype=dtype))
        self.register_buffer('shape_prior_weight',
                             torch.tensor(shape_prior_weight, dtype=dtype))
        self.to(device=device)

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                if 'torch.Tensor' in str(type(loss_weight_dict[key])):
                    weight_tensor = loss_weight_dict[key].clone().detach()
                else:
                    weight_tensor = torch.tensor(loss_weight_dict[key],
                                                 dtype=weight_tensor.dtype,
                                                 device=weight_tensor.device)
                setattr(self, key, weight_tensor)

    def forward(self, body_model_output, **kwargs):
        pred_joints = body_model_output.joints[:, 25:][:, constants.SMPL_TO_H36]
        
        # Loss 3 : Shape prior loss
        sprior_loss = torch.sum(self.shape_prior(
            body_model_output.betas)) * self.shape_prior_weight ** 2
        
        pred_diff, gt_diff = [], []
        for part, idxs in self.body_segments_idx.items():
            pred_diff.append(pred_joints[:, idxs[0]] - pred_joints[:, idxs[1]])
        
        pred_diff = torch.stack(pred_diff)
        gt_diff = torch.from_numpy(self.body_segments_idx)
        gt_diff = gt_diff.to(device=pred_diff.device).to(dtype=pred_diff.dtype)
        
        pred_length = pred_diff.pow(2).sum(dim=-1).sqrt()
        gt_length = gt_diff.pow(2).sum(dim=-1).sqrt()
        
        length_diff = pred_length - gt_length
        length_diff = length_diff.pow(2)
        segs_length_loss = length_diff * self.shape_prior_weight
        
        total_loss = sprior_loss + segs_length_loss

        return total_loss


class SMPLifyLoss(nn.Module):
    def __init__(self,
                 data_type='human36m',
                 rho=100, 
                 body_segments_idx=None,
                 body_pose_prior=None,
                 shape_prior=None,
                 angle_prior=None,
                 align_two_joints=True,
                 use_joint_conf=True,
                 dtype=torch.float32,
                 device='cuda:0',
                 joint_dist_weight=0.0,
                 body_pose_weight=0.0,
                 shape_prior_weight=0.0,
                 bending_prior_weight=0.0,
                 **kwargs):
        super(SMPLifyLoss, self).__init__()
        
        self.J_regressor = torch.from_numpy(np.load(constants.JOINT_REGRESSOR_H36M)).float()
        
        self.data_type = data_type
        self.rho = rho
        self.body_segments_idx = body_segments_idx

        self.body_pose_prior = body_pose_prior
        self.shape_prior = shape_prior
        self.angle_prior = angle_prior
        
        self.align_two_joints = align_two_joints
        self.use_joint_conf = use_joint_conf
        
        self.dtype = dtype
        self.device = device
        
        self.register_buffer('joint_dist_weight',
                             torch.tensor(joint_dist_weight, dtype=dtype))
        self.register_buffer('body_pose_weight',
                             torch.tensor(body_pose_weight, dtype=dtype))
        self.register_buffer('shape_prior_weight',
                             torch.tensor(shape_prior_weight, dtype=dtype))
        self.register_buffer('bending_prior_weight',
                             torch.tensor(bending_prior_weight, dtype=dtype))
        self.to(device=device)

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                if 'torch.Tensor' in str(type(loss_weight_dict[key])):
                    weight_tensor = loss_weight_dict[key].clone().detach()
                else:
                    weight_tensor = torch.tensor(loss_weight_dict[key],
                                                 dtype=weight_tensor.dtype,
                                                 device=weight_tensor.device)
                setattr(self, key, weight_tensor)


    def forward(self, 
                body_model_output, 
                gt_joints, 
                ign_joint_idx=None,
                joint_conf=None,
                use_vposer=False,
                pose_embedding=None,
                vposer=None,
                optim_type='pose',
                **kwargs):
        
        def robustifier(value):
            dist = torch.div(value**2, value**2 + self.rho ** 2)
            return self.rho ** 2 * dist

        def centering_joints(joints):
            if self.data_type == 'human36m':
                sacrum_center = joints[:, 6]    
            elif self.data_type == 'cmu':
                lpelvis, rpelvis = joints[:, 6].clone(), joints[:, 12].clone()
                sacrum_center = (lpelvis + rpelvis)/2
            
            joints_ = joints - sacrum_center.unsqueeze(1)

            return joints_

        def align_two_joints(gt_joints, pred_joints):
            # gt_joints = gt_joints[:, :, [0, 2, 1]]
            gt_joints = centering_joints(gt_joints)/1000

            flip = torch.tensor([1, -1, 1], 
                device=pred_joints.device, dtype=pred_joints.dtype)
            # pred_joints = pred_joints * flip
            pred_joints = centering_joints(pred_joints)

            return gt_joints, pred_joints

        # def align_two_joints(self, gt_joints, pred_joints):
        #     if self.data_type == 'human36m':
        #         # gt_joints = gt_joints[:, :, [0, 2, 1]]
        #         flip = torch.tensor([1, -1, 1], 
        #             device=pred_joints.device, dtype=pred_joints.dtype)
        #     else:
        #         flip = torch.tensor([1, -1, 1],
        #             device=pred_joints.device, dtype=pred_joints.dtype)

        #     gt_joints = self.centering_joints(gt_joints)
        #     # pred_joints = pred_joints * flip
        #     pred_joints = self.centering_joints(pred_joints)

        #     if opt_joints is not None:
        #         flip = torch.tensor([1, -1, 1], 
        #             device=opt_joints.device, dtype=opt_joints.dtype)
        #         # opt_joints = opt_joints * flip
        #         opt_joints = self.centering_joints(opt_joints)

        #     return gt_joints, pred_joints, opt_joints
        J_regressor = self.J_regressor[None, :].expand(gt_joints.shape[0], -1, -1).to(self.device)

        joint_weight = torch.ones(gt_joints.shape[-2], dtype=self.dtype, device=self.device)
        if ign_joint_idx is not None:
            joint_weight[ign_joint_idx] = 0
        if joint_conf is not None and self.use_joint_conf:
            joint_weight = joint_weight * joint_conf
        joint_weight.unsqueeze_(-1)

        # pred_joints = body_model_output.joints[:, 25:][:, constants.SMPL_TO_H36]
        pred_joints = torch.matmul(J_regressor, body_model_output.vertices)[:, constants.H36M_TO_J17, :]

        if self.align_two_joints:
            gt_joints, pred_joints = align_two_joints(gt_joints, pred_joints)

        if optim_type == 'pose':
            # Loss 1 : Joint distance loss
            joint_dist = robustifier(gt_joints - pred_joints)
            joint_loss = torch.sum(joint_dist * joint_weight ** 2) * self.joint_dist_weight
            sprior_loss = torch.sum(self.shape_prior(
                body_model_output.betas)) * self.shape_prior_weight ** 2

            # Loss 2 : Pose prior loss
            if use_vposer:
                pprior_loss = (pose_embedding.pow(2).sum() *
                            self.body_pose_weight ** 2)
            else:
                pprior_loss = self.body_pose_weight ** 2 * self.body_pose_prior(
                    body_model_output.body_pose, body_model_output.betas)

            total_loss = joint_loss + pprior_loss.sum() + sprior_loss

        elif optim_type == 'shape':
            # Loss 3 : Shape prior loss
            sprior_loss = torch.sum(self.shape_prior(
                body_model_output.betas)) * self.shape_prior_weight ** 2
            
            pred_diff, gt_diff, segs_conf = [], [], []
            for part, idxs in self.body_segments_idx.items():
                pred_diff.append(pred_joints[:, idxs[0]] - pred_joints[:, idxs[1]])
                gt_diff.append(gt_joints[:, idxs[0]] - gt_joints[:, idxs[1]])
                segs_conf.append(joint_weight[:, idxs].prod(dim=-2))

            pred_diff = torch.stack(pred_diff)
            gt_diff = torch.stack(gt_diff)
            segs_conf = torch.stack(segs_conf)
            
            pred_length = pred_diff.pow(2).sum(dim=-1).sqrt()
            gt_length = gt_diff.pow(2).sum(dim=-1).sqrt()
            
            length_diff = pred_length - gt_length
            length_diff = (length_diff.pow(2) * segs_conf[:, :, 0]).sum().sqrt()
            segs_length_loss = length_diff * self.shape_prior_weight
            
            total_loss = sprior_loss + segs_length_loss
        
        return total_loss


class JointLoss(nn.Module):
    def __init__(self, 
                 rho=100,
                 body_pose_prior=None,
                 use_joint_conf=True,
                 dtype=torch.float32,
                 device='cuda:0',
                 body_pose_weight=0.0,
                 **kwargs):
        super(JointLoss, self).__init__()

        self.rho = rho
        self.body_pose_prior = body_pose_prior

        self.use_joint_conf = use_joint_conf


    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                if 'torch.Tensor' in str(type(loss_weight_dict[key])):
                    weight_tensor = loss_weight_dict[key].clone().detach()
                else:
                    weight_tensor = torch.tensor(loss_weight_dict[key],
                                                 dtype=weight_tensor.dtype,
                                                 device=weight_tensor.device)
                setattr(self, key, weight_tensor)


def build_loss_function(loss_type='smplify', body_segments_length=None, **kwargs):
    if body_segments_length is None:
        return SMPLifyLoss(**kwargs)
    else:
        return ShapeFittingLoss(body_segments_length, **kwargs)