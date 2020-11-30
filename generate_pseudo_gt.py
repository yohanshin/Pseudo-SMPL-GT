from models.smpl import build_body_model
from models.smplify.smplify_3d import SMPLify3D
from data.h36m import setup_human36m_dataloaders
from data.data_utils import prepare_batch
from utils.loss import build_loss_function
from utils.cmd import get_cfg, get_cmd
from utils.geometry import rot6d_to_rotmat
from utils.visualization import *
from cfg import constants

import torch
from torch import nn
import numpy as np
import os

from tqdm import tqdm, trange

render = False

args = get_cmd().parse_args()
cfg, args = get_cfg(args)

device = cfg.DEVICE
dtype = torch.float32
batch_size = cfg.TRAIN.BATCH_SIZE

loss_function = build_loss_function(cfg)

body_model = build_body_model(cfg)

smplify = SMPLify3D(joint_regressor='data/models/J_regressor_extra.npy',
                     model_folder=os.path.join('data/models', 'smpl'),
                     batch_size=batch_size,
                     lr=5e-2,
                     pred_rotmat=False,
                     maxiters=200,
                     optimizer_type='adam')

if cfg.DATASET.TYPE == 'human36m':
    train_dataloader, val_dataloader = setup_human36m_dataloaders(cfg)
else:
    AssertionError, "Pseudo gt generation is only implemented for Human3.6M dataset"

mean_params = np.load("data/dataset/SPIN/data/smpl_mean_params.npz")
init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0).expand(batch_size, -1)
init_betas = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
init_pose = init_pose.to(device=device).expand(batch_size, -1)
init_betas = init_betas.to(device=device).expand(batch_size, -1)
init_pose = rot6d_to_rotmat(init_pose).view(batch_size, 24, 3, 3)
init_orient = init_pose[:, 0].unsqueeze(1)
init_pose = init_pose[:, 1:]

J_regressor_single_batch = torch.from_numpy(np.load(constants.JOINT_REGRESSOR_H36M)).float()

output_file = 'H36M_smpl.npy' if cfg.DATASET.TYPE == 'human36m' else 'MPI-INF_smpl.npy'
if os.path.isfile(output_file):
    output = np.load(output_file, allow_pickle=True)
    opt_pose_output = output['pose']
    opt_betas_output = output['betas']
    opt_orient_output = output['orient']
else:
    opt_pose_output, opt_betas_output, opt_orient_output = None, None, None

iterator = enumerate(train_dataloader)
with tqdm(total = len(train_dataloader), desc='SMPLify-3D', leave=True) as prog_bar:
    for i_iter in range(len(train_dataloader)):
        if opt_pose_output is not None:
            if (i_iter + 1) * batch_size <= opt_pose_output.shape[0]:
                prog_bar.update(1)
                prog_bar.refresh()
                continue
        
        _, batch = next(iterator)
        img, keypoints_3d_gt, proj_matricies = prepare_batch(batch, device, prepare_images=False)

        if keypoints_3d_gt.shape[0] != batch_size:
            batch_size = keypoints_3d_gt.shape[0]
            smplify.batch_size = batch_size
            body_model.batch_size = batch_size
            init_pose = init_pose[:1].expand(batch_size, 23, 3, 3)
            init_betas = init_betas[:1].expand(batch_size, 10)
            init_orient = init_orient[:1].expand(batch_size, 1, 3, 3)

        opt_pose, opt_betas, opt_global_orient = \
                            smplify(init_pose, init_betas, init_orient, 
                                    body_model, keypoints_3d_gt, False, 
                                    'rotmat', batch, device, dtype)

        opt_output = body_model(betas=opt_betas, body_pose=opt_pose, 
                                global_orient=opt_global_orient,  pose2rot=True)

        _mpjpe, _recone = loss_function.eval(opt_output, keypoints_3d_gt)
        
        msg = 'MPJPE: %.2f'%_mpjpe + ' | RECONE: %.2f'%_recone
        prog_bar.set_postfix_str(msg)
        prog_bar.update(1)
        prog_bar.refresh()

        if i_iter == 0:
            opt_pose_output = opt_pose.detach().cpu().numpy()
            opt_betas_output = opt_betas.detach().cpu().numpy()
            opt_orient_output = opt_global_orient.detach().cpu().numpy()
        else:
            opt_pose_output = np.concatenate((opt_pose_output, opt_pose.detach().cpu().numpy()), axis=0)
            opt_betas_output = np.concatenate((opt_betas_output, opt_betas.detach().cpu().numpy()), axis=0)
            opt_orient_output = np.concatenate((opt_orient_output, opt_global_orient.detach().cpu().numpy()), axis=0)

        output_dtype = np.dtype([('pose', opt_pose_output.dtype, opt_pose_output.shape[1:]), 
                                 ('betas', opt_betas_output.dtype, opt_betas_output.shape[1:]),
                                 ('orient', opt_orient_output.dtype, opt_orient_output.shape[1:])])

        output = np.empty(opt_pose_output.shape[0], dtype=output_dtype)
        output['pose'] = opt_pose_output
        output['betas'] = opt_betas_output
        output['orient'] = opt_orient_output

        if render:
            generate_figure(batch['cameras'], opt_output, body_model, batch['images_before_norm'], 
                            keypoints_3d_gt, iters=i_iter, save_org=False)
        
        np.save(output_file, output)