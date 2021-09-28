from models.smpl import build_body_model
from models.smplify.smplify_3d import build_smplify3d
from data.h36m import setup_human36m_dloader
from utils.evaluator import build_evaluator
from utils.geometry import rot6d_to_rotmat
from utils.visualization import *

import torch
from torch import nn
import numpy as np
import os

import argparse
from tqdm import tqdm, trange

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-pth', default='dataset/human36m/processed')
    parser.add_argument('--label-pth', default='label_multiview.npy')
    parser.add_argument('--outfile', default='human36m_smpl.npy')

    parser.add_argument('--batch-size', default=16)
    parser.add_argument('--num-workers', default=16)

    parser.add_argument('--model-fldr', default='dataset/body_models/smpl/')
    parser.add_argument('--regressor', default='dataset/body_models/J_regressor_h36m.npy')
    parser.add_argument('--mean-params', default='dataset/body_models/smpl_mean_params.npz')
    parser.add_argument('--prior-pth', default='dataset/body_models/')

    parser.add_argument('--lr', default=5e-2)
    parser.add_argument('--maxiters', default=200)
    parser.add_argument('--optim-type', default='adam')

    parser.add_argument('--viz-proc', default=False, action='store_true')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_arguments()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32
    batch_size = args.batch_size

    fitting_evaluator = build_evaluator(args, device)
    body_model = build_body_model(args, device)
    smplify = build_smplify3d(args)
    train_dataloader = setup_human36m_dloader(args)

    mean_params = np.load(args.mean_params)
    init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0).expand(batch_size, -1)
    init_betas = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
    init_pose = init_pose.to(device=device).expand(batch_size, -1)
    init_betas = init_betas.to(device=device).expand(batch_size, -1)
    init_pose = rot6d_to_rotmat(init_pose).view(batch_size, 24, 3, 3)
    init_orient = init_pose[:, 0].unsqueeze(1)
    init_pose = init_pose[:, 1:]

    J_regressor_single_batch = torch.from_numpy(np.load(args.regressor)).float()

    if os.path.isfile(args.outfile):
        output = np.load(args.outfile, allow_pickle=True)
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
            keypoints_gt = torch.from_numpy(
                np.stack(batch['keypoints'], axis=0)).float().to(device)

            if keypoints_gt.shape[0] != batch_size:
                batch_size = keypoints_gt.shape[0]
                smplify.batch_size = batch_size
                body_model.batch_size = batch_size
                init_pose = init_pose[:1].expand(batch_size, 23, 3, 3)
                init_betas = init_betas[:1].expand(batch_size, 10)
                init_orient = init_orient[:1].expand(batch_size, 1, 3, 3)

            opt_pose, opt_betas, opt_global_orient = \
                                smplify(init_pose, init_betas, init_orient,
                                        body_model, keypoints_gt, False,
                                        'rotmat', batch, device, dtype)

            opt_output = body_model(betas=opt_betas, body_pose=opt_pose,
                                    global_orient=opt_global_orient,  pose2rot=True)

            _mpjpe, _recone = fitting_evaluator(opt_output, keypoints_gt)

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

            if args.viz_proc:
                generate_figure(batch['cameras'], opt_output, body_model, batch['images_before_norm'],
                                keypoints_gt, iters=i_iter, save_org=False)

            np.save(args.outfile, output)
