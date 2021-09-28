from cfg import constants

import torch
import numpy as np

import trimesh
import pyrender
import cv2

import os




def render_smpl(vertices, faces, image, intrinsics, pose, transl,
                alpha=1.0, filename='render_sample.png'):

    img_size = image.shape[-2]
    material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=(0.8, 0.3, 0.3, 1.0))

    # Generate SMPL vertices mesh
    mesh = trimesh.Trimesh(vertices, faces)

    # Default rotation of SMPL body model
    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)

    mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

    scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
    scene.add(mesh, 'mesh')

    camera_pose = np.eye(4)
    camera_pose[:3, :3] = pose
    camera_pose[:3, 3] = transl
    camera = pyrender.IntrinsicsCamera(fx=intrinsics[0, 0], fy=intrinsics[1, 1],
                                       cx=intrinsics[0, 2], cy=intrinsics[1, 2])
    scene.add(camera, pose=camera_pose)

    # Light information
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
    light_pose = np.eye(4)

    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)

    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)

    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    renderer = pyrender.OffscreenRenderer(
        viewport_width=img_size, viewport_height=img_size, point_size=1.0)

    color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    valid_mask = (rend_depth > 0)[:,:,None]

    color = color.astype(np.float32) / 255.0
    valid_mask = (rend_depth > 0)[:,:,None]
    output_img = color[:, :, :3] * valid_mask + (1 - valid_mask) * image / 255.0

    cv2.imwrite(filename, 255 * output_img)


def generate_figure(camera, pred_output, body_model, images, gt, iters, save_org=False):
    from scipy.spatial.transform import Rotation as R

    betas = pred_output.betas.clone()
    body_pose = pred_output.body_pose.clone()
    glob_ori = pred_output.global_orient.clone()

    if body_pose.shape[-1] != 3:
        body_pose = torch.from_numpy(R.from_rotvec(body_pose.cpu().detach().numpy().reshape(-1, 3)).as_matrix())[None].cuda().float()
        glob_ori = torch.from_numpy(R.from_rotvec(glob_ori.cpu().detach().numpy().reshape(-1, 3)).as_matrix())[None].cuda().float()

    faces = body_model.faces

    gt = gt.detach().cpu().numpy()

    for cam_idx in range(4):
        image = images[0, cam_idx]

        # Get camera information
        camera_info = camera[cam_idx][0]
        pose = camera_info.R
        intrinsics = camera_info.K
        transl = (camera_info.t.reshape(3)) / 1e3
        transl[0] *= -1     # Adjust x-axis translation

        # Change body orientation so that camera matrix to be identical
        rot = torch.from_numpy(pose).to(device=glob_ori.device, dtype=glob_ori.dtype)
        glob_ori_R = rot @ glob_ori

        pred_output_ = body_model(betas=betas, body_pose=body_pose, global_orient=glob_ori_R, pose2rot=False)

        # Match and tranform keypoints gt and pred
        loc_gt = gt @ pose.T
        loc_pred_ = pred_output_.joints[:, 25:][:, constants.SMPL_TO_H36].detach().cpu().numpy() * 1e3
        loc_diff = loc_gt[:, 14] - loc_pred_[:, 14]
        loc_pred_ = loc_pred_ + loc_diff

        vertices = pred_output_.vertices[0].detach().cpu().numpy()
        vertices = vertices + loc_diff / 1e3

        if save_org:
            cv2.imwrite('demo_figs/render_sample_%03d_org_%d.png'%(iters, cam_idx+1), image[:, :, ::-1])
        filename = 'demo_figs/render_sample_%03d_%d.png'%(iters, cam_idx+1)

        image_org = image[:, :, ::-1].copy()
        render_smpl(vertices, faces, image[:, :, ::-1], intrinsics, np.eye(3), transl, filename=filename)
