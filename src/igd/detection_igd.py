import time

import numpy as np
import trimesh
from scipy import ndimage
import torch

#from vgn import vis
from igd.grasp import *
from igd.utils.transform import Transform, Rotation
from igd.networks import load_network
from igd.utils import visual
from igd.utils.implicit import as_mesh
import matplotlib.pyplot as plt

LOW_TH = 0.4

class IGD(object):
    def __init__(self, model_path, model_type, best=False, force_detection=False, qual_th=0.9, out_th=0.5, visualize=False, resolution=40, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.net = load_network(model_path, self.device, model_type=model_type)
        self.net.eval()
        self.qual_th = qual_th
        self.best = best
        self.force_detection = force_detection
        self.out_th = out_th
        self.visualize = visualize
        
        self.resolution = resolution
        x, y, z = torch.meshgrid(torch.linspace(start=-0.5, end=0.5 - 1.0 / self.resolution, steps=self.resolution), torch.linspace(start=-0.5, end=0.5 - 1.0 / self.resolution, steps=self.resolution), torch.linspace(start=-0.5, end=0.5 - 1.0 / self.resolution, steps=self.resolution))
        pos = torch.stack((x, y, z), dim=-1).float().unsqueeze(0).to(self.device)
        self.pos = pos.view(1, self.resolution * self.resolution * self.resolution, 3)
        self.sample_points = None

    def __call__(self, state, scene_mesh=None, aff_kwargs={}):
        if hasattr(state, 'tsdf_process'):
            tsdf_process = state.tsdf_process
        else:
            tsdf_process = state.tsdf

        if isinstance(state.tsdf, np.ndarray):
            tsdf_vol = state.tsdf
            voxel_size = 0.3 / self.resolution
            size = 0.3
        else:
            tsdf_vol = state.tsdf.get_grid()
            voxel_size = tsdf_process.voxel_size
            tsdf_process = tsdf_process.get_grid()
            size = state.tsdf.size

        tic = time.time()

        qual_vol, rot_vol, width_vol = self.predict(tsdf_vol, self.pos, self.net, self.device)

        qual_vol = qual_vol.reshape((self.resolution, self.resolution, self.resolution))
        rot_vol = rot_vol.reshape((self.resolution, self.resolution, self.resolution, 4))
        width_vol = width_vol.reshape((self.resolution, self.resolution, self.resolution))

        qual_vol, rot_vol, width_vol = process(tsdf_process, qual_vol, rot_vol, width_vol, out_th=self.out_th)
        qual_vol = bound(qual_vol, voxel_size)
        if self.visualize:
            colored_scene_mesh = scene_mesh
            # colored_scene_mesh = visual.affordance_visual(qual_vol, rot_vol, scene_mesh, size, self.resolution, **aff_kwargs)
        self.sample_points = self.net.feature_sampler.sample_points.reshape(qual_vol.shape+(8,3,))
        grasps, scores, sample_points = select(qual_vol.copy(), self.pos.view(self.resolution, self.resolution, self.resolution, 3).cpu(), rot_vol, width_vol, threshold=self.qual_th, force_detection=self.force_detection, max_filter_size=4 if self.visualize else 4, sample_points=self.sample_points)
        toc = time.time() - tic

        grasps, scores = np.asarray(grasps), np.asarray(scores)
        sample_points = np.asarray(sample_points)
        sample_points = (sample_points + 0.5) * size

        new_grasps = []
        if len(grasps) > 0:
            if self.best:
                p = np.arange(len(grasps))
            else:
                p = np.random.permutation(len(grasps))
            for g in grasps[p]:
                pose = g.pose
                pose.translation = (pose.translation + 0.5) * size
                width = g.width * size
                # width = 0.08
                new_grasps.append(Grasp(pose, width))
            scores = scores[p]
        grasps = new_grasps

        self.sample_points = sample_points
        if self.visualize:
            # p_cloud_tri = trimesh.points.PointCloud(np.asarray(state.pc.points))
            # grasp_mesh_list = [visual.grasp2mesh(g, s) for g, s in zip(grasps, scores)]
            # composed_scene = trimesh.Scene(p_cloud_tri)
            # for i, g_mesh in enumerate(grasp_mesh_list):
            #     composed_scene.add_geometry(g_mesh, node_name=f'grasp_{i}')
            return grasps, scores, toc, colored_scene_mesh
        else:
            return grasps, scores, toc
    
    def predict(self, tsdf_vol, pos, net, device):
        assert tsdf_vol.shape == (1, 40, 40, 40)

        # move input to the GPU
        tsdf_vol = torch.from_numpy(tsdf_vol).to(device)

        with torch.no_grad():
            qual_vol, rot_vol, width_vol = net(tsdf_vol, pos)

        qual_vol = qual_vol.detach()
        rot_vol = rot_vol.detach()
        width_vol = width_vol.detach()
        # move output back to the CPU
        qual_vol = qual_vol.cpu().squeeze().numpy()
        rot_vol = rot_vol.cpu().squeeze().numpy()
        width_vol = width_vol.cpu().squeeze().numpy()
        return qual_vol, rot_vol, width_vol

def bound(qual_vol, voxel_size, limit=[0.02, 0.02, 0.055]):
    # avoid grasp out of bound [0.02  0.02  0.055]
    x_lim = int(limit[0] / voxel_size)
    y_lim = int(limit[1] / voxel_size)
    z_lim = int(limit[2] / voxel_size)
    qual_vol[:x_lim] = 0.0
    qual_vol[-x_lim:] = 0.0
    qual_vol[:, :y_lim] = 0.0
    qual_vol[:, -y_lim:] = 0.0
    qual_vol[:, :, :z_lim] = 0.0
    return qual_vol

def bound_with_mask(tsdf_vol, voxel_size, out_th, limit=[0.02, 0.02, 0.055]):
    # avoid grasp out of bound [0.02  0.02  0.055]
    x_lim = int(limit[0] / voxel_size)
    y_lim = int(limit[1] / voxel_size)
    z_lim = int(limit[2] / voxel_size)
    mask = np.ones(tsdf_vol.shape)
    mask[:x_lim] = 0.0
    mask[-x_lim:] = 0.0
    mask[:, :y_lim] = 0.0
    mask[:, -y_lim:] = 0.0
    mask[:, :, :z_lim] = 0.0
    outside_voxels = tsdf_vol > out_th
    inside_voxels = np.logical_and(1e-3 < tsdf_vol, tsdf_vol < out_th)
    valid_voxels = ndimage.morphology.binary_dilation(
        outside_voxels, iterations=2, mask=np.logical_not(inside_voxels)
    )
    mask[valid_voxels == False] = 0.0
    return mask.bool()


def process(
    tsdf_vol,
    qual_vol,
    rot_vol,
    width_vol,
    gaussian_filter_sigma=1.0,
    min_width=0.033,
    max_width=0.233,
    out_th=0.5
):
    tsdf_vol = tsdf_vol.squeeze()

    # smooth quality volume with a Gaussian
    qual_vol = ndimage.gaussian_filter(
        qual_vol, sigma=gaussian_filter_sigma, mode="nearest"
    )

    # mask out voxels too far away from the surface
    outside_voxels = tsdf_vol > out_th
    inside_voxels = np.logical_and(1e-3 < tsdf_vol, tsdf_vol < out_th)
    valid_voxels = ndimage.morphology.binary_dilation(
        outside_voxels, iterations=2, mask=np.logical_not(inside_voxels)
    )
    qual_vol[valid_voxels == False] = 0.0

    # reject voxels with predicted widths that are too small or too large
    qual_vol[np.logical_or(width_vol < min_width, width_vol > max_width)] = 0.0

    return qual_vol, rot_vol, width_vol


def select(qual_vol, center_vol, rot_vol, width_vol, threshold=0.90, max_filter_size=4, force_detection=False, sample_points=None):
    best_only = False
    # if qual_vol[qual_vol < LOW_TH].sum() == qual_vol.sum():
    #     print('no grasp')
    qual_vol[qual_vol < LOW_TH] = 0.0
    if force_detection and (qual_vol >= threshold).sum() == 0:
        best_only = True
    else:
        # threshold on grasp quality
        qual_vol[qual_vol < threshold] = 0.0

    # non maximum suppression
    max_vol = ndimage.maximum_filter(qual_vol, size=max_filter_size)
    qual_vol = np.where(qual_vol == max_vol, qual_vol, 0.0)


    mask = np.where(qual_vol, 1.0, 0.0)
    sample_points_list = []
    # construct grasps
    grasps, scores = [], []
    for index in np.argwhere(mask):
        if sample_points is not None:
            i,j,k = index
            sample_points_list.append(sample_points[i,j,k].detach().cpu().numpy())
        grasp, score = select_index(qual_vol, center_vol, rot_vol, width_vol, index)
        grasps.append(grasp)
        scores.append(score)
    # if len(grasps) == 0:
    #     print()

    sorted_grasps = [grasps[i] for i in reversed(np.argsort(scores))]
    sorted_scores = [scores[i] for i in reversed(np.argsort(scores))]
    if sample_points is not None:
        sorted_sample_points = [sample_points_list[i] for i in reversed(np.argsort(scores))]

    if best_only and len(sorted_grasps) > 0:
        sorted_grasps = [sorted_grasps[0]]
        sorted_scores = [sorted_scores[0]]
        if sample_points is not None:
            sorted_sample_points = [sorted_sample_points[0]]

    if sample_points is not None:
        return sorted_grasps, sorted_scores, sorted_sample_points
    return sorted_grasps, sorted_scores



def select_index(qual_vol, center_vol, rot_vol, width_vol, index):
    i, j, k = index
    score = qual_vol[i, j, k]
    ori = Rotation.from_quat(rot_vol[i, j, k])
    #pos = np.array([i, j, k], dtype=np.float64)
    pos = center_vol[i, j, k].numpy()
    width = width_vol[i, j, k]
    return Grasp(Transform(ori, pos), width), score
