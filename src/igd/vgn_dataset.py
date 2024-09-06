import numpy as np
from scipy import ndimage
import torch.utils.data
from pathlib import Path
# import sys
# sys.path.append("/home/pinhao/Desktop/GIGA/src")
from igd.io import *
from igd.perception import *
from igd.utils.transform import Rotation, Transform
from igd.utils.implicit import get_scene_from_mesh_pose_list
import trimesh
from urdfpy import URDF, Mesh
from igd.dataset_voxel import apply_transform, sample_point_cloud
import torch.nn.functional as F

import open3d as o3d

def farthest_point_sampling(point_cloud, num_samples):
    sampled_indices = []
    pc = np.asarray(point_cloud.points)
    distances = np.ones(len(point_cloud.points))

    # 选择第一个点
    current_index = np.random.choice(len(point_cloud.points))
    sampled_indices.append(current_index)

    for _ in range(1, num_samples):
        current_point = np.asarray(point_cloud.points[current_index])
        # 计算当前点与所有其他点的距离
        distances = np.minimum(distances, np.linalg.norm(np.asarray(point_cloud.points) - current_point, axis=1))
        # 选择距离最远的点
        current_index = np.argmax(distances)
        sampled_indices.append(current_index)

    return sampled_indices

def mesh_to_cloud_signed_distances(o3d_mesh, cloud):
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(o3d_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)
    # o3d.core.Tensor
    sdf = scene.compute_signed_distance(o3d.core.Tensor(cloud.astype('float32')))
    return sdf.numpy()

class SampleDatasetVoxelOccFile(torch.utils.data.Dataset):
    def __init__(self, root, raw_root, num_point=2048, num_point_occ=2048, augment=False, test=False):
        self.root = root
        self.augment = augment
        self.num_point = num_point
        self.num_point_occ = num_point_occ
        self.raw_root = raw_root
        self.num_th = 32
        self.df = read_df(raw_root)
        self.size, _, _, _ = read_setup(raw_root)
        # self.panda_hand = URDF.load("/home/pinhao/Desktop/GIGA/data/urdfs/panda/hand.urdf")
        # fk = self.panda_hand.collision_trimesh_fk()
        # meshes = list(fk.keys())
        # self.meshes_o3d = []
        # for mesh in meshes:
        #     self.meshes_o3d.append(mesh)
        # self.bias = np.array([0.0, -0.0047, 0.033])
        # self.meshes_o3d[0].as_open3d.get_center()

        # self.meshes_o3d[0].translate(np.array([0.0,0.0,0.04]))
        # self.meshes_o3d[1].apply_translation(np.array([0.0,0.0,0.0584]))
        # self.meshes_o3d[2].apply_translation(np.array([0.0,0.0,0.0584]))
        self.scale = 8
        self.n_grasps = 10
        self.test = test

        self.scene_set = set(self.df.loc[:,"scene_id"].to_numpy())

        self.scene_list = list(set(self.df.loc[:,"scene_id"].to_numpy()))

    def __len__(self):
        return len(self.scene_list)
    
    def translate_gripper(self, trans):
        previous_pos = self.meshes_o3d[0].get_center()
        self.meshes_o3d[0].translate(trans,relative=False)
        shift = self.meshes_o3d[0].get_center() - previous_pos
        self.meshes_o3d[1].translate(trans,relative=True)
        return
    
    def rotate_gripper(self, rot, center=None):
        # center = self.meshes_o3d[0].get_center() 
        for mesh in self.meshes_o3d:
            mesh.rotate(rot, center=center)
        return
    
    def open_gripper(self, width):
        self.meshes_o3d[1].apply_translation(np.array([0.0,-width/2,0.0]))
        self.meshes_o3d[2].apply_translation(np.array([0.0,width/2,0.0]))
        return
    
    def resample(self, ori, pos, width):
        # sample_type = np.random.choice([0,1,2])
        sample_type = 1
        trans_perturb_level = 0.1
        rot_perturb_level = 1
        width_perturb_level = 0.02

        ori = ori.as_rotvec()

        if sample_type == 0:
            # translation sampling
            noise = (np.random.rand(pos.size) * 2 - 1) * trans_perturb_level
            pos += noise
        if sample_type == 1:
            # rotation sampling
            noise = (np.random.rand(ori.size) * 2 - 1) * rot_perturb_level
            ori += noise
            ori = Rotation.from_rotvec(ori)
        if sample_type == 2:
            # width sampling
            noise = (np.random.rand(width.size) * 2 - 1) * width_perturb_level
            width += noise
        return ori, pos, width

    """
    def __getitem__(self, i):
        scene_id = self.df.loc[i, "scene_id"]
        ori = Rotation.from_quat(self.df.loc[i, "qx":"qw"].to_numpy(np.single))
        pos = self.df.loc[i, "x":"z"].to_numpy(np.single)
        width = self.df.loc[i, "width"].astype(np.single)
        label = self.df.loc[i, "label"].astype(np.int64)
        voxel_grid = read_voxel_grid(self.root, scene_id)

        # self.visualize_gripper(i, width, pos, ori, label)
        # ori, pos, width = self.resample(ori, pos, width)

        # self.visualize_gripper(i, width, pos, ori, label)


        if self.augment:
            voxel_grid, ori, pos = apply_transform(voxel_grid, ori, pos)
        
        pos = pos / self.size - 0.5
        width = width / self.size

        rotations = np.empty((2, 4), dtype=np.single)
        R = Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, 1.0])
        rotations[0] = ori.as_quat()
        rotations[1] = (ori * R).as_quat()



        x, y = voxel_grid[0], (label, rotations, width)

        occ_points, occ = self.read_occ(scene_id, self.num_point_occ)
        occ_points = occ_points / self.size - 0.5

        return x, y, pos, occ_points, occ
    """

    def __getitem__(self, i):
        scene_id = self.scene_list[i]
        pos_array, ori_list, width_array, label_array, voxel_grid, pcl_ = self.get_item_in_one_scene(scene_id)
        while (label_array==1).sum() == 0:
            i = np.random.randint(low=0, high=self.__len__())
            scene_id = self.scene_list[i]
            # scene_id = self.df.loc[i, "scene_id"]
            pos_array, ori_list, width_array, label_array, voxel_grid, pcl_ = self.get_item_in_one_scene(scene_id)

        occ_points, occ = self.read_occ(scene_id, self.num_point_occ)
        
        occ_points = occ_points / self.size - 0.5
        pos_array = pos_array / self.size - 0.5
        width_array = width_array / self.size

        rot_list = []
        for ori in ori_list:
            rotations = np.empty((2, 4), dtype=np.single)
            R = Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, 1.0])
            rotations[0] = ori.as_quat()
            rotations[1] = (ori * R).as_quat()
            rot_list.append(rotations)
        ori_array = np.stack((rot_list), axis=0)


        pos_array  = pos_array[label_array==1]
        width_array = width_array[label_array==1]
        ori_array = ori_array[label_array==1]
        
        if self.test:
            x, y = voxel_grid[0], (label_array, ori_array, width_array)
        else:
            rix = np.random.randint(low=0, high=ori_array.shape[0], size=self.n_grasps)
            x, y = voxel_grid[0], (label_array[rix], ori_array[rix], width_array[rix])
            pos_array = pos_array[rix]

        return x, y, pos_array, occ_points, occ


    def get_item_in_one_scene(self, scene_id):
        # scene_id = self.df.loc[i, "scene_id"]
        mask = self.df.loc[:, "scene_id"] == scene_id

        ori_list = []
        ori_array = self.df.loc[mask, "qx":"qw"].to_numpy(np.single)
        for term in ori_array:
            ori_list.append(Rotation.from_quat(term))
        # ori_list = np.stack(ori_list,axis=0)
        pos_array = self.df.loc[mask, "x":"z"].to_numpy(np.single)
        width_array = self.df.loc[mask, "width"].to_numpy(np.single)
        label_array = self.df.loc[mask, "label"].to_numpy(np.single).astype(np.int64)
        voxel_grid = read_voxel_grid(self.root, scene_id)
        pcl =  read_point_cloud(self.root, scene_id)

        return pos_array, ori_list, width_array, label_array, voxel_grid, pcl


    def read_occ(self, scene_id, num_point):
        occ_paths = list((self.raw_root / 'occ' / scene_id).glob('*.npz'))
        path_idx = torch.randint(high=len(occ_paths), size=(1,), dtype=int).item()
        occ_path = occ_paths[path_idx]
        occ_data = np.load(occ_path)
        points = occ_data['points']
        occ = occ_data['occ']
        points, idxs = sample_point_cloud(points, num_point, return_idx=True)
        occ = occ[idxs]
        return points, occ

    def get_mesh(self, scene_id):
        # scene_id = self.df.loc[idx, "scene_id"]
        mesh_pose_list_path = self.raw_root / 'mesh_pose_list' / (scene_id + '.npz')
        mesh_pose_list = np.load(mesh_pose_list_path, allow_pickle=True)['pc']
        scene = get_scene_from_mesh_pose_list(mesh_pose_list, return_list=False)
        return scene

class SceneBasedDatasetVoxelOccFile(torch.utils.data.Dataset):
    def __init__(self, root, raw_root, num_point=2048, num_point_occ=2048, augment=False, test=False):
        self.root = root
        self.augment = augment
        self.num_point = num_point
        self.num_point_occ = num_point_occ
        self.raw_root = raw_root
        self.num_th = 32
        self.df = read_df(raw_root)
        self.size, _, _, _ = read_setup(raw_root)
        # self.panda_hand = URDF.load("/home/pinhao/Desktop/GIGA/data/urdfs/panda/hand.urdf")
        # fk = self.panda_hand.collision_trimesh_fk()
        # meshes = list(fk.keys())
        # self.meshes_o3d = []
        # for mesh in meshes:
        #     self.meshes_o3d.append(mesh)
        # self.bias = np.array([0.0, -0.0047, 0.033])
        # self.meshes_o3d[0].as_open3d.get_center()

        # self.meshes_o3d[0].translate(np.array([0.0,0.0,0.04]))
        # self.meshes_o3d[1].apply_translation(np.array([0.0,0.0,0.0584]))
        # self.meshes_o3d[2].apply_translation(np.array([0.0,0.0,0.0584]))
        self.scale = 8
        self.n_grasps = 10
        self.test = test

        self.scene_set = set(self.df.loc[:,"scene_id"].to_numpy())

        self.scene_list = list(set(self.df.loc[:,"scene_id"].to_numpy()))

    def __len__(self):
        return len(self.scene_list)

    def __getitem__(self, i):
        scene_id = self.scene_list[i]
        pos_array, ori_list, width_array, label_array, voxel_grid, pcl_ = self.get_item_in_one_scene(scene_id)
        occ_points, occ = self.read_occ(scene_id, self.num_point_occ)
        
        occ_points = occ_points / self.size - 0.5
        pos_array = pos_array / self.size - 0.5
        width_array = width_array / self.size

        rot_list = []
        for ori in ori_list:
            rotations = np.empty((2, 4), dtype=np.single)
            R = Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, 1.0])
            rotations[0] = ori.as_quat()
            rotations[1] = (ori * R).as_quat()
            rot_list.append(rotations)
        ori_array = np.stack((rot_list), axis=0)


        pos_array  = pos_array
        width_array = width_array
        ori_array = ori_array
        
        if self.test:
            x, y = voxel_grid[0], (label_array, ori_array, width_array)
        else:
            rix = np.random.randint(low=0, high=ori_array.shape[0], size=self.n_grasps)
            x, y = voxel_grid[0], (label_array[rix], ori_array[rix], width_array[rix])
            pos_array = pos_array[rix]

        return x, y, pos_array, occ_points, occ


    def get_item_in_one_scene(self, scene_id):
        # scene_id = self.df.loc[i, "scene_id"]
        mask = self.df.loc[:, "scene_id"] == scene_id

        ori_list = []
        ori_array = self.df.loc[mask, "qx":"qw"].to_numpy(np.single)
        for term in ori_array:
            ori_list.append(Rotation.from_quat(term))
        # ori_list = np.stack(ori_list,axis=0)
        pos_array = self.df.loc[mask, "x":"z"].to_numpy(np.single)
        width_array = self.df.loc[mask, "width"].to_numpy(np.single)
        label_array = self.df.loc[mask, "label"].to_numpy(np.single).astype(np.int64)
        voxel_grid = read_voxel_grid(self.root, scene_id)
        pcl =  read_point_cloud(self.root, scene_id)

        return pos_array, ori_list, width_array, label_array, voxel_grid, pcl


    def read_occ(self, scene_id, num_point):
        occ_paths = list((self.raw_root / 'occ' / scene_id).glob('*.npz'))
        path_idx = torch.randint(high=len(occ_paths), size=(1,), dtype=int).item()
        occ_path = occ_paths[path_idx]
        occ_data = np.load(occ_path)
        points = occ_data['points']
        occ = occ_data['occ']
        points, idxs = sample_point_cloud(points, num_point, return_idx=True)
        occ = occ[idxs]
        return points, occ

    def get_mesh(self, scene_id):
        # scene_id = self.df.loc[idx, "scene_id"]
        mesh_pose_list_path = self.raw_root / 'mesh_pose_list' / (scene_id + '.npz')
        mesh_pose_list = np.load(mesh_pose_list_path, allow_pickle=True)['pc']
        scene = get_scene_from_mesh_pose_list(mesh_pose_list, return_list=False)
        return scene
if __name__=="__main__":
    dataset = SceneBasedDatasetVoxelOccFile(Path("/home/amax/GIGA/data/data_pile_train_processed_dex_noise"), Path("/home/amax/GIGA/data/data_pile_train_raw"))
    for data in dataset:
        print()