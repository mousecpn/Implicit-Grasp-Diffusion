import numpy as np
from scipy import ndimage
import torch.utils.data
from pathlib import Path

from igd.io import *
from igd.perception import *
from igd.utils.transform import Rotation, Transform
from igd.utils.implicit import get_scene_from_mesh_pose_list
import trimesh
from urdfpy import URDF, Mesh
import open3d as o3d
import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def to_numpy(x):
    return x.detach().cpu().numpy()

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

class DatasetVoxel(torch.utils.data.Dataset):
    def __init__(self, root, raw_root, num_point=2048, augment=False):
        self.root = root
        self.augment = augment
        self.num_point = num_point
        self.raw_root = raw_root
        self.num_th = 32
        self.df = read_df(raw_root)
        self.size, _, _, _ = read_setup(raw_root)

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, i):
        scene_id = self.df.loc[i, "scene_id"]
        ori = Rotation.from_quat(self.df.loc[i, "qx":"qw"].to_numpy(np.single))
        pos = self.df.loc[i, "x":"z"].to_numpy(np.single)
        width = self.df.loc[i, "width"].astype(np.single)
        label = self.df.loc[i, "label"].astype(np.int64)
        voxel_grid = read_voxel_grid(self.root, scene_id)
        
        if self.augment:
            voxel_grid, ori, pos = apply_transform(voxel_grid, ori, pos)
        
        pos = pos / self.size - 0.5
        width = width / self.size

        rotations = np.empty((2, 4), dtype=np.single)
        R = Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, 1.0])
        rotations[0] = ori.as_quat()
        rotations[1] = (ori * R).as_quat()

        x, y = voxel_grid[0], (label, rotations, width)

        return x, y, pos

    def get_mesh(self, idx):
        scene_id = self.df.loc[idx, "scene_id"]
        mesh_pose_list_path = self.raw_root / 'mesh_pose_list' / (scene_id + '.npz')
        mesh_pose_list = np.load(mesh_pose_list_path, allow_pickle=True)['pc']
        scene = get_scene_from_mesh_pose_list(mesh_pose_list, return_list=False)
        return scene


class DatasetVoxelOccFile(torch.utils.data.Dataset):
    def __init__(self, root, raw_root, num_point=2048, num_point_occ=2048, augment=False, load_occ=True):
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
            # self.meshes_o3d.append(mesh)
        self.bias = np.array([0.0, -0.0047, 0.033])
        # self.meshes_o3d[0].as_open3d.get_center()

        # self.meshes_o3d[0].translate(np.array([0.0,0.0,0.04]))
        # self.meshes_o3d[1].apply_translation(np.array([0.0,0.0,0.0584]))
        # self.meshes_o3d[2].apply_translation(np.array([0.0,0.0,0.0584]))
        # self.voxelgrid = o3d.pipelines.integration.UniformTSDFVolume(
        #     length=0.3,
        #     resolution=40,
        #     sdf_trunc=4*0.3/40,
        #     color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
        # )
        self.load_occ = load_occ

    def __len__(self):
        return len(self.df.index)
    
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
        # samp le_type = np.random.choice([0,1,2])
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

    def __getitem__(self, i):
        scene_id = self.df.loc[i, "scene_id"]
        ori = Rotation.from_quat(self.df.loc[i, "qx":"qw"].to_numpy(np.single))
        pos = self.df.loc[i, "x":"z"].to_numpy(np.single)
        width = self.df.loc[i, "width"].astype(np.single)
        label = self.df.loc[i, "label"].astype(np.int64)
        # t1 = time.time()
        voxel_grid = read_voxel_grid(self.root, scene_id)
        # t2 = time.time()
        if self.load_occ:
            occ_points, occ = self.read_occ(scene_id, self.num_point_occ)

        if self.augment:
            voxel_grid, ori, pos, occ_points = apply_transform(voxel_grid, ori, pos, occ_points, self.size)

        # self.visualize_gripper(i, width, pos, ori, label)
        # ori, pos, width = self.resample(ori, pos, width)

        # self.visualize_gripper(i, width, pos, ori, label)
        
        # scene = self.get_mesh(i).as_open3d
        # pc = o3d.geometry.PointCloud(scene.sample_points_uniformly(1000))
        # p_cloud_tri = trimesh.points.PointCloud(np.asarray(pc.points))
        # # + grasp_mesh_list
        # scene = trimesh.Scene([p_cloud_tri,])
        # scene.show()
        # if label == 1:
        #     self.print_grasp(voxel_grid, ori, pos)
        
        # if label == 1:
        #     self.print_grasp(voxel_grid_, ori_, pos_)
        
        pos = pos / self.size - 0.5
        width = width / self.size

        rotations = np.empty((2, 4), dtype=np.single)
        R = Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, 1.0])
        rotations[0] = ori.as_quat()
        rotations[1] = (ori * R).as_quat()

        x, y = voxel_grid[0], (label, rotations, width)
        # t3 = time.time()

        if self.load_occ:
            occ_points = occ_points / self.size - 0.5
        
        # print("load occ:", t2-t1)
        # print("process:", t3-t2)

            return x, y, pos, occ_points, occ
        else:
            return x, y, pos


    def print_grasp(self, voxel_grid, ori, pos):
        ax = plt.figure().add_subplot(projection='3d')
        points = np.stack(np.where((voxel_grid.squeeze()>0.0) & (voxel_grid.squeeze()<0.1)),axis=-1)
        # ax.scatter(points[:,0], points[:,1], points[:,2], c='green')

        num_inter_points = 5
        palm_y = np.linspace(-0.04, 0.04, num_inter_points)
        palm_points = np.stack((np.zeros(palm_y.shape), palm_y, np.zeros(palm_y.shape)),axis=-1)
        finger_z = np.linspace(0.0, 0.05, num_inter_points)
        left_finger = np.stack((np.zeros(finger_z.shape), 0.04*np.ones(finger_z.shape), finger_z),axis=-1)
        right_finger = np.stack((np.zeros(finger_z.shape), -0.04*np.ones(finger_z.shape), finger_z),axis=-1)
        control_points = np.concatenate((palm_points, left_finger, right_finger), axis=0)

        R = ori.as_matrix() # (3,3)

        control_points2 = (R @ control_points.transpose(1,0)).transpose(1,0) / self.size * 40

        control_points2 += (pos.reshape(-1,3) / self.size) * 40

        ax.scatter(points[:,0], points[:,1], points[:,2], c='green')
        ax.scatter(control_points2[:,0], control_points2[:,1], control_points2[:,2], c='red', s=30)

        plt.show()

    def print_grasp(self, voxel_grid, ori, pos):
        ax = plt.figure().add_subplot(projection='3d')
        points = np.stack(np.where((voxel_grid.squeeze()>0.0) & (voxel_grid.squeeze()<0.1)),axis=-1)
        # ax.scatter(points[:,0], points[:,1], points[:,2], c='green')

        num_inter_points = 5
        palm_y = np.linspace(-0.04, 0.04, num_inter_points)
        palm_points = np.stack((np.zeros(palm_y.shape), palm_y, np.zeros(palm_y.shape)),axis=-1)
        finger_z = np.linspace(0.0, 0.05, num_inter_points)
        left_finger = np.stack((np.zeros(finger_z.shape), 0.04*np.ones(finger_z.shape), finger_z),axis=-1)
        right_finger = np.stack((np.zeros(finger_z.shape), -0.04*np.ones(finger_z.shape), finger_z),axis=-1)
        control_points = np.concatenate((palm_points, left_finger, right_finger), axis=0)

        R = ori.as_matrix() # (3,3)

        control_points2 = (R @ control_points.transpose(1,0)).transpose(1,0) / self.size * 40

        control_points2 += (pos.reshape(-1,3) / self.size) * 40

        ax.scatter(points[:,0], points[:,1], points[:,2], c='green')
        ax.scatter(control_points2[:,0], control_points2[:,1], control_points2[:,2], c='red', s=30)

        plt.show()
    

    def get_item_in_one_scene(self, i):
        scene_id = self.df.loc[i, "scene_id"]
        mask = self.df.loc[:, "scene_id"] == scene_id

        ori_list = []
        ori_array = self.df.loc[mask, "qx":"qw"].to_numpy(np.single)
        for term in ori_array:
            ori_list.append(Rotation.from_quat(term))
        pos_array = self.df.loc[mask, "x":"z"].to_numpy(np.single)
        width_array = self.df.loc[mask, "width"].to_numpy(np.single)
        label_array = self.df.loc[mask, "label"].to_numpy(np.single).astype(np.int64)
        voxel_grid = read_voxel_grid(self.root, scene_id)

        mesh_list = []
        for j in range(ori_array.shape[0]):
            ori = ori_list[j]
            pos = pos_array[j]
            width = width_array[j]
            label = label_array[j]
            if label == 1:
                scene_mesh, gripper_mesh = self.visualize_gripper(i, width, pos, ori, label, viz=False)
                mesh_list.append(gripper_mesh)
        
                o3d.visualization.draw_geometries([scene_mesh,gripper_mesh], window_name="franka hand", width=800,height=600, left=50, top=50, point_show_normal=False, mesh_show_wireframe=True, mesh_show_back_face=True)
        


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

    def get_mesh(self, idx):
        scene_id = self.df.loc[idx, "scene_id"]
        mesh_pose_list_path = self.raw_root / 'mesh_pose_list' / (scene_id + '.npz')
        mesh_pose_list = np.load(mesh_pose_list_path, allow_pickle=True)['pc']
        scene = get_scene_from_mesh_pose_list(mesh_pose_list, return_list=False)
        return scene
    

    def visualize_gripper(self, i, width, pos, ori, label, viz=True):
        ########visualize gripper##############
        
        mesh_scene = self.get_mesh(i).as_open3d

        self.open_gripper(width)

        gripper_mesh = trimesh.util.concatenate(self.meshes_o3d)
        self.open_gripper(-width)
        # gripper_mesh.apply_translation(-self.bias)
        

        T = np.concatenate((ori.as_matrix(),pos.reshape(-1,1)),axis=1)
        T = np.concatenate((T,np.array([0,0,0,1]).reshape(1,-1)),axis=0)
        gripper_mesh.apply_transform(T)
        bias = T[:3,:3] @ self.bias.reshape(3,1)


        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.06, origin=(gripper_mesh.as_open3d.get_center()+bias.reshape(-1)).tolist())
        gripper_mesh.apply_translation(-bias.reshape(-1))

        # gripper_mesh.rotate(ori.as_matrix())
        # gripper_mesh.translate(pos, relative=False)
        # self.rotate_gripper(ori.as_matrix(),center=self.meshes_o3d[0].get_center())

        # self.translate_gripper(pos-self.bias)
        print('label:',label)

        pc_scene = mesh_scene.sample_points_uniformly(number_of_points=1000)
        pc_gripper = gripper_mesh.as_open3d.sample_points_uniformly(number_of_points=1000)
        # cluster = pc_scene.cluster_dbscan(0.03, 20)
        # cluster = np.asarray(cluster)
        # cluster = (cluster - cluster.min() + np.random.random(1)[0]*0.5)/ (cluster.max() - cluster.min())
        # cluster = np.clip(cluster, a_max=1.,a_min=0.)
        # pc_scene.colors = o3d.utility.Vector3dVector(np.repeat(cluster.reshape(-1,1),3,axis=-1))

        dists = pc_gripper.compute_point_cloud_distance(pc_scene)
        dists = np.asarray(dists)
        ind = np.where(dists < 0.01)[0]

        inlier_cloud = pc_gripper.select_by_index(ind)
        outlier_cloud = pc_gripper.select_by_index(ind, invert=True)

        overlap = (len(inlier_cloud.points))/1000.0
        print("overlap:", overlap)



        # voxel_scene = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh_scene, voxel_size=0.005)
        # voxel_gripper = o3d.geometry.VoxelGrid.create_from_triangle_mesh(gripper_mesh.as_open3d, voxel_size=0.005)
            
        if viz == True:
            o3d.visualization.draw_geometries([pc_scene, pc_gripper], window_name="franka hand", width=800,height=600, left=50, top=50, point_show_normal=False, mesh_show_wireframe=True, mesh_show_back_face=True)
        else:
            return mesh_scene, gripper_mesh.as_open3d

        # o3d.visualization.draw_geometries([mesh_scene,gripper_mesh.as_open3d, mesh_frame], window_name="franka hand", width=800,height=600, left=50, top=50, point_show_normal=False, mesh_show_wireframe=True, mesh_show_back_face=True)
        
        ########visualize gripper##############


def apply_transform(voxel_grid, orientation, position, occ_points, size):
    resolution = voxel_grid.shape[-1]
    position = position / size * resolution
    occ_points = occ_points / size * resolution
    # angle = np.pi / 2.0 * np.random.choice(4)
    angle = 0
    # angle = np.pi / 8.0 * np.random.randn()
    R_augment = Rotation.from_rotvec(np.r_[0.0, 0.0, angle])

    # z_offset = np.random.uniform(6, 34) - position[2]
    # z_offset = 0
    z_offset = np.random.uniform(-3, 3) 

    t_augment = np.r_[0.0, 0.0, z_offset]
    T_augment = Transform(R_augment, t_augment)

    T_center = Transform(Rotation.identity(), np.r_[20.0, 20.0, 20.0])
    T = T_center * T_augment * T_center.inverse()

    # transform voxel grid
    T_inv = T.inverse()
    matrix, offset = T_inv.rotation.as_matrix(), T_inv.translation
    voxel_grid[0] = ndimage.affine_transform(voxel_grid[0], matrix, offset, order=0)

    # transform grasp pose
    position = T.transform_point(position)
    occ_points = T.transform_point(occ_points)
    orientation = T.rotation * orientation
    position = position * size / resolution
    occ_points = occ_points * size / resolution
    return voxel_grid, orientation, position, occ_points

def sample_point_cloud(pc, num_point, return_idx=False):
    num_point_all = pc.shape[0]
    idxs = np.random.choice(np.arange(num_point_all), size=(num_point,), replace=num_point > num_point_all)
    if return_idx:
        return pc[idxs], idxs
    else:
        return pc[idxs]

if __name__=='__main__':
    dataset = DatasetVoxelOccFile(Path("/home/pinhao/Desktop/GIGA/data/data_pile_train_processed_dex_noise"), Path("/home/pinhao/Desktop/GIGA/data/data_pile_train_raw"), augment=True)

    # mesh_scene = dataset.get_mesh(0)
    # panda_hand = URDF.load("/home/pinhao/Desktop/GIGA/data/urdfs/panda/hand.urdf")
    # for joint in panda_hand.actuated_joints:
    #     print(joint)

    # fk = panda_hand.collision_trimesh_fk()
    # meshes = list(fk.keys())
    # meshes_o3d = []
    # for mesh in meshes:
    #     meshes_o3d.append(mesh.as_open3d)
    # meshes_o3d[1].translate(np.array([0.0,0.1,0.0584]))
    # meshes_o3d[2].translate(np.array([0.0,0.0,0.0584]))
    # all_mesh = Mesh("/home/pinhao/Desktop/GIGA/data/urdfs/panda/hand.urdf", meshes=meshes)
    # o3d.visualization.draw_geometries([mesh_scene.as_open3d], window_name="franka hand", width=800,height=600, left=50, top=50, point_show_normal=False, mesh_show_wireframe=True, mesh_show_back_face=True)

    # o3d.visualization.draw_geometries(meshes_o3d, window_name="franka hand", width=800,height=600, left=50, top=50, point_show_normal=False, mesh_show_wireframe=True, mesh_show_back_face=True)
    # for i in range(len(dataset)):
    #     dataset.get_item_in_one_scene(i)
    # train_loader = torch.utils.data.DataLoader(
    #     dataset, batch_size=128, shuffle=True, drop_last=True, num_workers=64, pin_memory=True
    # )
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=True, drop_last=True, pin_memory=True, num_workers=32, persistent_workers=True
    )
    t = time.time()
    for data in train_loader:
        print("time:", time.time()-t)
        t = time.time()
    # for data in dataset:
    #     print()
        # panda_hand.show({'panda_finger_joint1':0.04, 'panda_finger_joint2':0.04} )