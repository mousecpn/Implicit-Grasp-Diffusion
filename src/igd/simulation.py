from pathlib import Path
import time

import numpy as np
import pybullet

from igd.grasp import Label
from igd.perception import *
from igd.utils import btsim, workspace_lines
from igd.utils.transform import Rotation, Transform
from igd.utils.misc import apply_noise, apply_translational_noise,apply_dex_noise
import matplotlib.pyplot as plt
import cv2

def create_point_cloud_from_depth_image(depth, camera, organized=False):
    """ Generate point cloud using depth image only.

        Input:
            depth: [numpy.ndarray, (H,W), numpy.float32]
                depth image
            camera: [CameraInfo]
                camera intrinsics
            organized: bool
                whether to keep the cloud in image shape (H,W,3)

        Output:
            cloud: [numpy.ndarray, (H,W,3)/(H*W,3), numpy.float32]
                generated cloud, (H,W,3) for organized=True, (H*W,3) for organized=False
    """
    assert(depth.shape[0] == camera.height and depth.shape[1] == camera.width)
    xmap = np.arange(camera.width)
    ymap = np.arange(camera.height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth / camera.scale
    points_x = (xmap - camera.cx) * points_z / camera.fx
    points_y = (ymap - camera.cy) * points_z / camera.fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    if not organized:
        cloud = cloud.reshape([-1, 3])
    return cloud

class ClutterRemovalSim(object):
    def __init__(self, scene, object_set, gui=True, seed=None, add_noise=False, sideview=False, save_dir=None, save_freq=8):
        assert scene in ["pile", "packed", "real_packed","real_pile"]

        self.urdf_root = Path("data/urdfs")
        self.scene = scene
        self.object_set = object_set
        self.discover_objects()

        self.global_scaling = {
            "blocks": 1.67,
            "google": 0.7,
            'google_pile': 0.7,
            'google_packed': 0.7,
            
        }.get(object_set, 1.0)
        self.gui = gui
        self.add_noise = add_noise
        self.sideview = sideview

        self.rng = np.random.RandomState(seed) if seed else np.random
        self.world = btsim.BtWorld(self.gui, save_dir, save_freq)
        self.gripper = Gripper(self.world)
        self.size = 6 * self.gripper.finger_depth
        self.intrinsic = CameraIntrinsic(640, 480, 540.0, 540.0, 320.0, 240.0)
        self.camera = self.world.add_camera(self.intrinsic, 0.1, 2.0)
        self.dot_pattern_ = cv2.imread("/home/pinhao/Desktop/simkinect/data/kinect-pattern_3x3.png", 0)
    @property
    def num_objects(self):
        return max(0, self.world.p.getNumBodies() - 1)  # remove table from body count

    def discover_objects(self):
        root = self.urdf_root / self.object_set
        self.object_urdfs = [f for f in root.iterdir() if f.suffix == ".urdf"]

    def save_state(self):
        self._snapshot_id = self.world.save_state()

    def restore_state(self):
        self.world.restore_state(self._snapshot_id)

    def reset(self, object_count):
        self.world.reset()
        self.world.set_gravity([0.0, 0.0, -9.81])
        self.draw_workspace()

        if self.gui:
            self.world.p.resetDebugVisualizerCamera(
                cameraDistance=1.0,
                cameraYaw=0.0,
                cameraPitch=-45,
                cameraTargetPosition=[0.15, 0.50, -0.3],
            )

        table_height = self.gripper.finger_depth
        self.place_table(table_height)

        if self.scene == "pile":
            self.generate_pile_scene(object_count, table_height)
        elif self.scene == "packed":
            self.generate_packed_scene(object_count, table_height)
        elif self.scene == "real_packed":
            self.generate_real_packed_scene(object_count, table_height)
        else:
            raise ValueError("Invalid scene argument")

    def draw_workspace(self):
        points = workspace_lines(self.size)
        color = [0.5, 0.5, 0.5]
        for i in range(0, len(points), 2):
            self.world.p.addUserDebugLine(
                lineFromXYZ=points[i], lineToXYZ=points[i + 1], lineColorRGB=color
            )

    def place_table(self, height):
        urdf = self.urdf_root / "setup" / "plane.urdf"
        pose = Transform(Rotation.identity(), [0.15, 0.15, height])
        self.world.load_urdf(urdf, pose, scale=0.6)

        # define valid volume for sampling grasps
        lx, ux = 0.02, self.size - 0.02
        ly, uy = 0.02, self.size - 0.02
        lz, uz = height + 0.005, self.size
        nlz = 0.005
        self.lower = np.r_[lx, ly, lz]
        self.newlower = np.r_[lx, ly, nlz]
        self.upper = np.r_[ux, uy, uz]

    def generate_pile_scene(self, object_count, table_height):
        # place box
        urdf = self.urdf_root / "setup" / "box.urdf"
        pose = Transform(Rotation.identity(), np.r_[0.02, 0.02, table_height])
        box = self.world.load_urdf(urdf, pose, scale=1.3)

        # drop objects
        urdfs = self.rng.choice(self.object_urdfs, size=object_count)
        for urdf in urdfs:
            rotation = Rotation.random(random_state=self.rng)
            xy = self.rng.uniform(1.0 / 3.0 * self.size, 2.0 / 3.0 * self.size, 2)
            pose = Transform(rotation, np.r_[xy, table_height + 0.2])
            scale = self.rng.uniform(0.8, 1.0)
            self.world.load_urdf(urdf, pose, scale=self.global_scaling * scale)
            self.wait_for_objects_to_rest(timeout=1.0)

        # remove box
        self.world.remove_body(box)
        self.remove_and_wait()

    def generate_packed_scene(self, object_count, table_height):
        attempts = 0
        max_attempts = 12

        while self.num_objects < object_count and attempts < max_attempts:
            self.save_state()
            urdf = self.rng.choice(self.object_urdfs)
            x = self.rng.uniform(0.08, 0.22)
            y = self.rng.uniform(0.08, 0.22)
            z = 1.0
            angle = self.rng.uniform(0.0, 2.0 * np.pi)
            rotation = Rotation.from_rotvec(angle * np.r_[0.0, 0.0, 1.0])
            pose = Transform(rotation, np.r_[x, y, z])
            scale = self.rng.uniform(0.7, 0.9)
            body = self.world.load_urdf(urdf, pose, scale=self.global_scaling * scale)
            lower, upper = self.world.p.getAABB(body.uid)
            z = table_height + 0.5 * (upper[2] - lower[2]) + 0.002
            body.set_pose(pose=Transform(rotation, np.r_[x, y, z]))
            self.world.step()

            if self.world.get_contacts(body):
                self.world.remove_body(body)
                self.restore_state()
            else:
                self.remove_and_wait()
            attempts += 1
    
    def generate_real_packed_scene(self, object_count, table_height):
        attempts = 0
        max_attempts = 12
        object_list = [
            "/home/pinhao/Desktop/GIGA/data/urdfs/packed/test/wooden_block.urdf",
            "/home/pinhao/Desktop/GIGA/data/urdfs/packed/test/pudding_box.urdf",
            "/home/pinhao/Desktop/GIGA/data/urdfs/packed/test/chips_can.urdf",
            "/home/pinhao/Desktop/GIGA/data/urdfs/packed/train/sugar_box.urdf",
            "/home/pinhao/Desktop/GIGA/data/urdfs/packed/train/cheez_it_white_cheddar.urdf",
            "/home/pinhao/Desktop/GIGA/data/urdfs/pile/train/box_poisson_006.urdf",
            "/home/pinhao/Desktop/GIGA/data/urdfs/pile/train/BlueSaltCube_800_tex.urdf",
            "/home/pinhao/Desktop/GIGA/data/urdfs/blocks/cube.urdf",
            "/home/pinhao/Desktop/GIGA/data/urdfs/blocks/cylinder.urdf"
        ]
        scale_dict = {
            "/home/pinhao/Desktop/GIGA/data/urdfs/packed/test/wooden_block.urdf": 0.45,
            "/home/pinhao/Desktop/GIGA/data/urdfs/packed/test/pudding_box.urdf": 1.33,
            "/home/pinhao/Desktop/GIGA/data/urdfs/packed/test/chips_can.urdf": 0.48,
            "/home/pinhao/Desktop/GIGA/data/urdfs/packed/train/sugar_box.urdf":0.6,
            "/home/pinhao/Desktop/GIGA/data/urdfs/packed/train/cheez_it_white_cheddar.urdf":0.45,
            "/home/pinhao/Desktop/GIGA/data/urdfs/pile/train/box_poisson_006.urdf":1,
            "/home/pinhao/Desktop/GIGA/data/urdfs/pile/train/BlueSaltCube_800_tex.urdf":1.1,
            "/home/pinhao/Desktop/GIGA/data/urdfs/blocks/cube.urdf":1.66,
            "/home/pinhao/Desktop/GIGA/data/urdfs/blocks/cylinder.urdf":1,
        }

        while self.num_objects < object_count and attempts < max_attempts:
            self.save_state()

            urdf = self.rng.choice(object_list)
            scale = scale_dict[urdf]
            x = self.rng.uniform(0.08, 0.22)
            y = self.rng.uniform(0.08, 0.22)
            z = 1.0
            # angle = 0#0.23*np.pi
            angle = self.rng.uniform(0.0, 2.0 * np.pi)
            rotation = Rotation.from_rotvec(angle * np.r_[0.0, 0.0, 1.0])
            pose = Transform(rotation, np.r_[x, y, z])
            body = self.world.load_urdf(urdf, pose, scale=self.global_scaling * scale)
            self.world.p.changeDynamics(body.uid, -1, lateralFriction=0.6)
            lower, upper = self.world.p.getAABB(body.uid)
            z = table_height + 0.5 * (upper[2] - lower[2]) + 0.002
            body.set_pose(pose=Transform(rotation, np.r_[x, y, z]))
            self.world.step()
            # lower_, upper_ = self.world.p.getAABB(body.uid)
            # size = np.array(upper_) - np.array(lower_)
            # print(size)

            if self.world.get_contacts(body):
                self.world.remove_body(body)
                self.restore_state()
            else:
                self.remove_and_wait()
            attempts += 1
    
    def recovered_scene(self, mesh_list):
        # texture_id = world.p.loadTexture('/home/pinhao/Desktop/GIGA/texture_0.jpg')
        self.world.reset()
        self.world.set_gravity([0.0, 0.0, -9.81])
        self.draw_workspace()

        if self.gui:
            self.world.p.resetDebugVisualizerCamera(
                cameraDistance=1.0,
                cameraYaw=0.0,
                cameraPitch=-45,
                cameraTargetPosition=[0.15, 0.50, -0.3],
            )

        table_height = self.gripper.finger_depth
        self.place_table(table_height)
        for (mesh_path, scale, pose) in mesh_list:
            pose = Transform.from_matrix(pose)
            mesh_path = '_'.join(mesh_path.split('_')[:-1])+'.urdf'
            body = self.world.load_urdf(mesh_path, pose, scale=self.global_scaling * scale)
            lower, upper = self.world.p.getAABB(body.uid)
            # body.set_pose(pose=pose)
            self.world.step()


    def acquire_tsdf(self, n, N=None, resolution=40):
        """Render synthetic depth images from n viewpoints and integrate into a TSDF.

        If N is None, the n viewpoints are equally distributed on circular trajectory.

        If N is given, the first n viewpoints on a circular trajectory consisting of N points are rendered.
        """
        tsdf = TSDFVolume(self.size, resolution)
        high_res_tsdf = TSDFVolume(self.size, 120)

        if self.sideview:
            origin = Transform(Rotation.identity(), np.r_[self.size / 2, self.size / 2, self.size / 3])
            theta = np.pi / 3.0
        else:
            origin = Transform(Rotation.identity(), np.r_[self.size / 2, self.size / 2, 0])
            theta = np.pi / 6.0
        r = 1.0 * self.size

        N = N if N else n
        if self.sideview:
            assert n == 1
            # phi_list = [0.0]
            phi_list = [- np.pi / 2.0]
        else:
            phi_list = 2.0 * np.pi * np.arange(n) / N
        extrinsics = [camera_on_sphere(origin, r, theta, phi) for phi in phi_list]

        for extrinsic in extrinsics:
            depth_img = self.camera.render(extrinsic)[1]
  
            # add noise 
            depth_img = apply_noise(depth_img, self.add_noise)

            tic = time.time()
            tsdf.integrate(depth_img, self.camera.intrinsic, extrinsic)
            timing += time.time() - tic
            high_res_tsdf.integrate(depth_img, self.camera.intrinsic, extrinsic)
        bounding_box = o3d.geometry.AxisAlignedBoundingBox(self.lower, self.upper)
        pc = high_res_tsdf.get_cloud()
        pc = pc.crop(bounding_box)

        return tsdf, pc, timing

    def execute_grasp(self, grasp, remove=True, allow_contact=False):
        T_world_grasp = grasp.pose
        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_world_pregrasp = T_world_grasp * T_grasp_pregrasp

        approach = T_world_grasp.rotation.as_matrix()[:, 2]
        angle = np.arccos(np.dot(approach, np.r_[0.0, 0.0, -1.0]))
        if angle > np.pi / 3.0:
            # side grasp, lift the object after establishing a grasp
            T_grasp_pregrasp_world = Transform(Rotation.identity(), [0.0, 0.0, 0.1])
            T_world_retreat = T_grasp_pregrasp_world * T_world_grasp
        else:
            T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.0, -0.1])
            T_world_retreat = T_world_grasp * T_grasp_retreat

        self.gripper.reset(T_world_pregrasp)

        if self.gripper.detect_contact():
            result = Label.FAILURE, self.gripper.max_opening_width
        else:
            self.gripper.move_tcp_xyz(T_world_grasp, abort_on_contact=True)
            if self.gripper.detect_contact() and not allow_contact:
                result = Label.FAILURE, self.gripper.max_opening_width
            else:
                self.gripper.move(0.0)
                self.gripper.move_tcp_xyz(T_world_retreat, abort_on_contact=False)
                if self.check_success(self.gripper):
                    result = Label.SUCCESS, self.gripper.read()
                    if remove:
                        contacts = self.world.get_contacts(self.gripper.body)
                        self.world.remove_body(contacts[0].bodyB)
                else:
                    result = Label.FAILURE, self.gripper.max_opening_width

        self.world.remove_body(self.gripper.body)

        if remove:
            self.remove_and_wait()

        return result

    def remove_and_wait(self):
        # wait for objects to rest while removing bodies that fell outside the workspace
        removed_object = True
        while removed_object:
            self.wait_for_objects_to_rest()
            removed_object = self.remove_objects_outside_workspace()

    def wait_for_objects_to_rest(self, timeout=2.0, tol=0.01):
        timeout = self.world.sim_time + timeout
        objects_resting = False
        while not objects_resting and self.world.sim_time < timeout:
            # simulate a quarter of a second
            for _ in range(60):
                self.world.step()
            # check whether all objects are resting
            objects_resting = True
            for _, body in self.world.bodies.items():
                if np.linalg.norm(body.get_velocity()) > tol:
                    objects_resting = False
                    break

    def remove_objects_outside_workspace(self):
        removed_object = False
        for body in list(self.world.bodies.values()):
            xyz = body.get_pose().translation
            if np.any(xyz < 0.0) or np.any(xyz > self.size):
                self.world.remove_body(body)
                removed_object = True
        return removed_object

    def check_success(self, gripper):
        # check that the fingers are in contact with some object and not fully closed
        contacts = self.world.get_contacts(gripper.body)
        res = len(contacts) > 0 and gripper.read() > 0.1 * gripper.max_opening_width
        return res


class Gripper(object):
    """Simulated Panda hand."""

    def __init__(self, world):
        self.world = world
        self.urdf_path = Path("data/urdfs/panda/hand.urdf")

        self.max_opening_width = 0.08
        self.finger_depth = 0.05
        self.T_body_tcp = Transform(Rotation.identity(), [0.0, 0.0, 0.03])
        self.T_tcp_body = self.T_body_tcp.inverse()

    def reset(self, T_world_tcp):
        T_world_body = T_world_tcp * self.T_tcp_body
        self.body = self.world.load_urdf(self.urdf_path, T_world_body)
        self.body.set_pose(T_world_body)  # sets the position of the COM, not URDF link
        self.constraint = self.world.add_constraint(
            self.body,
            None,
            None,
            None,
            pybullet.JOINT_FIXED,
            [0.0, 0.0, 0.0],
            Transform.identity(),
            T_world_body,
        )
        self.update_tcp_constraint(T_world_tcp)
        # constraint to keep fingers centered
        self.world.add_constraint(
            self.body,
            self.body.links["panda_leftfinger"],
            self.body,
            self.body.links["panda_rightfinger"],
            pybullet.JOINT_GEAR,
            [1.0, 0.0, 0.0],
            Transform.identity(),
            Transform.identity(),
        ).change(gearRatio=-1, erp=0.1, maxForce=50)
        self.joint1 = self.body.joints["panda_finger_joint1"]
        self.joint1.set_position(0.5 * self.max_opening_width, kinematics=True)
        self.joint2 = self.body.joints["panda_finger_joint2"]
        self.joint2.set_position(0.5 * self.max_opening_width, kinematics=True)

    def update_tcp_constraint(self, T_world_tcp):
        T_world_body = T_world_tcp * self.T_tcp_body
        self.constraint.change(
            jointChildPivot=T_world_body.translation,
            jointChildFrameOrientation=T_world_body.rotation.as_quat(),
            maxForce=300,
        )

    def set_tcp(self, T_world_tcp):
        T_word_body = T_world_tcp * self.T_tcp_body
        self.body.set_pose(T_word_body)
        self.update_tcp_constraint(T_world_tcp)

    def move_tcp_xyz(self, target, eef_step=0.002, vel=0.10, abort_on_contact=True):
        T_world_body = self.body.get_pose()
        T_world_tcp = T_world_body * self.T_body_tcp

        diff = target.translation - T_world_tcp.translation
        n_steps = int(np.linalg.norm(diff) / eef_step)
        dist_step = diff / n_steps
        dur_step = np.linalg.norm(dist_step) / vel

        for _ in range(n_steps):
            T_world_tcp.translation += dist_step
            self.update_tcp_constraint(T_world_tcp)
            for _ in range(int(dur_step / self.world.dt)):
                self.world.step()
            if abort_on_contact and self.detect_contact():
                return

    def detect_contact(self, threshold=5):
        if self.world.get_contacts(self.body):
            return True
        else:
            return False

    def move(self, width):
        self.joint1.set_position(0.5 * width)
        self.joint2.set_position(0.5 * width)
        for _ in range(int(0.5 / self.world.dt)):
            self.world.step()

    def read(self):
        width = self.joint1.get_position() + self.joint2.get_position()
        return width
