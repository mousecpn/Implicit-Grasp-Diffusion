import collections
import argparse
from datetime import datetime
import uuid

import numpy as np
import pandas as pd
import tqdm
import io as sysio
import os
from igd import io
from igd.grasp import *
from igd.simulation import ClutterRemovalSim
from igd.utils.transform import Rotation, Transform
from igd.utils.implicit import get_mesh_pose_list_from_world, get_scene_from_mesh_pose_list
from igd.utils import visual
import trimesh
from PIL import Image
import matplotlib.pyplot as plt
import pickle

MAX_CONSECUTIVE_FAILURES = 2

State = collections.namedtuple("State", ["tsdf", "pc"])


def run(
    grasp_plan_fn,
    logdir,
    description,
    scene,
    object_set,
    num_objects=5,
    n=6,
    N=None,
    num_rounds=40,
    seed=1,
    sim_gui=False,
    result_path=None,
    add_noise=False,
    sideview=False,
    resolution=40,
    silence=False,
    visualize=False
):
    """Run several rounds of simulated clutter removal experiments.

    Each round, m objects are randomly placed in a tray. Then, the grasping pipeline is
    run until (a) no objects remain, (b) the planner failed to find a grasp hypothesis,
    or (c) maximum number of consecutive failed grasp attempts.
    """
    #sideview=False
    #n = 6
    sim = ClutterRemovalSim(scene, object_set, gui=sim_gui, seed=seed, add_noise=add_noise, sideview=sideview)
    logger = Logger(logdir, description)
    cnt = 0
    success = 0
    left_objs = 0
    total_objs = 0
    cons_fail = 0
    no_grasp = 0
    planning_times = []
    total_times = []

    # record_file = 'record5.pkl'
    # if os.path.exists(record_file):
    #     with open(record_file, 'rb') as f:
    #         dicts = pickle.load(f)
    #     success_list = dicts['success_list']
    #     object_num_list = dicts['object_num_list']
    # else:
    #     dicts = {}
    #     success_list = []
    #     object_num_list = []

    for _ in tqdm.tqdm(range(num_rounds), disable=silence):
        sim.reset(num_objects)
        sim.save_state()

        round_id = logger.last_round_id() + 1
        logger.log_round(round_id, sim.num_objects)
        # if round_id <5:
        #     continue
        total_objs += sim.num_objects
        # object_num_list.append(sim.num_objects)
        consecutive_failures = 1
        last_label = None
        trial_id = -1
        last_trial = None
        while sim.num_objects > 0 and consecutive_failures < MAX_CONSECUTIVE_FAILURES:
            trial_id += 1
            timings = {}

            # scan the scene
            tsdf, pc, timings["integration"] = sim.acquire_tsdf(n=n, N=N, resolution=40)
            state = argparse.Namespace(tsdf=tsdf, pc=pc)
            if resolution != 40:
                extra_tsdf, _, _ = sim.acquire_tsdf(n=n, N=N, resolution=resolution)
                state.tsdf_process = extra_tsdf
            

            if pc.is_empty():
                # print('pc is empty')
                break  # empty point cloud, abort this round TODO this should not happen

            # plan grasps
            if visualize:
                mesh_pose_list = get_mesh_pose_list_from_world(sim.world, object_set)
                scene_mesh = get_scene_from_mesh_pose_list(mesh_pose_list)
                grasps, scores, timings["planning"], visual_mesh = grasp_plan_fn(state, scene_mesh=scene_mesh)
            else:
                grasps, scores, timings["planning"] = grasp_plan_fn(state, last_trial)
            planning_times.append(timings["planning"])
            total_times.append(timings["planning"] + timings["integration"])

            if len(grasps) == 0:
                no_grasp += 1
                # print("no grasp")
                break  # no detections found, abort this round
            
            grasp, score = grasps[0], scores[0]
            label, _ = sim.execute_grasp(grasp, allow_contact=True)


            cnt += 1
            if label != Label.FAILURE:
                success += 1
                last_trial = None
                # success_list.append(1)
            else:
                last_trial = grasp
                # success_list.append(0)
            # log the grasp
            logger.log_grasp(round_id, state, timings, grasp, score, label)


            
            if visualize:
                color = np.array([0, 0, 255, 255]).astype(np.uint8)
                grasp_mesh_list = [visual.grasp2mesh(g, s) for g, s in zip(grasps, scores)]
                if label == Label.SUCCESS:
                    color = np.array([0, 250, 0, 180]).astype(np.uint8)
                else:
                    color = np.array([250, 0, 0, 180]).astype(np.uint8)
                grasp_mesh_list = [visual.grasp2mesh(g, s, color=color) for g, s in zip(grasps, scores)]
                grasp_mesh_list = [grasp_mesh_list[0]]

                
                composed_scene = trimesh.Scene(visual_mesh)
                for i, g_mesh in enumerate(grasp_mesh_list):
                    composed_scene.add_geometry(g_mesh, node_name=f'grasp_{i}')
                # composed_scene.show()
                composed_scene.set_camera(angles=(np.pi / 3.0, 0, - np.pi / 2.0), distance=1.6 * sim.size, center=composed_scene.centroid)
                data = composed_scene.save_image(resolution=(1280,1080),line_settings= {'point_size': 20})
                image = np.array(Image.open(sysio.BytesIO(data)))
                if not os.path.exists("logs"):
                    os.mkdir("logs")
                plt.imsave("logs/"+f'round_{round_id:03d}_trial_{trial_id:03d}'+'.png', image)
            
            if last_label == Label.FAILURE and label == Label.FAILURE:
                consecutive_failures += 1
            else:
                consecutive_failures = 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                cons_fail += 1
            last_label = label
        

        left_objs += sim.num_objects
    success_rate = 100.0 * success / cnt
    declutter_rate = 100.0 * success / total_objs
    print('Grasp success rate: %.2f %%, Declutter rate: %.2f %%' % (success_rate, declutter_rate))
    print(f'Average planning time: {np.mean(planning_times)}, total time: {np.mean(total_times)}')
    # print('Consecutive failures and no detections: %d, %d' % (cons_fail, no_grasp))
    if result_path is not None:
        with open(result_path, 'w') as f:
            f.write('%.2f%%, %.2f%%; %d, %d\n' % (success_rate, declutter_rate, cons_fail, no_grasp))
    return success_rate, declutter_rate
    


class Logger(object):
    def __init__(self, root, description):
        time_stamp = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        description = "{}_{}".format(time_stamp, description).strip("_")

        self.logdir = root / description
        self.scenes_dir = self.logdir / "scenes"
        self.scenes_dir.mkdir(parents=True, exist_ok=True)

        self.mesh_dir = self.logdir / "meshes"
        self.mesh_dir.mkdir(parents=True, exist_ok=True)

        self.rounds_csv_path = self.logdir / "rounds.csv"
        self.grasps_csv_path = self.logdir / "grasps.csv"
        self._create_csv_files_if_needed()

    def _create_csv_files_if_needed(self):
        if not self.rounds_csv_path.exists():
            io.create_csv(self.rounds_csv_path, ["round_id", "object_count"])

        if not self.grasps_csv_path.exists():
            columns = [
                "round_id",
                "scene_id",
                "qx",
                "qy",
                "qz",
                "qw",
                "x",
                "y",
                "z",
                "width",
                "score",
                "label",
                "integration_time",
                "planning_time",
            ]
            io.create_csv(self.grasps_csv_path, columns)

    def last_round_id(self):
        df = pd.read_csv(self.rounds_csv_path)
        return -1 if df.empty else df["round_id"].max()

    def log_round(self, round_id, object_count):
        io.append_csv(self.rounds_csv_path, round_id, object_count)

    def log_mesh(self, scene_mesh, aff_mesh, name):
        scene_mesh.export(self.mesh_dir / (name + "_scene.obj"))
        aff_mesh.export(self.mesh_dir / (name + "_aff.obj"))

    def log_grasp(self, round_id, state, timings, grasp, score, label):
        # log scene
        tsdf, points = state.tsdf, np.asarray(state.pc.points)
        scene_id = uuid.uuid4().hex
        scene_path = self.scenes_dir / (scene_id + ".npz")
        np.savez_compressed(scene_path, grid=tsdf.get_grid(), points=points)

        # log grasp
        qx, qy, qz, qw = grasp.pose.rotation.as_quat()
        x, y, z = grasp.pose.translation
        width = grasp.width
        label = int(label)
        io.append_csv(
            self.grasps_csv_path,
            round_id,
            scene_id,
            qx,
            qy,
            qz,
            qw,
            x,
            y,
            z,
            width,
            score,
            label,
            timings["integration"],
            timings["planning"],
        )


class Data(object):
    """Object for loading and analyzing experimental data."""

    def __init__(self, logdir):
        self.logdir = logdir
        self.rounds = pd.read_csv(logdir / "rounds.csv")
        self.grasps = pd.read_csv(logdir / "grasps.csv")

    def num_rounds(self):
        return len(self.rounds.index)

    def num_grasps(self):
        return len(self.grasps.index)

    def success_rate(self):
        return self.grasps["label"].mean() * 100

    def percent_cleared(self):
        df = (
            self.grasps[["round_id", "label"]]
            .groupby("round_id")
            .sum()
            .rename(columns={"label": "cleared_count"})
            .merge(self.rounds, on="round_id")
        )
        return df["cleared_count"].sum() / df["object_count"].sum() * 100

    def avg_planning_time(self):
        return self.grasps["planning_time"].mean()

    def read_grasp(self, i):
        scene_id, grasp, label = io.read_grasp(self.grasps, i)
        score = self.grasps.loc[i, "score"]
        scene_data = np.load(self.logdir / "scenes" / (scene_id + ".npz"))

        return scene_data["points"], grasp, score, label
