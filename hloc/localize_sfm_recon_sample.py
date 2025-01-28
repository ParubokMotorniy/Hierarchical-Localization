import argparse
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import poselib
import pycolmap
from tqdm import tqdm
from time import process_time_ns

from . import logger
from .utils.io import get_keypoints, get_matches
from .utils.parsers import parse_image_lists, parse_retrieval


def do_covisibility_clustering(
    frame_ids: List[int], reconstruction: pycolmap.Reconstruction
):
    clusters = []
    visited = set()
    for frame_id in frame_ids:
        # Check if already labeled
        if frame_id in visited:
            continue

        # New component
        clusters.append([])
        queue = {frame_id}
        while len(queue):
            exploration_frame = queue.pop()

            # Already part of the component
            if exploration_frame in visited:
                continue
            visited.add(exploration_frame)
            clusters[-1].append(exploration_frame)

            observed = reconstruction.images[exploration_frame].points2D
            connected_frames = {
                obs.image_id
                for p2D in observed
                if p2D.has_point3D()
                for obs in reconstruction.points3D[p2D.point3D_id].track.elements
            }
            connected_frames &= set(frame_ids)
            connected_frames -= visited
            queue |= connected_frames

    clusters = sorted(clusters, key=len, reverse=True)
    return clusters


class QueryLocalizer:
    def __init__(self, reconstruction, config=None):
        self.reconstruction = reconstruction
        self.config = config or {}
        self.rand_seed = 129

    def colmapQueryCamToDict(self, query_camera_to_convert: pycolmap.Camera):
        dict_camera = dict()
        dict_camera["model"] = str(query_camera_to_convert.model).split('.')[1]
        dict_camera["width"] = query_camera_to_convert.width
        dict_camera["height"] = query_camera_to_convert.height
        dict_camera["params"] = query_camera_to_convert.params
        return dict_camera

    def dictToColmapCamera(self, camera_to_convert: dict):
        colmapCamera = pycolmap.Camera
        colmapCamera.model = camera_to_convert["model"]
        colmapCamera.width = camera_to_convert["width"]
        colmapCamera.height = camera_to_convert["height"]
        colmapCamera.params = [param for param in camera_to_convert["params"]]
        return colmapCamera

    def addRefinementOption(self, options_src, options_tar, key_source, key_target):
        if key_source in options_src:
            options_tar[key_target] = options_src[key_source]

    def transformPointsToPoselib(self, colmap_points):
        poselib_points = [point.astype(np.float64) for point in colmap_points]
        return poselib_points

    def transformPoselibPoseToColmapPose(self, poselib_pose:poselib.CameraPose):
        quat = poselib_pose.q
        trans = poselib_pose.t

        quat_arr = [[component] for component in quat[1:]]
        quat_arr.append([quat[0]])
        colmap_rotation = pycolmap.Rotation3d(np.array(quat_arr, dtype=np.float64))

        colmap_translation = np.array([[component] for component in trans], dtype=np.float64)
        colmap_pose = pycolmap.Rigid3d(colmap_rotation, colmap_translation)
        return colmap_pose

    def transformPoselibMaskToColmapMask(self, poselib_list_of_bools):
        colmap_mask = np.array([np.bool_(if_inlier) for if_inlier in poselib_list_of_bools], dtype=np.bool_)
        return colmap_mask

    def countInliers(self, poselib_list_of_bools):
        num_inliers = 0
        for if_inlier in poselib_list_of_bools:
            if if_inlier:
                num_inliers += 1

        return num_inliers
    def localizeRECON(self, points2D_all, points2D_idxs, points3D_id, query_camera, outer_iteration_limit:int):
        points2D = points2D_all[points2D_idxs]
        points3D = [self.reconstruction.points3D[j].xyz for j in points3D_id]
        points2D = self.transformPointsToPoselib(points2D)
        points3D = self.transformPointsToPoselib(points3D)

        cam_to_use = query_camera
        if isinstance(query_camera, pycolmap.Camera):
            cam_to_use = self.colmapQueryCamToDict(query_camera)

        refinement_options = self.config.get("refinement", {})

        refinement_opts_to_use = dict()
        if isinstance(refinement_options, pycolmap.AbsolutePoseRefinementOptions):
            refinement_opts_to_use["loss_scale"] = refinement_options.loss_function_scale
            refinement_opts_to_use["max_iterations"] = refinement_options.max_num_iterations
            refinement_opts_to_use["gradient_tol"] = refinement_options.gradient_tolerance
        elif isinstance(refinement_options, dict):
            self.addRefinementOption(refinement_options,refinement_opts_to_use, "loss_function_scale","loss_scale")
            self.addRefinementOption(refinement_options,refinement_opts_to_use, "max_num_iterations","max_iterations")
            self.addRefinementOption(refinement_options,refinement_opts_to_use, "gradient_tolerance","gradient_tol")
        
        self.rand_seed += 1
        recon_opts = {'outerIterations': outer_iteration_limit, 'nP3pSamples': 15, 'nBestModelsConsidered': 3, 'randSeed': self.rand_seed,
                      'strictConsistencyAlpha': 0.99, 'nRECONStrictModels': 3, 'p3pInlierThreshold': 35.0,
                      'up2pInlierThreshold': 35.0, 'nOuterUp2pPoses': 3, 'nInlierSetsRequired': 1, 'failure_probability' : 0.85}

        time_start = process_time_ns()
        camR, inliersR = poselib.RECON_threshold_solver_iterations_bounded(points2D, points3D, cam_to_use, recon_opts, refinement_opts_to_use)
        time_end = process_time_ns()

        ret = {}
        ret["cam_from_world"] = self.transformPoselibPoseToColmapPose(camR)
        ret["num_inliers"] = len(inliersR["inliers"])
        ret["inliers"] = self.transformPoselibMaskToColmapMask(inliersR["inlierMask"])
        ret["time"] = math.floor((time_end - time_start) / 1000000)

        return ret

def pose_from_cluster(
        localizer: QueryLocalizer,
	        qname: str,
        query_camera: pycolmap.Camera,
        db_ids: List[int],
        features_path: Path,
        matches_path: Path,
        recon_iteration_limit:int,
        **kwargs,
):
    kpq = get_keypoints(features_path, qname)
    kpq += 0.5  # COLMAP coordinates

    kp_idx_to_3D = defaultdict(list)
    kp_idx_to_3D_to_db = defaultdict(lambda: defaultdict(list))
    num_matches = 0
    for i, db_id in enumerate(db_ids):
        image = localizer.reconstruction.images[db_id]
        if image.num_points3D == 0:
            logger.debug(f"No 3D points found for {image.name}.")
            continue
        points3D_ids = np.array(
            [p.point3D_id if p.has_point3D() else -1 for p in image.points2D]
        )

        matches, _ = get_matches(matches_path, qname, image.name)
        matches = matches[points3D_ids[matches[:, 1]] != -1]
        num_matches += len(matches)
        for idx, m in matches:
            id_3D = points3D_ids[m]
            kp_idx_to_3D_to_db[idx][id_3D].append(i)
            # avoid duplicate observations
            if id_3D not in kp_idx_to_3D[idx]:
                kp_idx_to_3D[idx].append(id_3D)

    idxs = list(kp_idx_to_3D.keys())
    mkp_idxs = [i for i in idxs for _ in kp_idx_to_3D[i]]
    mp3d_ids = [j for i in idxs for j in kp_idx_to_3D[i]]
    ret = localizer.localizeRECON(kpq, mkp_idxs, mp3d_ids, query_camera, recon_iteration_limit)

    if ret is not None:
        ret["camera"] = query_camera

    # mostly for logging and post-processing
    mkp_to_3D_to_db = [
        (j, kp_idx_to_3D_to_db[i][j]) for i in idxs for j in kp_idx_to_3D[i]
    ]
    log = {
        "db": db_ids,
        "PnP_ret": ret,
        "keypoints_query": kpq[mkp_idxs],
        "points3D_ids": mp3d_ids,
        "points3D_xyz": None,  # we don't log xyz anymore because of file size
        "num_matches": num_matches,
        "keypoint_index_to_db": (mkp_idxs, mkp_to_3D_to_db),
    }
    return ret, log


def main(
        reference_sfm: Union[Path, pycolmap.Reconstruction],
        queries: Path,
        retrieval: Path,
        features: Path,
        matches: Path,
        results: Path,
        ransac_thresh: int = 12,
        covisibility_clustering: bool = False,
        prepend_camera_name: bool = False,
        config: Dict = None,
        iterations_bounds=None,
        iteration_repetitions:int = 20,
):
    if iterations_bounds is None or isinstance(iterations_bounds, list):
        iterations_bounds = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]

    assert retrieval.exists(), retrieval
    assert features.exists(), features
    assert matches.exists(), matches

    queries = parse_image_lists(queries, with_intrinsics=True)
    retrieval_dict = parse_retrieval(retrieval)

    logger.info("Reading the 3D model...")
    if not isinstance(reference_sfm, pycolmap.Reconstruction):
        reference_sfm = pycolmap.Reconstruction(reference_sfm)
    db_name_to_id = {img.name: i for i, img in reference_sfm.images.items()}

    config = {"estimation": {"ransac": {"max_error": ransac_thresh}}, **(config or {})}
    localizer = QueryLocalizer(reference_sfm, config)

    cam_from_world = {}
    logs = {
        "features": features,
        "matches": matches,
        "retrieval": retrieval,
        "loc": {},
    }
    logger.info("Starting localization...")
    for qname, qcam in tqdm(queries):
        if qname not in retrieval_dict:
            logger.warning(f"No images retrieved for query image {qname}. Skipping...")
            continue
        db_names = retrieval_dict[qname]
        db_ids = []
        for n in db_names:
            if n not in db_name_to_id:
                logger.warning(f"Image {n} was retrieved but not in database")
                continue
            db_ids.append(db_name_to_id[n])

        cam_from_world[qname] = {num_iter : [] for num_iter in iterations_bounds}

        for iterations_upper_bound in iterations_bounds:
            print(f"[ {qname} : {iterations_upper_bound} ]")

            for i in range(iteration_repetitions):
                ret, log = pose_from_cluster(
                    localizer, qname, qcam, db_ids, features, matches, iterations_upper_bound
                )

                if ret is not None:
                    rec_pos = ret["cam_from_world"]
                    # print(f"[{iterations_upper_bound} : {i}] Recorded pos: {rec_pos}")

                    rec_time = ret["time"]
                    # print(f"[{iterations_upper_bound} : {i}] Recorded time: {rec_time}")

                    cam_from_world[qname][iterations_upper_bound].append((ret["cam_from_world"], ret["time"], ret["num_inliers"] > 0))
                else:
                    closest = reference_sfm.images[db_ids[0]]
                    cam_from_world[qname][iterations_upper_bound].append((closest.cam_from_world,-1, False))

                log["covisibility_clustering"] = covisibility_clustering
                logs["loc"][qname] = log

    logger.info(f"Localized {len(cam_from_world)} / {len(queries)} images.")
    logger.info(f"Writing poses to {results}...")
    with open(results, "w") as f:
        for query, iteration_data in cam_from_world.items():
            for iterations_upper_bound, data_list in iteration_data.items():
                for t, time, success in data_list:
                    qvec = " ".join(map(str, t.rotation.quat[[3, 0, 1, 2]])) if success else "2 2 2 2" #cosine can never be > 1
                    tvec = " ".join(map(str, t.translation)) if success else "-1 -1 -1"
                    name = query.split("/")[-1]
                    if prepend_camera_name:
                        name = query.split("/")[-2] + "/" + name
                    f.write(f"{name} {qvec} {tvec} {time}\n")

    logs_path = f"{results}_logs.pkl"
    logger.info(f"Writing logs to {logs_path}...")
    # TODO: Resolve pickling issue with pycolmap objects.
    with open(logs_path, "wb") as f:
        pickle.dump(logs, f)
    logger.info("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_sfm", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--ransac_thresh", type=float, default=12.0)
    parser.add_argument("--covisibility_clustering", action="store_true")
    parser.add_argument("--prepend_camera_name", action="store_true")
    args = parser.parse_args()
    main(**args.__dict__)
