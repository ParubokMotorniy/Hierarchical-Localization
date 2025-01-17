import pycolmap
import numpy as np
import poselib


def colmapQueryCamToDict(query_camera_to_convert):
    if not isinstance(query_camera_to_convert, pycolmap.Camera):
        return query_camera_to_convert

    dict_camera = dict()
    dict_camera["model"] = str(query_camera_to_convert.model).split('.')[1]
    dict_camera["width"] = query_camera_to_convert.width
    dict_camera["height"] = query_camera_to_convert.height
    dict_camera["params"] = query_camera_to_convert.params
    return dict_camera


def dictToColmapCamera(camera_to_convert: dict):
    colmapCamera = pycolmap.Camera
    colmapCamera.model = camera_to_convert["model"]
    colmapCamera.width = camera_to_convert["width"]
    colmapCamera.height = camera_to_convert["height"]
    colmapCamera.params = [param for param in camera_to_convert["params"]]
    return colmapCamera


def addRefinementOption(options_src, options_tar, key_source, key_target):
    if key_source in options_src:
        options_tar[key_target] = options_src[key_source]


def transformPointsToPoselib(colmap_points):
    poselib_points = [point.astype(np.float64) for point in colmap_points]
    return poselib_points


def transformPoselibPoseToColmapPose(poselib_pose: poselib.CameraPose):
    quat = poselib_pose.q
    trans = poselib_pose.t

    quat_arr = [[component] for component in quat[1:]]
    quat_arr.append([quat[0]])
    colmap_rotation = pycolmap.Rotation3d(np.array(quat_arr, dtype=np.float64))

    colmap_translation = np.array([[component] for component in trans], dtype=np.float64)
    colmap_pose = pycolmap.Rigid3d(colmap_rotation, colmap_translation)
    return colmap_pose


def transformPoselibMaskToColmapMask(poselib_list_of_bools):
    colmap_mask = np.array([np.bool_(if_inlier) for if_inlier in poselib_list_of_bools], dtype=np.bool_)
    return colmap_mask


def colmapToPoselibRefinementOptions(refinement_options):
    refinement_opts_to_use = dict()

    if isinstance(refinement_options, pycolmap.AbsolutePoseRefinementOptions):
        refinement_opts_to_use["loss_scale"] = refinement_options.loss_function_scale
        refinement_opts_to_use["max_iterations"] = refinement_options.max_num_iterations
        refinement_opts_to_use["gradient_tol"] = refinement_options.gradient_tolerance
    elif isinstance(refinement_options, dict):
        addRefinementOption(refinement_options, refinement_opts_to_use, "loss_function_scale", "loss_scale")
        addRefinementOption(refinement_options, refinement_opts_to_use, "max_num_iterations", "max_iterations")
        addRefinementOption(refinement_options, refinement_opts_to_use, "gradient_tolerance", "gradient_tol")

    return refinement_opts_to_use
