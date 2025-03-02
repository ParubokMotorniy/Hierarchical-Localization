import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

## hloc copy

# Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

import collections
import struct

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict(
    [(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS]
)
CAMERA_MODEL_NAMES = dict(
    [(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS]
)


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_images_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack(
                    [tuple(map(float, elems[0::3])), tuple(map(float, elems[1::3]))]
                )
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    point3D_ids=point3D_ids,
                )
    return images


def read_images_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[
                0
            ]
            x_y_id_s = read_next_bytes(
                fid,
                num_bytes=24 * num_points2D,
                format_char_sequence="ddq" * num_points2D,
            )
            xys = np.column_stack(
                [tuple(map(float, x_y_id_s[0::3])), tuple(map(float, x_y_id_s[1::3]))]
            )
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images

def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )
###

def extract_solver_data(model, results, iterations_bounds: list, repetitions_per_bound: int, sequences: list,
                     num_repetitions_considered: int, list_file=None, ext=".bin", only_localized=False,
                     sort_errors_by_time=False, default_rot_error: int = 90, default_trans_error: int = 5):
    predictions = {}

    with open(results, "r") as f:
        entire_file = f.read().rstrip().split("\n")
        idx = 0

        while idx < len(entire_file):
            for iterations_upper_bound in iterations_bounds:
                for i in range(repetitions_per_bound):
                    data = entire_file[idx].split()
                    name = data[0]

                    if name not in predictions.keys():
                        if not (iterations_upper_bound == iterations_bounds[0] and i == 0):
                            raise RuntimeError("Stride mismatch!")
                        else:
                            predictions[name] = {bound: [] for bound in iterations_bounds}

                    q, t, time = np.split(np.array(data[1:], float), [4, 7])
                    t = t[:3]
                    time = time[0]
                    if_error = np.all([i == 2 for i in q]) or np.all([i == 0.0 for i in t])

                    predictions[name][iterations_upper_bound].append((qvec2rotmat(q), t, time, not if_error))

                    idx += 1

    if ext == "bin":
        images = read_images_binary(model / "images.bin")
    else:
        images = read_images_text(model / "images.txt")
    name2id = {image.name: i for i, image in images.items()}

    if list_file is None:
        test_names = list(name2id)
    else:
        with open(list_file, "r") as f:
            test_names = f.read().rstrip().split("\n")

    per_sequence_error_data = {seq: {'errors_t': [], 'errors_R': [], 'durations': []} for seq in sequences}

    for name in test_names:
        if name not in predictions.keys():
            if only_localized:
                continue
            e_t = np.inf
            e_R = 180.0
        else:
            image = images[name2id[name]]
            R_gt, t_gt = image.qvec2rotmat(), image.tvec

            localizer_progress_r = []
            localizer_progress_t = []
            localizer_progress_time = []

            for iterations_upper_bound in iterations_bounds:
                local_err_r = []
                local_err_t = []
                local_time = []

                for i in range(repetitions_per_bound):
                    R, t, time, if_no_error = predictions[name][iterations_upper_bound][i]

                    if time == -1 or not if_no_error: #either indicates an error
                        e_t = default_trans_error
                        e_R = default_rot_error                            
                    else:
                        e_t = np.linalg.norm(-R_gt.T @ t_gt + R.T @ t, axis=0)
                        cos = np.clip((np.trace(np.dot(R_gt.T, R)) - 1) / 2, -1.0, 1.0)
                        e_R = np.rad2deg(np.abs(np.arccos(cos))) 
                        
                    local_err_r.append(e_R)
                    local_err_t.append(e_t)
                    local_time.append(time)

                consideration_limit = min(num_repetitions_considered, len(local_err_t))

                localizer_progress_r.append(np.nan_to_num(np.median(local_err_r[:consideration_limit]), True, default_rot_error,default_rot_error,default_rot_error))
                localizer_progress_t.append(np.nan_to_num(np.median(local_err_t[:consideration_limit]), True, default_trans_error, default_trans_error, default_trans_error))
                localizer_progress_time.append(np.median(local_time[:consideration_limit]))

            localizer_progress_r = np.array(localizer_progress_r)
            localizer_progress_t = np.array(localizer_progress_t)
            localizer_progress_time = np.array(localizer_progress_time)

            sort_indices = np.argsort(localizer_progress_time) if sort_errors_by_time else [i for i in range(
                len(localizer_progress_time))]

            sequence_name = name.split('/')[0]

            if (sequence_name in per_sequence_error_data.keys()):
                per_sequence_error_data[sequence_name]['errors_t'].append(localizer_progress_t[sort_indices])
                per_sequence_error_data[sequence_name]['errors_R'].append(localizer_progress_r[sort_indices])
                per_sequence_error_data[sequence_name]['durations'].append(localizer_progress_time[sort_indices])

    # sequence -> statistic -> lists of values sorted by time (for each bound) for each frame in that sequence
    for seq_name in per_sequence_error_data.keys():
        per_sequence_error_data[seq_name]['errors_t'] = np.array(per_sequence_error_data[seq_name]['errors_t'])
        per_sequence_error_data[seq_name]['errors_R'] = np.array(per_sequence_error_data[seq_name]['errors_R'])
        per_sequence_error_data[seq_name]['durations'] = np.array(per_sequence_error_data[seq_name]['durations'])

    return per_sequence_error_data

def compare_aggregate_by_bound_any(present_sequences: list, sample_data: list, iteration_bounds: list,
                                   solvers_names: list):
    available_statistics = ['errors_t', 'errors_R', 'durations']

    subtitle = ""
    for idx, solver_name in enumerate(solvers_names):
        subtitle += f"{solver_name} iteration limits: {iteration_bounds[idx]}\n"

    for seq_name in present_sequences:
        print(f"\n-----[ Processing sequence {seq_name} ]-----\n")

        solvers_median_errors_t = []
        solvers_median_errors_r = []
        solvers_median_runtimes = []

        for solver_sample_data in sample_data:
            targets = []

            seq_data = solver_sample_data[seq_name]

            for statistic in available_statistics:
                statistics_data = np.array(seq_data[statistic])
                targets.append(np.median(statistics_data, axis=0))

            solvers_median_errors_t.append(targets[0])
            solvers_median_errors_r.append(targets[1])
            solvers_median_runtimes.append(targets[2])

        metric = np.concatenate(
            [[f"{name}_accuracy" for i in range(len(solvers_median_runtimes[0]))] for name in solvers_names]).tolist()
        time_points = np.concatenate(solvers_median_runtimes).tolist()
        errors_t = np.concatenate(solvers_median_errors_t).tolist()
        errors_r = np.concatenate(solvers_median_errors_r).tolist()

        # plotting

        ##translation

        error_t_data_to_plot = pd.DataFrame.from_dict({"metric": metric, "time_point": time_points, "error": errors_t})
        sns.lineplot(data=error_t_data_to_plot, x="time_point", y="error", hue="metric", style="metric", legend="brief")
        sns.scatterplot(data=error_t_data_to_plot, x="time_point", y="error", hue="metric", style="metric",
                        legend=False)

        plt.suptitle(f"Translation error comparison; Sequence: {seq_name};\n {subtitle}")

        plt.grid(axis='y')
        plt.ylabel("Translation error")
        plt.xlabel("Runtime (ms)")

        fig = plt.gcf()
        fig.set_size_inches(11, 8)

        plt.savefig(f"./figures/aggregate_bound_all_translation_error_{seq_name}.png",
                    dpi=1300,
                    bbox_inches='tight',
                    facecolor='floralwhite')
        # plt.show()

        plt.clf()

        ##rotation

        error_r_data_to_plot = pd.DataFrame.from_dict({"metric": metric, "time_point": time_points, "error": errors_r})
        sns.lineplot(data=error_r_data_to_plot, x="time_point", y="error", hue="metric", style="metric", legend="brief")
        sns.scatterplot(data=error_r_data_to_plot, x="time_point", y="error", hue="metric", style="metric",
                        legend=False)

        plt.suptitle(f"Rotation error comparison; Sequence: {seq_name};\n {subtitle}")
        plt.grid(axis='y')
        plt.ylabel("Rotation error")
        plt.xlabel("Runtime (ms)")

        fig = plt.gcf()
        fig.set_size_inches(11, 8)

        plt.savefig(f"./figures/aggregate_bound_all_rotation_error_{seq_name}.png", dpi=1300,
                    bbox_inches='tight',
                    facecolor='floralwhite')
        # plt.show()

        plt.clf()

def compare_aggregate_by_time_any(present_sequences: list, sample_data: list, iteration_bounds: list,
                                  solvers_names: list, n_time_points: int = 10, default_rot_error: int = 1.5,
                                  default_trans_error: int = 1.5):
    subtitle = ""
    for idx, solver_name in enumerate(solvers_names):
        subtitle += f"{solver_name} iteration limits: {iteration_bounds[idx]}\n"

    for seq_name in present_sequences:
        print(f"\n-----[ Processing sequence {seq_name} ]-----\n")

        solvers_errors_t = []
        solvers_errors_r = []
        solvers_timepoints = []

        for solver_sample_data in sample_data:
            seq_data = solver_sample_data[seq_name]

            solver_time_points = np.linspace(
                np.min([np.min(frame_time_data) for frame_time_data in seq_data['durations']]),
                np.max([np.max(frame_time_data) for frame_time_data in seq_data['durations']]),
                n_time_points + 1)
            solver_time_points = solver_time_points[1:]  # skipping the first one

            translation_error_medians_over_time_frames = []
            rotation_error_medians_over_time_frames = []

            for time_point in solver_time_points:
                fitting_indices = []  # enumerates the best result indices for all frames

                for frame_time_data in seq_data['durations']:  # over lists of durations (for each frame)
                    best_time_idx = None
                    for idx, result_time_point in enumerate(frame_time_data):
                        if result_time_point <= time_point:
                            best_time_idx = idx

                    fitting_indices.append(best_time_idx)

                indexed_errors_t = []
                indexed_errors_r = []

                for idx, errors_t, errors_r in zip(fitting_indices, seq_data['errors_t'], seq_data['errors_R']):
                    indexed_errors_t.append(errors_t[idx] if idx is not None else default_trans_error)
                    indexed_errors_r.append(errors_r[idx] if idx is not None else default_rot_error)

                translation_error_medians_over_time_frames.append(np.median(indexed_errors_t))
                rotation_error_medians_over_time_frames.append(np.median(indexed_errors_r))

            solvers_errors_t.append(translation_error_medians_over_time_frames)
            solvers_errors_r.append(rotation_error_medians_over_time_frames)
            solvers_timepoints.append(solver_time_points)

        metric = np.concatenate(
            [[f"{name}_accuracy" for i in range(len(solvers_timepoints[0]))] for name in solvers_names]).tolist()
        time_points = np.concatenate(solvers_timepoints).tolist()
        errors_t = np.concatenate(solvers_errors_t).tolist()
        errors_r = np.concatenate(solvers_errors_r).tolist()

        # plotting

        ##translation

        error_t_data_to_plot = pd.DataFrame.from_dict({"metric": metric, "time_point": time_points, "error": errors_t})
        sns.lineplot(data=error_t_data_to_plot, x="time_point", y="error", hue="metric", style="metric", legend="brief")
        sns.scatterplot(data=error_t_data_to_plot, x="time_point", y="error", hue="metric", style="metric",
                        legend=False)

        plt.suptitle(f"Translation error comparison; Sequence: {seq_name};\n {subtitle}")
        plt.grid(axis='y')
        plt.ylabel("Translation error")
        plt.xlabel("Runtime (ms)")

        fig = plt.gcf()
        fig.set_size_inches(11, 8)

        plt.savefig(f"./figures/aggregate_time_all_translation_error_{seq_name}.png", dpi=1300,
                    bbox_inches='tight',
                    facecolor='floralwhite')
        # plt.show()

        plt.clf()

        ##rotation

        error_r_data_to_plot = pd.DataFrame.from_dict({"metric": metric, "time_point": time_points, "error": errors_r})
        sns.lineplot(data=error_r_data_to_plot, x="time_point", y="error", hue="metric", style="metric", legend="brief")
        sns.scatterplot(data=error_r_data_to_plot, x="time_point", y="error", hue="metric", style="metric",
                        legend=False)

        plt.suptitle(f"Rotation error comparison; Sequence: {seq_name};\n{subtitle}")
        plt.grid(axis='y')
        plt.ylabel("Rotation error")
        plt.xlabel("Runtime (ms)")

        fig = plt.gcf()
        fig.set_size_inches(11, 8)

        plt.savefig(f"./figures/aggregate_time_all_rotation_error_{seq_name}.png", dpi=1300,
                    bbox_inches='tight',
                    facecolor='floralwhite')
        # plt.show()

        plt.clf()

def main_sample_all(dataset_name: str, if_generalize_for_dataset: bool, gt_dirs: Path, results_dir: Path, gt_type: str):
    present_sequences = ['nexus4', 'nexus5x', 'milestone'] #TODO: vary this for different scenes

    model_path = gt_dirs
    list_file = gt_dirs  / "list_query.txt"

    results_paths = {"p3p": results_dir / "results_p3p.txt",
                     "up2p": results_dir / "results_up2p.txt",
                     "recon": results_dir / "results_recon.txt"}
    
    iteration_bounds_recon =    [5,   10,  15, 20,   25,  30,  35,  45, 55,   65,   75,   90,  105,  120,  140,  160,  185]
    iteration_bounds_standard = [75, 150, 225, 300, 375, 450, 525, 675, 825, 975, 1125, 1350, 1575, 1800, 2100, 2400, 2775]
    
    num_repetitions_recon = 3
    num_repetitions_p3p = 20
    num_repetitions_up2p = 10

    solvers_names = ["p3p", "up2p", "recon"]
    solvers_iterations_bounds = [iteration_bounds_standard, iteration_bounds_standard, iteration_bounds_recon]

    print(f"\n-----[ Processing recon data ]-----\n")
    sample_recon_data = extract_solver_data(model_path, results_paths['recon'], iteration_bounds_recon,
                                         num_repetitions_recon, present_sequences, num_repetitions_recon, list_file,
                                         ext=gt_type, sort_errors_by_time=False)  # TODO: note sorting here

    print(f"\n-----[ Processing p3p data ]-----\n")
    sample_p3p_data = extract_solver_data(model_path, results_paths['p3p'], iteration_bounds_standard, num_repetitions_p3p,
                                       present_sequences, num_repetitions_p3p, list_file, ext=gt_type,
                                       sort_errors_by_time=False)  # TODO: note sorting here

    print(f"\n-----[ Processing up2p data ]-----\n")
    sample_up2p_data = extract_solver_data(model_path, results_paths['up2p'], iteration_bounds_standard,
                                        num_repetitions_up2p, present_sequences, num_repetitions_up2p, list_file,
                                        ext=gt_type, sort_errors_by_time=False)  # TODO: note sorting here

    solvers_sample_data = [sample_p3p_data, sample_up2p_data, sample_recon_data]

    if if_generalize_for_dataset:  # if the data has to be aggregated for the entire dataset
        for sample_data in solvers_sample_data:
            sample_data['entire_set'] = {}
            sample_data["entire_set"]['durations'] = [duration_data for seq_name in present_sequences for duration_data
                                                      in sample_data[seq_name]['durations']]
            sample_data["entire_set"]['errors_t'] = [error_data for seq_name in present_sequences for error_data in
                                                     sample_data[seq_name]['errors_t']]
            sample_data["entire_set"]['errors_R'] = [error_data for seq_name in present_sequences for error_data in
                                                     sample_data[seq_name]['errors_R']]
            for seq_name in present_sequences:
                sample_data.pop(seq_name)

        present_sequences = ['entire_set']

    compare_aggregate_by_bound_any(present_sequences, solvers_sample_data, solvers_iterations_bounds,
                                   solvers_names)
    compare_aggregate_by_time_any(present_sequences, solvers_sample_data, solvers_iterations_bounds,
                                  solvers_names, 10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=Path,
        required=True,
        help="Path to the directory with results files.",
    )
    parser.add_argument(
        "--aggregate_entire_set",
        type=bool,
        default=True,
        help="Whether to aggregate data for entire dataset. Default: %(default)s. If `False`, aggregates data on per-sequence basis.",
    )
    parser.add_argument(
        "--gt_dir",
        type=Path,
        help="Path to the directory where ground truth data is located. The script expects to find `images.txt`/`images.bin` and `list_query.txt` files there.",
        required=True
    )
    parser.add_argument(
        "--gt_type",
        type=str,
        default="txt",
        help="The format in which gt data is stored: either `txt` or `bin`. Default: %(default)s"
    )
    args = parser.parse_args()

    main_sample_all("aachen", args.aggregate_entire_set, args.gt_dir, args.results_dir, args.gt_type)