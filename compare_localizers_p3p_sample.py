import logging
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from hloc.utils.read_write_model import (
    qvec2rotmat,
    read_cameras_binary,
    read_cameras_text,
    read_images_binary,
    read_images_text,
    read_model,
    write_model,
)

def evaluate_standard(model, results, list_file=None, ext=".bin", only_localized=False):
    predictions = {}
    with open(results, "r") as f:
        for data in f.read().rstrip().split("\n"):
            data = data.split()
            name = data[0]
            q, t, time = np.split(np.array(data[1:], float), [4,7])
            t = t[:3]
            time = time[0]
            predictions[name] = (qvec2rotmat(q), t, time)
    if ext == ".bin":
        images = read_images_binary(model / "images.bin")
    else:
        images = read_images_text(model / "images.txt")
    name2id = {image.name: i for i, image in images.items()}

    if list_file is None:
        test_names = list(name2id)
    else:
        with open(list_file, "r") as f:
            test_names = f.read().rstrip().split("\n")

    errors_t = []
    errors_R = []
    durations = []
    for name in test_names:
        if name not in predictions:
            if only_localized:
                continue
            e_t = np.inf
            e_R = 180.0
        else:
            image = images[name2id[name]]
            R_gt, t_gt = image.qvec2rotmat(), image.tvec

            R, t, time = predictions[name]

            e_t = np.linalg.norm(-R_gt.T @ t_gt + R.T @ t, axis=0)
            cos = np.clip((np.trace(np.dot(R_gt.T, R)) - 1) / 2, -1.0, 1.0)
            e_R = np.rad2deg(np.abs(np.arccos(cos)))
        errors_t.append(e_t)
        errors_R.append(e_R)
        durations.append(time)

    errors_t = np.array(errors_t)
    errors_R = np.array(errors_R)

    med_t = np.median(errors_t)
    med_R = np.median(errors_R)
    out = f"Results for file {results.name}:"
    out += f"\nMedian errors: {med_t:.3f}m, {med_R:.3f}deg"

    out += "\nPercentage of test images localized within:"
    threshs_t = [0.01, 0.02, 0.03, 0.05, 0.25, 0.5, 5.0]
    threshs_R = [1.0, 2.0, 3.0, 5.0, 2.0, 5.0, 10.0]
    for th_t, th_R in zip(threshs_t, threshs_R):
        ratio = np.mean((errors_t < th_t) & (errors_R < th_R))
        out += f"\n\t{th_t*100:.0f}cm, {th_R:.0f}deg : {ratio*100:.2f}%"
    print(out)

    return errors_t, errors_R, durations

def evaluate_sampled(model, results, iterations_bounds:list, repetitions_per_bound:int,list_file=None, ext=".bin", only_localized=False):
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
                            predictions[name] = {bound : [] for bound in iterations_bounds}

                    q, t, time = np.split(np.array(data[1:], float), [4,7])
                    t = t[:3]
                    time = time[0]
                    predictions[name][iterations_upper_bound].append((qvec2rotmat(q), t, time))

                    idx += 1

    if ext == ".bin":
        images = read_images_binary(model / "images.bin")
    else:
        images = read_images_text(model / "images.txt")
    name2id = {image.name: i for i, image in images.items()}

    if list_file is None:
        test_names = list(name2id)
    else:
        with open(list_file, "r") as f:
            test_names = f.read().rstrip().split("\n")

    errors_t = []
    errors_R = []
    durations = []

    for name in test_names:
        if name not in predictions:
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
                    R, t, time = predictions[name][iterations_upper_bound][i]

                    e_t = np.linalg.norm(-R_gt.T @ t_gt + R.T @ t, axis=0)
                    cos = np.clip((np.trace(np.dot(R_gt.T, R)) - 1) / 2, -1.0, 1.0)
                    e_R = np.rad2deg(np.abs(np.arccos(cos)))

                    local_err_r.append(e_R)
                    local_err_t.append(e_t)
                    local_time.append(time)

                print('\nIteration bound local t:', local_err_t)
                print('Iteration bound local r:', local_err_r)
                print('Iteration bound local time:', local_time)

                localizer_progress_r.append(np.mean(local_err_r))
                localizer_progress_t.append(np.mean(local_err_t))
                localizer_progress_time.append(np.mean(local_time))

            print("\nImage progress t:", localizer_progress_t)
            print("Image progress r:", localizer_progress_r)
            print("Image progress time:", localizer_progress_time)

            errors_t.append(localizer_progress_t)
            errors_R.append(localizer_progress_r)
            durations.append(localizer_progress_time)

    errors_t = np.array(errors_t)
    errors_R = np.array(errors_R)
    durations = np.array(durations)

    # med_t = np.median(errors_t)
    # med_R = np.median(errors_R)
    # out = f"Results for file {results.name}:"
    # out += f"\nMedian errors: {med_t:.3f}m, {med_R:.3f}deg"
    #
    # out += "\nPercentage of test images localized within:"
    # threshs_t = [0.01, 0.02, 0.03, 0.05, 0.25, 0.5, 5.0]
    # threshs_R = [1.0, 2.0, 3.0, 5.0, 2.0, 5.0, 10.0]
    # for th_t, th_R in zip(threshs_t, threshs_R):
    #     ratio = np.mean((errors_t < th_t) & (errors_R < th_R))
    #     out += f"\n\t{th_t*100:.0f}cm, {th_R:.0f}deg : {ratio*100:.2f}%"
    # print(out)

    return errors_t, errors_R, durations

# def plot_simple_barplot(title:str, x_title:str, y_title:str, save_path:str, data:list, x_labels:list):
#     fig, ax = plt.subplots()
#     ax.set_facecolor('lightblue')
#     sns.boxplot(data=data, ax=ax, palette='tab10')
#     sns.stripplot(data=data, ax=ax, palette='cividis', jitter=True, size=3)
#
#     ax.set_xticklabels(x_labels)
#     ax.set_xlabel(x_title)
#     ax.set_ylabel(y_title)
#     ax.set_title(title)
#
#     plt.savefig(save_path, dpi=1300,
#                 bbox_inches='tight',
#                 facecolor='floralwhite')
#     plt.close()
#     plt.clf()
#     plt.cla()

def plot_series_and_point(x_time_data, y_accuracy_data, x_time_tick, y_accuracy_tick, y_axis_label, title, save_path):
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))

    # Plot the main x-y data with a line connecting the points
    sns.lineplot(x=x_time_data, y=y_accuracy_data, label='p3p progress', marker='o')

    # Add the crossed dashed lines for the specific point using Matplotlib
    plt.axhline(y=y_accuracy_tick, color='red', linestyle='--', linewidth=1, label='RECON accuracy')
    plt.axvline(x=x_time_tick, color='blue', linestyle='--', linewidth=1, label='RECON time')

    # Optionally, mark the specific point with a marker
    plt.scatter(x_time_tick, y_accuracy_tick, color='green', s=100, zorder=5)

    # Add labels and legend
    plt.xlabel('Time (ns)')
    plt.ylabel(y_axis_label)
    plt.title(title)
    plt.legend()

    plt.savefig(save_path, dpi=1300,
            bbox_inches='tight',
            facecolor='floralwhite')
    plt.close()
    plt.clf()
    plt.cla()

gt_dirs = Path("./datasets/cambridge/CambridgeLandmarks_Colmap_Retriangulated_1024px")
model_path = gt_dirs / "ShopFacade" / "empty_all"
list_file =  gt_dirs / "ShopFacade" / "list_query.txt" 

results_paths = {"p3p":Path("./outputs/cambridge/ShopFacade/results.txt"),"RECON":Path("./outputs_recon/cambridge/ShopFacade/results.txt")}

if __name__ == "__main__":
    t_err_results = list()
    r_err_results = list()
    running_durations = list()
    names=list()

    sample_p3p_data = evaluate_sampled(model_path,results_paths['p3p'], [250, 1000, 2000, 3000, 5000, 7000, 10000, 15000], 25, list_file,ext=".txt")
    recon_data = evaluate_standard(model_path,results_paths['RECON'],list_file,ext=".txt")

    idx = 0
    for recon_t, recon_r, recon_time, p3p_t, p3p_r, p3p_time in zip(*recon_data, *sample_p3p_data):
        plot_series_and_point(p3p_time,p3p_r,recon_time,recon_r,"Rotation error", "Rotation error progression", f"./plots/comparisons/r_err_prog{idx}.png")
        plot_series_and_point(p3p_time,p3p_t,recon_time,recon_t,"Translation error", "Translation error progression", f"./plots/comparisons/t_err_prog{idx}.png")

        idx += 1
