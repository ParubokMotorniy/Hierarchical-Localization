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

def evaluate(model, results, list_file=None, ext=".bin", only_localized=False):
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

def plot_simple_barplot(title:str, x_title:str, y_title:str, save_path:str, data:list, x_labels:list):
    fig, ax = plt.subplots()
    ax.set_facecolor('lightblue')
    sns.boxplot(data=data, ax=ax, palette='tab10')
    sns.stripplot(data=data, ax=ax, palette='cividis', jitter=True, size=3)

    ax.set_xticklabels(x_labels)
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    ax.set_title(title)

    plt.savefig(save_path, dpi=1300,
                bbox_inches='tight',
                facecolor='floralwhite')
    plt.close()
    plt.clf()
    plt.cla()

gt_dirs = Path("./datasets/cambridge/CambridgeLandmarks_Colmap_Retriangulated_1024px")
model_path = gt_dirs / "ShopFacade" / "empty_all"
list_file =  gt_dirs / "ShopFacade" / "list_query.txt" 

results_paths = {"p3p":Path("./outputs_p3p/cambridge/ShopFacade/results.txt"),"RECON":Path("./outputs_recon/cambridge/ShopFacade/results.txt")}

if __name__ == "__main__":
    t_err_results = list()
    r_err_results = list()
    running_durations = list()
    names=list()

    for localizer_name, results_path in results_paths.items():
        t_err, r_err, time = evaluate(model_path, results_path, list_file, ext=".txt")

        t_err_results.append(t_err)
        r_err_results.append(r_err)
        running_durations.append(time)
        names.append(localizer_name)

    plot_simple_barplot("Translation errors measured for ShopFacade dataset", "Solvers", "Translation error", "./plots/comparisons/t_err.png",t_err_results,names)
    plot_simple_barplot("Rotation errors measured for ShopFacade dataset", "Solvers", "Rotation error", "./plots/comparisons/r_err.png",r_err_results,names)
    plot_simple_barplot("Running durations measured for ShopFacade dataset", "Solvers", "Time (ns)", "./plots/comparisons/time.png",running_durations,names)
