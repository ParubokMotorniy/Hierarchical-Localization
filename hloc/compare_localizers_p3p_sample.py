import argparse
import logging
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d

from hloc.utils.read_write_model import (
    qvec2rotmat,
    read_images_binary,
    read_images_text,
)

def evaluate_standard(model, results,  sequences:list, list_file=None, ext=".bin", only_localized=False):
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

    per_sequence_error_data = { seq:{'errors_t':[], 'errors_R':[], 'durations':[]} for seq in sequences}

    # errors_t = []
    # errors_R = []
    # durations = []
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

        sequence_name = name.split('/')[0]
        if sequence_name in per_sequence_error_data.keys():
            per_sequence_error_data[sequence_name]['errors_t'].append(e_t)
            per_sequence_error_data[sequence_name]['errors_R'].append(e_R)
            per_sequence_error_data[sequence_name]['durations'].append(time)

        # errors_t.append(e_t)
        # errors_R.append(e_R)
        # durations.append(time)

    for seq_name in per_sequence_error_data.keys():

        # errors_t = np.array(errors_t)
        # errors_R = np.array(errors_R)
        # durations = np.array(durations)

        per_sequence_error_data[seq_name]['errors_t'] = np.array(per_sequence_error_data[seq_name]['errors_t']) 
        per_sequence_error_data[seq_name]['errors_R'] = np.array(per_sequence_error_data[seq_name]['errors_R'])
        per_sequence_error_data[seq_name]['durations'] = np.array(per_sequence_error_data[seq_name]['durations']) 

        # med_t = np.median(errors_t)
        # med_R = np.median(errors_R)
        # out = f"Results for file {results.name}:"
        # out += f"\nMedian errors: {med_t:.3f}m, {med_R:.3f}deg"

        # out += "\nPercentage of test images localized within:"
        # threshs_t = [0.01, 0.02, 0.03, 0.05, 0.25, 0.5, 5.0]
        # threshs_R = [1.0, 2.0, 3.0, 5.0, 2.0, 5.0, 10.0]
        # for th_t, th_R in zip(threshs_t, threshs_R):
        #     ratio = np.mean((errors_t < th_t) & (errors_R < th_R))
        #     out += f"\n\t{th_t*100:.0f}cm, {th_R:.0f}deg : {ratio*100:.2f}%"
        # print(out)

    print("Recon error data:", per_sequence_error_data)
    return per_sequence_error_data

def evaluate_sampled(model, results, iterations_bounds:list, repetitions_per_bound:int, sequences:list,list_file=None, ext=".bin", only_localized=False):
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

    # errors_t = []
    # errors_R = []
    # durations = []
    per_sequence_error_data = { seq:{'errors_t':[], 'errors_R':[], 'durations':[]} for seq in sequences}

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

                # print('\nIteration bound local t:', local_err_t)
                # print('Iteration bound local r:', local_err_r)
                # print('Iteration bound local time:', local_time)

                localizer_progress_r.append(np.median(local_err_r))
                localizer_progress_t.append(np.median(local_err_t))
                localizer_progress_time.append(np.median(local_time))

            # print("\nImage progress t:", localizer_progress_t)
            # print("Image progress r:", localizer_progress_r)
            # print("Image progress time:", localizer_progress_time)

            localizer_progress_r = np.array(localizer_progress_r)
            localizer_progress_t = np.array(localizer_progress_t)
            localizer_progress_time = np.array(localizer_progress_time)

            sort_indices = np.argsort(localizer_progress_time)

            sequence_name = name.split('/')[0]

            if(sequence_name in per_sequence_error_data.keys()):
                per_sequence_error_data[sequence_name]['errors_t'].append(localizer_progress_t[sort_indices])
                per_sequence_error_data[sequence_name]['errors_R'].append(localizer_progress_r[sort_indices])
                per_sequence_error_data[sequence_name]['durations'].append(localizer_progress_time[sort_indices])

            # errors_t.append(localizer_progress_t[sort_indices])
            # errors_R.append(localizer_progress_r[sort_indices])
            # durations.append(localizer_progress_time[sort_indices])

    # errors_t = np.array(errors_t)
    # errors_R = np.array(errors_R)
    # durations = np.array(durations)

    for seq_name in per_sequence_error_data.keys():
        per_sequence_error_data[seq_name]['errors_t'] = np.array(per_sequence_error_data[seq_name]['errors_t']) 
        per_sequence_error_data[seq_name]['errors_R'] = np.array(per_sequence_error_data[seq_name]['errors_R'])
        per_sequence_error_data[seq_name]['durations'] = np.array(per_sequence_error_data[seq_name]['durations']) 

    print("Sampled p3p error data", per_sequence_error_data)
    return per_sequence_error_data
    # return errors_t, errors_R, durations

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

def main(dataset_name:str):
    present_sequences = ['seq1', 'seq3']

    gt_dirs = Path("./datasets/cambridge/CambridgeLandmarks_Colmap_Retriangulated_1024px")
    model_path = gt_dirs / dataset_name / "empty_all"
    list_file =  gt_dirs / dataset_name / "list_query.txt"

    results_paths = {"p3p":Path(f"./outputs/cambridge/{dataset_name}/results.txt"),"RECON":Path(f"./outputs_recon/cambridge/{dataset_name}/results.txt")}

    sample_p3p_data = evaluate_sampled(model_path,results_paths['p3p'],[5, 25, 100, 200, 500, 1000, 5000, 7000, 10000], 30, present_sequences, list_file,ext=".txt")
    recon_data = evaluate_standard(model_path,results_paths['RECON'],present_sequences, list_file,ext=".txt")

    #plot by frame #valid only for 09d447ad06276852fade825ee41d8f6dad0cfabf
    # idx = 0
    # for recon_t, recon_r, recon_time, p3p_t, p3p_r, p3p_time in zip(*recon_data, *sample_p3p_data):
    #     plot_series_and_point(p3p_time,p3p_r,recon_time,recon_r,"Rotation error", "Rotation error progression", f"./plots/comparisons/r_err_prog{idx}.png")
    #     plot_series_and_point(p3p_time,p3p_t,recon_time,recon_t,"Translation error", "Translation error progression", f"./plots/comparisons/t_err_prog{idx}.png")

    #     idx += 1

    #summarizing by scene
    
    for seq_name in present_sequences:

        plot_resolution = 200

        # errors_t, errors_R, durations = sample_p3p_data

        errors_t = sample_p3p_data[seq_name]['errors_t']
        errors_R = sample_p3p_data[seq_name]['errors_R']
        durations = sample_p3p_data[seq_name]['durations']

        error_frames_t = []
        error_frames_R = []

        min_time_point = durations[0][0]
        max_time_point = durations[0][0]
        for image_time_list in durations:
            for time_point in image_time_list:
                if time_point < min_time_point:
                    min_time_point = time_point
                if time_point > max_time_point:
                    max_time_point = time_point

        time_step_size = (max_time_point - min_time_point) / plot_resolution
        p3p_durations = np.linspace(min_time_point, max_time_point, plot_resolution)

        print(f"Min time:{min_time_point}")
        print(f"Max time:{max_time_point}")
        print(f"Step size:{time_step_size}")

        for image_errors_t, image_durations in zip(errors_t, durations):
            start_bin = math.ceil((image_durations[0] - min_time_point) / time_step_size)
            bin_length = math.floor((image_durations[-1] - image_durations[0]) / time_step_size)

            interpolator = interp1d(image_durations,[float(err) for err in image_errors_t],kind='linear')
            arguments_to_interpolate = [min_time_point + time_step_size * float(i + start_bin) for i in range(bin_length)]
            interpolated_errors = interpolator(arguments_to_interpolate)

            error_frame_t = [0 for i in range(plot_resolution)]
            for ii in range(start_bin, start_bin + bin_length):
                error_frame_t[ii] = interpolated_errors[ii-start_bin]

            print("Input errors:", image_errors_t)
            print("Interpolated translation errors frame:", error_frame_t)
            print("")

            error_frames_t.append(error_frame_t)

        t_err_time_vector = [0.0 for i in range(plot_resolution)]
        for i in range(plot_resolution):
            items = []
            for frame in error_frames_t:
                if frame[i] != 0:
                    items.append(frame[i])

            if len(items) != 0:
                t_err_time_vector[i] = np.median(items)

        for image_errors_R, image_durations in zip(errors_R, durations):
            start_bin = math.ceil((image_durations[0] - min_time_point) / time_step_size)
            bin_length = math.floor((image_durations[-1] - image_durations[0]) / time_step_size)

            interpolator = interp1d(image_durations,[float(err) for err in image_errors_R],kind='linear')
            arguments_to_interpolate = [min_time_point + time_step_size * float(i + start_bin) for i in range(bin_length)]
            interpolated_errors = interpolator(arguments_to_interpolate)

            error_frame_R = [0 for i in range(plot_resolution)]
            for ii in range(start_bin, start_bin + bin_length):
                error_frame_R[ii] = interpolated_errors[ii-start_bin]

            print("Input errors:", image_errors_R)
            print("Interpolated rotation errors frame:", error_frame_R)
            print("")

            error_frames_R.append(error_frame_R)

        R_err_time_vector = [0.0 for i in range(plot_resolution)]
        for i in range(plot_resolution):
            items = []
            for frame in error_frames_R:
                if frame[i] != 0:
                    items.append(frame[i])

            if len(items) != 0:
                R_err_time_vector[i] = np.median(items)

        #recon interpolation
        # errors_t_recon, errors_R_recon, durations = recon_data

        errors_t_recon = recon_data[seq_name]['errors_t']
        errors_R_recon = recon_data[seq_name]['errors_R']
        durations = recon_data[seq_name]['durations']

        sorted_indices = np.argsort(durations)

        errors_t_recon = errors_t_recon[sorted_indices]
        errors_R_recon = errors_R_recon[sorted_indices]
        durations = durations[sorted_indices]

        recon_time_points = np.linspace(durations[0],durations[-1], plot_resolution)

        recon_interpolator_t = interp1d(durations,errors_t_recon,kind='linear')
        t_err_time_vector_recon = recon_interpolator_t(recon_time_points)

        recon_interpolator_R = interp1d(durations,errors_R_recon,kind='linear')
        R_err_time_vector_recon = recon_interpolator_R(recon_time_points)

        print("p3p t:", t_err_time_vector)
        print("p3p r:", R_err_time_vector)

        print("recon t:", t_err_time_vector_recon)
        print("recon r:",  R_err_time_vector_recon)

        #translation error plot
        sns.set(style="whitegrid")
        plt.figure(figsize=(8, 6))
        # Plot the main x-y data with a line connecting the points
        sns.lineplot(x=p3p_durations, y=t_err_time_vector, label='p3p progress', marker='o')
        sns.lineplot(x=recon_time_points, y=t_err_time_vector_recon, label='recon progress', marker='o')

        plt.xlabel('Time (ns)')
        plt.ylabel("Translation error")
        plt.title(f"Averaged+interpolated translation errors. Dataset : ShopFacade. Scene: {seq_name}")
        plt.legend()

        # plt.show()
        plt.savefig(f"./plots/sequence_error_progression/translation_error_{seq_name}_median.png", dpi=1300,
                    bbox_inches='tight',
                    facecolor='floralwhite')
        plt.clf()
        plt.cla()

        #rotation error plot
        sns.set(style="whitegrid")
        plt.figure(figsize=(8, 6))
        # Plot the main x-y data with a line connecting the points
        sns.lineplot(x=p3p_durations, y=R_err_time_vector, label='p3p progress', marker='o')
        sns.lineplot(x=recon_time_points, y=R_err_time_vector_recon, label='recon progress', marker='o')

        plt.xlabel('Time (ns)')
        plt.ylabel("Rotation error")
        plt.title(f"Averaged+interpolated rotation errors. Dataset : ShopFacade. Scene: {seq_name}")
        plt.legend()

        plt.savefig(f"./plots/sequence_error_progression/rotation_error_{seq_name}_median.png", dpi=1300,
                    bbox_inches='tight',
                    facecolor='floralwhite')
        # plt.show()
        plt.clf()
        plt.cla()

if __name__ == "__main__":
    main(sys.argv[1])
