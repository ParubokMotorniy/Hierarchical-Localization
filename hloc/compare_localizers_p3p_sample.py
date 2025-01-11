import argparse
import logging
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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

    for seq_name in per_sequence_error_data.keys():

        per_sequence_error_data[seq_name]['errors_t'] = np.array(per_sequence_error_data[seq_name]['errors_t'])
        per_sequence_error_data[seq_name]['errors_R'] = np.array(per_sequence_error_data[seq_name]['errors_R'])
        per_sequence_error_data[seq_name]['durations'] = np.array(per_sequence_error_data[seq_name]['durations'])

    # print("Recon error data:", per_sequence_error_data)
    return per_sequence_error_data

def evaluate_sampled(model, results, iterations_bounds:list, repetitions_per_bound:int, sequences:list, num_repetitions_considered:int, list_file=None, ext=".bin", only_localized=False, sort_errors_by_time=True, default_rot_error:int=90, default_trans_error:int=5):
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
                    if_no_error = not np.all([i == 2 for i in q]) #cosine can not exceed 1

                    print(f"?? RECON results string: {data} ??")
                    if if_no_error:
                        print(f"++ RECON found inlier set estimate for {name} within {iterations_upper_bound} iterations ++")
                    else:
                        print(f"-- RECON FAILED to find inlier set estimate for {name} within {iterations_upper_bound} iterations --")

                    predictions[name][iterations_upper_bound].append((qvec2rotmat(q), t, time, if_no_error))

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
                    R, t, time, if_no_error = predictions[name][iterations_upper_bound][i]

                    e_t = np.linalg.norm(-R_gt.T @ t_gt + R.T @ t, axis=0) if if_no_error else default_trans_error
                    cos = np.clip((np.trace(np.dot(R_gt.T, R)) - 1) / 2, -1.0, 1.0)
                    e_R = np.rad2deg(np.abs(np.arccos(cos))) if if_no_error else default_rot_error

                    local_err_r.append(e_R)
                    local_err_t.append(e_t)
                    local_time.append(time)

                # print('\nIteration bound local t:', local_err_t)
                # print('Iteration bound local r:', local_err_r)
                # print('Iteration bound local time:', local_time)

                localizer_progress_r.append(np.median(local_err_r[:num_repetitions_considered]))
                localizer_progress_t.append(np.median(local_err_t[:num_repetitions_considered]))
                localizer_progress_time.append(np.median(local_time[:num_repetitions_considered]))

            # print("\nImage progress t:", localizer_progress_t)
            # print("Image progress r:", localizer_progress_r)
            # print("Image progress time:", localizer_progress_time)

            localizer_progress_r = np.array(localizer_progress_r)
            localizer_progress_t = np.array(localizer_progress_t)
            localizer_progress_time = np.array(localizer_progress_time)

            sort_indices = np.argsort(localizer_progress_time) if sort_errors_by_time else [i for i in range(len(localizer_progress_time))]

            sequence_name = name.split('/')[0]

            if(sequence_name in per_sequence_error_data.keys()):
                per_sequence_error_data[sequence_name]['errors_t'].append(localizer_progress_t[sort_indices])
                per_sequence_error_data[sequence_name]['errors_R'].append(localizer_progress_r[sort_indices])
                per_sequence_error_data[sequence_name]['durations'].append(localizer_progress_time[sort_indices])

    #sequence -> statistic -> lists of values sorted by time (for each bound) for each frame in that sequence
    for seq_name in per_sequence_error_data.keys():
        per_sequence_error_data[seq_name]['errors_t'] = np.array(per_sequence_error_data[seq_name]['errors_t'])
        per_sequence_error_data[seq_name]['errors_R'] = np.array(per_sequence_error_data[seq_name]['errors_R'])
        per_sequence_error_data[seq_name]['durations'] = np.array(per_sequence_error_data[seq_name]['durations'])

        # print("Sampled p3p error data", per_sequence_error_data)
    return per_sequence_error_data

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

#expects data to be split by sequence
def compare_interpolate(present_sequences:list, sample_p3p_data:dict, recon_data:dict):
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

        # print("p3p t:", t_err_time_vector)
        # print("p3p r:", R_err_time_vector)
        #
        # print("recon t:", t_err_time_vector_recon)
        # print("recon r:",  R_err_time_vector_recon)

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
        plt.savefig(f"./plots/sequence_error_progression/interpolated_translation_error_{seq_name}_median.png", dpi=1300,
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

        plt.savefig(f"./plots/sequence_error_progression/interpolated_rotation_error_{seq_name}_median.png", dpi=1300,
                    bbox_inches='tight',
                    facecolor='floralwhite')
        # plt.show()
        plt.clf()
        plt.cla()

#works for zero-shot RECON results and iteration-limit p3p results
def compare_aggregate_by_bound(present_sequences:list, sample_p3p_data:dict, recon_data:dict, iteration_bounds:list):
    available_statistics = ['errors_t', 'errors_R', 'durations']

    for seq_name in present_sequences:
        print(f"\n-----[ Processing sequence {seq_name} ]-----\n")

        #p3p processing
        targets = []

        seq_data = sample_p3p_data[seq_name]

        for statistic in available_statistics:
            statistics_data = np.array(seq_data[statistic])
            targets.append(np.mean(statistics_data, axis=0))

        p3p_median_error_t = targets[0]
        p3p_median_error_r = targets[1]
        p3p_median_time = targets[2]

        print(f"Median p3p translation data: {p3p_median_error_t}")
        print(f"Median p3p rotation data: {p3p_median_error_r}")
        print(f"Median p3p time data: {p3p_median_time}")

        #recon processing

        recons_seq_data = recon_data[seq_name]
        interpolation_points = np.linspace(np.min(recons_seq_data['durations']), np.max(recons_seq_data['durations']), len(p3p_median_time))

        recon_t_interpolator = interp1d(recons_seq_data['durations'], recons_seq_data['errors_t'], kind='linear')#, fill_value="extrapolate")
        recon_interpolated_error_t = recon_t_interpolator(interpolation_points) #interpolating by p3p time

        recon_r_interpolator = interp1d(recons_seq_data['durations'], recons_seq_data['errors_R'], kind='linear')#, fill_value="extrapolate")
        recon_interpolated_error_r = recon_r_interpolator(interpolation_points) #interpolating by p3p time

        #plotting

        ##translation
        metric = []
        time_points = []
        errors = []
        for error, time_point in zip(p3p_median_error_t, p3p_median_time):
            metric.append("p3p_accuracy")
            time_points.append(time_point)
            errors.append(error)

        mean_error_t = np.mean(recons_seq_data['errors_t'])
        median_error_t = np.median(recons_seq_data['errors_t'])
        for error, time_point in zip(recon_interpolated_error_t, interpolation_points):
            metric.append("recon_accuracy")
            time_points.append(time_point)
            errors.append(error)

            metric.append("recon_accuracy_mean")
            time_points.append(time_point)
            errors.append(mean_error_t)

            metric.append("recon_accuracy_median")
            time_points.append(time_point)
            errors.append(median_error_t)

        error_t_data_to_plot = pd.DataFrame.from_dict({"metric":metric, "time_point":time_points, "error":errors})
        sns.lineplot(data=error_t_data_to_plot, x="time_point", y="error", hue="metric", style="metric", legend="brief")

        plt.title(f"Translation error comparison\nSequence: {seq_name}; P3P iteration limits: {iteration_bounds}")
        plt.grid(axis='y')
        plt.ylabel("Translation error")
        plt.xlabel("Runtime")

        fig = plt.gcf()
        fig.set_size_inches(9, 7)

        plt.savefig(f"./plots/sequence_error_progression/aggregate_bound_translation_error_{seq_name}.png", dpi=1300,
                    bbox_inches='tight',
                    facecolor='floralwhite')
        # plt.show()

        plt.clf()

        ##rotation
        metric = []
        time_points = []
        errors = []
        for error, time_point in zip(p3p_median_error_r, p3p_median_time):
            metric.append("p3p_accuracy")
            time_points.append(time_point)
            errors.append(error)

        mean_error_r = np.mean(recons_seq_data['errors_R'])
        median_error_r = np.median(recons_seq_data['errors_R'])
        for error, time_point in zip(recon_interpolated_error_r, interpolation_points):
            metric.append("recon_accuracy")
            time_points.append(time_point)
            errors.append(error)

            metric.append("recon_accuracy_mean")
            time_points.append(time_point)
            errors.append(mean_error_r)

            metric.append("recon_accuracy_median")
            time_points.append(time_point)
            errors.append(median_error_r)

        error_r_data_to_plot = pd.DataFrame.from_dict({"metric":metric, "time_point":time_points, "error":errors})
        sns.lineplot(data=error_r_data_to_plot, x="time_point", y="error", hue="metric", style="metric", legend="brief")

        plt.title(f"Rotation error comparison\nSequence: {seq_name}; P3P iteration limits: {iteration_bounds}")
        plt.grid(axis='y')
        plt.ylabel("Rotation error")
        plt.xlabel("Runtime")

        fig = plt.gcf()
        fig.set_size_inches(9, 7)

        plt.savefig(f"./plots/sequence_error_progression/aggregate_bound_rotation_error_{seq_name}.png", dpi=1300,
                    bbox_inches='tight',
                    facecolor='floralwhite')
        # plt.show()

        plt.clf()

#works for iteration_limit RECON results and iteration-limit p3p results
def compare_aggregate_by_bound_both(present_sequences:list, sample_p3p_data:dict, sample_recon_data:dict, iteration_bounds_p3p:list, iteration_bounds_recon:list):
    available_statistics = ['errors_t', 'errors_R', 'durations']

    for seq_name in present_sequences:
        print(f"\n-----[ Processing sequence {seq_name} ]-----\n")

        #p3p processing
        targets = []

        seq_data = sample_p3p_data[seq_name]

        for statistic in available_statistics:
            statistics_data = np.array(seq_data[statistic])
            targets.append(np.median(statistics_data, axis=0))

        p3p_median_error_t = targets[0]
        p3p_median_error_r = targets[1]
        p3p_median_time = targets[2]

        print(f"Median p3p translation data: {p3p_median_error_t}")
        print(f"Median p3p rotation data: {p3p_median_error_r}")
        print(f"Median p3p time data: {p3p_median_time}")

        #recon processing

        targets = []

        seq_data = sample_recon_data[seq_name]

        for statistic in available_statistics:
            statistics_data = np.array(seq_data[statistic])
            targets.append(np.median(statistics_data, axis=0))

        recon_median_error_t = targets[0]
        recon_median_error_r = targets[1]
        recon_median_time = targets[2]

        print(f"Median recon translation data: {recon_median_error_t}")
        print(f"Median recon rotation data: {recon_median_error_r}")
        print(f"Median recon time data: {recon_median_time}")

        #plotting

        ##translation
        metric = []
        time_points = []
        errors = []
        for error, time_point in zip(p3p_median_error_t, p3p_median_time):
            metric.append("p3p_accuracy")
            time_points.append(time_point)
            errors.append(error)

        for error, time_point in zip(recon_median_error_t, recon_median_time):
            metric.append("recon_accuracy")
            time_points.append(time_point)
            errors.append(error)

        error_t_data_to_plot = pd.DataFrame.from_dict({"metric":metric, "time_point":time_points, "error":errors})
        sns.lineplot(data=error_t_data_to_plot, x="time_point", y="error", hue="metric", style="metric", legend="brief")

        plt.title(f"Translation error comparison; Sequence: {seq_name};\nP3P iteration limits: {iteration_bounds_p3p};\nRECON iteration limits: {iteration_bounds_recon}")
        plt.grid(axis='y')
        plt.ylabel("Translation error")
        plt.xlabel("Runtime (ms)")

        fig = plt.gcf()
        fig.set_size_inches(9, 7)

        plt.savefig(f"./plots/sequence_error_progression/aggregate_bound_both_translation_error_{seq_name}.png", dpi=1300,
                    bbox_inches='tight',
                    facecolor='floralwhite')
        # plt.show()

        plt.clf()

        ##rotation
        metric = []
        time_points = []
        errors = []
        for error, time_point in zip(p3p_median_error_r, p3p_median_time):
            metric.append("p3p_accuracy")
            time_points.append(time_point)
            errors.append(error)

        for error, time_point in zip(recon_median_error_r, recon_median_time):
            metric.append("recon_accuracy")
            time_points.append(time_point)
            errors.append(error)

        error_r_data_to_plot = pd.DataFrame.from_dict({"metric":metric, "time_point":time_points, "error":errors})
        sns.lineplot(data=error_r_data_to_plot, x="time_point", y="error", hue="metric", style="metric", legend="brief")

        plt.title(f"Rotation error comparison; Sequence: {seq_name};\nP3P iteration limits: {iteration_bounds_p3p};\nRECON iteration limits: {iteration_bounds_recon}")
        plt.grid(axis='y')
        plt.ylabel("Rotation error")
        plt.xlabel("Runtime (ms)")

        fig = plt.gcf()
        fig.set_size_inches(9, 7)

        plt.savefig(f"./plots/sequence_error_progression/aggregate_bound_both_rotation_error_{seq_name}.png", dpi=1300,
                    bbox_inches='tight',
                    facecolor='floralwhite')
        # plt.show()

        plt.clf()

#works for zero-shot RECON results and iteration-limit p3p results
def compare_aggregate_by_time(present_sequences:list, sample_p3p_data:dict, recon_data:dict, iteration_bounds:list, n_time_points:int = 7, n_time_points_recon:int = 3, default_rot_error:int=3, default_trans_error:int=3):
    for seq_name in present_sequences:
        print(f"\n-----[ Processing sequence {seq_name} ]-----\n")

        #p3p processing

        seq_data = sample_p3p_data[seq_name]

        p3p_time_points = np.linspace(np.min([np.min(frame_time_data) for frame_time_data in seq_data['durations']]),
                                      np.max([np.max(frame_time_data) for frame_time_data in seq_data['durations']]),
                                      n_time_points+1)
        p3p_time_points = p3p_time_points[1:] #skipping the first one

        print(f"Computed time points for p3p: {p3p_time_points}; P3P iteration limits: {iteration_bounds}")

        translation_error_medians_over_time_frames = []
        rotation_error_medians_over_time_frames = []

        for time_point in p3p_time_points:
            fitting_indices = [] #enumerates the best result indices for all frames

            for frame_time_data in seq_data['durations']: #over lists of durations (for each frame)
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

            #            print(f"Fitting errors t: {indexed_errors_t}")
            #            print(f"Fitting errors r: {indexed_errors_r}")

            translation_error_medians_over_time_frames.append(np.median(indexed_errors_t))
            rotation_error_medians_over_time_frames.append(np.median(indexed_errors_r))

        #        print(f"Final translation errors: {translation_error_medians_over_time_frames}")
        #        print(f"Final rotation errors: {rotation_error_medians_over_time_frames}")

        #recon processing

        recons_seq_data = recon_data[seq_name]
        interpolation_points = np.linspace(np.min(recons_seq_data['durations']), np.max(recons_seq_data['durations']), n_time_points_recon)

        recon_t_interpolator = interp1d(recons_seq_data['durations'], recons_seq_data['errors_t'], kind='linear')
        recon_interpolated_error_t = recon_t_interpolator(interpolation_points) #interpolating by p3p time

        recon_r_interpolator = interp1d(recons_seq_data['durations'], recons_seq_data['errors_R'], kind='linear')
        recon_interpolated_error_r = recon_r_interpolator(interpolation_points) #interpolating by p3p time

        #plotting

        ##translation
        metric = []
        time_points = []
        errors = []
        for error, time_point in zip(translation_error_medians_over_time_frames, p3p_time_points):
            metric.append("p3p_accuracy")
            time_points.append(time_point)
            errors.append(error)

        mean_error_t = np.mean(recons_seq_data['errors_t'])
        median_error_t = np.median(recons_seq_data['errors_t'])
        for error, time_point in zip(recon_interpolated_error_t, interpolation_points):
            metric.append("recon_accuracy")
            time_points.append(time_point)
            errors.append(error)

            metric.append("recon_mean_accuracy")
            time_points.append(time_point)
            errors.append(mean_error_t)

            metric.append("recon_median_accuracy")
            time_points.append(time_point)
            errors.append(median_error_t)

        error_t_data_to_plot = pd.DataFrame.from_dict({"metric":metric, "time_point":time_points, "error":errors})
        sns.lineplot(data=error_t_data_to_plot, x="time_point", y="error", hue="metric", style="metric", legend="brief")

        plt.title(f"Translation error comparison\nSequence: {seq_name}; P3P iteration limits: {iteration_bounds}")
        plt.grid(axis='y')
        plt.ylabel("Translation error")
        plt.xlabel("Runtime")

        fig = plt.gcf()
        fig.set_size_inches(9, 7)

        plt.savefig(f"./plots/sequence_error_progression/aggregate_time_translation_error_{seq_name}.png", dpi=1300,
                    bbox_inches='tight',
                    facecolor='floralwhite')
        # plt.show()

        plt.clf()

        ##rotation
        metric = []
        time_points = []
        errors = []
        for error, time_point in zip(rotation_error_medians_over_time_frames, p3p_time_points):
            metric.append("p3p_accuracy")
            time_points.append(time_point)
            errors.append(error)

        mean_error_r = np.mean(recons_seq_data['errors_R'])
        median_error_r = np.median(recons_seq_data['errors_R'])
        for error, time_point in zip(recon_interpolated_error_r, interpolation_points):
            metric.append("recon_accuracy")
            time_points.append(time_point)
            errors.append(error)

            metric.append("recon_mean_accuracy")
            time_points.append(time_point)
            errors.append(mean_error_r)

            metric.append("recon_median_accuracy")
            time_points.append(time_point)
            errors.append(median_error_r)

        error_r_data_to_plot = pd.DataFrame.from_dict({"metric":metric, "time_point":time_points, "error":errors})
        sns.lineplot(data=error_r_data_to_plot, x="time_point", y="error", hue="metric", style="metric", legend="brief")

        plt.title(f"Rotation error comparison\nSequence: {seq_name}; P3P iteration limits: {iteration_bounds}")
        plt.grid(axis='y')
        plt.ylabel("Rotation error")
        plt.xlabel("Runtime")

        fig = plt.gcf()
        fig.set_size_inches(9, 7)

        plt.savefig(f"./plots/sequence_error_progression/aggregate_time_rotation_error_{seq_name}.png", dpi=1300,
                    bbox_inches='tight',
                    facecolor='floralwhite')
        # plt.show()

        plt.clf()


#works for iteration-limit RECON results and iteration-limit p3p results
def compare_aggregate_by_time_both(present_sequences:list, sample_p3p_data:dict, sample_recon_data:dict, iteration_bounds_p3p:list, iteration_bounds_recon:list, n_time_points:int = 10, default_rot_error:int=3, default_trans_error:int=3):
    for seq_name in present_sequences:
        print(f"\n-----[ Processing sequence {seq_name} ]-----\n")

        #p3p processing

        seq_data = sample_p3p_data[seq_name]

        p3p_time_points = np.linspace(np.min([np.min(frame_time_data) for frame_time_data in seq_data['durations']]),
                                      np.max([np.max(frame_time_data) for frame_time_data in seq_data['durations']]),
                                      n_time_points+1)
        p3p_time_points = p3p_time_points[1:] #skipping the first one

        print(f"Computed time points for p3p: {p3p_time_points}; P3P iteration limits: {iteration_bounds_p3p}")

        translation_error_medians_over_time_frames = []
        rotation_error_medians_over_time_frames = []

        for time_point in p3p_time_points:
            fitting_indices = [] #enumerates the best result indices for all frames

            for frame_time_data in seq_data['durations']: #over lists of durations (for each frame)
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

        #recon processing

        recon_seq_data = sample_recon_data[seq_name]

        recon_time_points = np.linspace(np.min([np.min(frame_time_data) for frame_time_data in recon_seq_data['durations']]),
                                      np.max([np.max(frame_time_data) for frame_time_data in recon_seq_data['durations']]),
                                      n_time_points+1)
        recon_time_points = recon_time_points[1:] #skipping the first one

        print(f"Computed time points for recon: {recon_time_points}; RECON iteration limits: {iteration_bounds_recon}")

        translation_error_medians_over_time_frames_recon = []
        rotation_error_medians_over_time_frames_recon = []

        for time_point in recon_time_points:
            fitting_indices = [] #enumerates the best result indices for all frames

            for frame_time_data in recon_seq_data['durations']: #over lists of durations (for each frame)
                best_time_idx = None
                for idx, result_time_point in enumerate(frame_time_data):
                    if result_time_point <= time_point:
                        best_time_idx = idx

                fitting_indices.append(best_time_idx)

            indexed_errors_t = []
            indexed_errors_r = []

            for idx, errors_t, errors_r in zip(fitting_indices, recon_seq_data['errors_t'], recon_seq_data['errors_R']):
                indexed_errors_t.append(errors_t[idx] if idx is not None else default_trans_error)
                indexed_errors_r.append(errors_r[idx] if idx is not None else default_rot_error)

            #            print(f"Fitting errors t: {indexed_errors_t}")
            #            print(f"Fitting errors r: {indexed_errors_r}")

            translation_error_medians_over_time_frames_recon.append(np.median(indexed_errors_t))
            rotation_error_medians_over_time_frames_recon.append(np.median(indexed_errors_r))

        #        print(f"Final translation errors: {translation_error_medians_over_time_frames}")
        #        print(f"Final rotation errors: {rotation_error_medians_over_time_frames}")

        #plotting

        ##translation
        metric = []
        time_points = []
        errors = []
        for error, time_point in zip(translation_error_medians_over_time_frames, p3p_time_points):
            metric.append("p3p_accuracy")
            time_points.append(time_point)
            errors.append(error)

        for error, time_point in zip(translation_error_medians_over_time_frames_recon, recon_time_points):
            metric.append("recon_accuracy")
            time_points.append(time_point)
            errors.append(error)

        error_t_data_to_plot = pd.DataFrame.from_dict({"metric":metric, "time_point":time_points, "error":errors})
        sns.lineplot(data=error_t_data_to_plot, x="time_point", y="error", hue="metric", style="metric", legend="brief")

        plt.title(f"Translation error comparison; Sequence: {seq_name};\nP3P iteration limits: {iteration_bounds_p3p};\nRECON iteration limits: {iteration_bounds_recon}")
        plt.grid(axis='y')
        plt.ylabel("Translation error")
        plt.xlabel("Runtime (ms)")

        fig = plt.gcf()
        fig.set_size_inches(9, 7)

        plt.savefig(f"./plots/sequence_error_progression/aggregate_time_both_translation_error_{seq_name}.png", dpi=1300,
                    bbox_inches='tight',
                    facecolor='floralwhite')
        # plt.show()

        plt.clf()

        ##rotation
        metric = []
        time_points = []
        errors = []
        for error, time_point in zip(rotation_error_medians_over_time_frames, p3p_time_points):
            metric.append("p3p_accuracy")
            time_points.append(time_point)
            errors.append(error)

        for error, time_point in zip(rotation_error_medians_over_time_frames_recon, recon_time_points):
            metric.append("recon_accuracy")
            time_points.append(time_point)
            errors.append(error)

        error_r_data_to_plot = pd.DataFrame.from_dict({"metric":metric, "time_point":time_points, "error":errors})
        sns.lineplot(data=error_r_data_to_plot, x="time_point", y="error", hue="metric", style="metric", legend="brief")

        plt.title(f"Rotation error comparison; Sequence: {seq_name};\nP3P iteration limits: {iteration_bounds_p3p};\nRECON iteration limits: {iteration_bounds_recon}")
        plt.grid(axis='y')
        plt.ylabel("Rotation error")
        plt.xlabel("Runtime (ms)")

        fig = plt.gcf()
        fig.set_size_inches(9, 7)

        plt.savefig(f"./plots/sequence_error_progression/aggregate_time_both_rotation_error_{seq_name}.png", dpi=1300,
                    bbox_inches='tight',
                    facecolor='floralwhite')
        # plt.show()

        plt.clf()

def main_sample_p3p(dataset_name:str, if_generalize_for_dataset:bool):
    present_sequences = ['seq3', 'seq5', 'seq15'] #TODO: vary this for different scenes

    gt_dirs = Path("./datasets/cambridge/CambridgeLandmarks_Colmap_Retriangulated_1024px")
    model_path = gt_dirs / dataset_name / "empty_all"
    list_file =  gt_dirs / dataset_name / "list_query.txt"

    results_paths = {"p3p":Path(f"./outputs/cambridge/{dataset_name}/results.txt"),"RECON":Path(f"./outputs_recon/cambridge/{dataset_name}/results.txt")}

    iteration_bounds_p3p = [25, 100, 200, 500, 1000, 5000, 7000, 10000]
    num_repetitions_p3p = 30

    sample_p3p_data = evaluate_sampled(model_path,results_paths['p3p'],iteration_bounds_p3p, num_repetitions_p3p, present_sequences, 30,list_file,ext=".txt",sort_errors_by_time=False) #TODO: note sorting here
    recon_data = evaluate_standard(model_path,results_paths['RECON'],present_sequences, list_file,ext=".txt")

    if if_generalize_for_dataset: #if the data has to be aggregated for the entire dataset
        sample_p3p_data['entire_set'] = {}
        sample_p3p_data["entire_set"]['durations'] = [duration_data for seq_name in present_sequences for duration_data in sample_p3p_data[seq_name]['durations']]
        sample_p3p_data["entire_set"]['errors_t'] = [error_data for seq_name in present_sequences for error_data in sample_p3p_data[seq_name]['errors_t']]
        sample_p3p_data["entire_set"]['errors_R'] = [error_data for seq_name in present_sequences for error_data in sample_p3p_data[seq_name]['errors_R']]
        for seq_name in present_sequences:
            sample_p3p_data.pop(seq_name)

        recon_data['entire_set'] = {}
        recon_data["entire_set"]['durations'] = [duration_data for seq_name in present_sequences for duration_data in recon_data[seq_name]['durations']]
        recon_data["entire_set"]['errors_t'] = [error_data for seq_name in present_sequences for error_data in recon_data[seq_name]['errors_t']]
        recon_data["entire_set"]['errors_R'] = [error_data for seq_name in present_sequences for error_data in recon_data[seq_name]['errors_R']]
        for seq_name in present_sequences:
            recon_data.pop(seq_name)

        present_sequences = ['entire_set']

    compare_aggregate_by_bound(present_sequences, sample_p3p_data, recon_data, iteration_bounds_p3p)
    compare_aggregate_by_time(present_sequences, sample_p3p_data, recon_data, iteration_bounds_p3p, 45,7)

def main_sample_both(dataset_name:str, if_generalize_for_dataset:bool):
    present_sequences = ['seq3', 'seq5', 'seq15'] #TODO: vary this for different scenes

    gt_dirs = Path("./datasets/cambridge/CambridgeLandmarks_Colmap_Retriangulated_1024px")
    model_path = gt_dirs / dataset_name / "empty_all"
    list_file =  gt_dirs / dataset_name / "list_query.txt"

    results_paths = {"p3p":Path(f"./outputs/cambridge/{dataset_name}/p3p/results.txt"),"RECON":Path(f"./outputs/cambridge/{dataset_name}/recon/results.txt")}

    #[5, 25, 100, 200, 500, 1000, 5000, 7000, 10000]
    iteration_bounds_p3p = [25, 100, 200, 500, 1000, 5000, 7000, 10000]
    num_repetitions_p3p = 30

    iteration_bounds_recon = [100, 200, 500, 1000, 3000, 5000]
    num_repetitions_recon = 30

    sample_recon_data = evaluate_sampled(model_path,results_paths['RECON'],iteration_bounds_recon, num_repetitions_recon, present_sequences, num_repetitions_recon,list_file,ext=".txt",sort_errors_by_time=False) #TODO: note sorting here
    sample_p3p_data = evaluate_sampled(model_path,results_paths['p3p'],iteration_bounds_p3p, num_repetitions_p3p, present_sequences, num_repetitions_p3p,list_file,ext=".txt",sort_errors_by_time=False) #TODO: note sorting here

    if if_generalize_for_dataset: #if the data has to be aggregated for the entire dataset
        sample_p3p_data['entire_set'] = {}
        sample_p3p_data["entire_set"]['durations'] = [duration_data for seq_name in present_sequences for duration_data in sample_p3p_data[seq_name]['durations']]
        sample_p3p_data["entire_set"]['errors_t'] = [error_data for seq_name in present_sequences for error_data in sample_p3p_data[seq_name]['errors_t']]
        sample_p3p_data["entire_set"]['errors_R'] = [error_data for seq_name in present_sequences for error_data in sample_p3p_data[seq_name]['errors_R']]
        for seq_name in present_sequences:
            sample_p3p_data.pop(seq_name)

        sample_recon_data['entire_set'] = {}
        sample_recon_data["entire_set"]['durations'] = [duration_data for seq_name in present_sequences for duration_data in sample_recon_data[seq_name]['durations']]
        sample_recon_data["entire_set"]['errors_t'] = [error_data for seq_name in present_sequences for error_data in sample_recon_data[seq_name]['errors_t']]
        sample_recon_data["entire_set"]['errors_R'] = [error_data for seq_name in present_sequences for error_data in sample_recon_data[seq_name]['errors_R']]
        for seq_name in present_sequences:
            sample_recon_data.pop(seq_name)

        present_sequences = ['entire_set']

    compare_aggregate_by_bound_both(present_sequences, sample_p3p_data, sample_recon_data, iteration_bounds_p3p, iteration_bounds_recon)
    compare_aggregate_by_time_both(present_sequences, sample_p3p_data, sample_recon_data, iteration_bounds_p3p, iteration_bounds_recon,10)

if __name__ == "__main__":
    # main_sample_p3p(sys.argv[1], True if (len(sys.argv) > 2 and sys.argv[2] == "1") else False)
    main_sample_both(sys.argv[1], True if (len(sys.argv) > 2 and sys.argv[2] == "1") else False)
