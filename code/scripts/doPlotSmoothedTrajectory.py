import sys
import pickle
import numpy as np
import pandas as pd
import datetime
import argparse
import plotly.graph_objects as go

sys.path.append("../../../../lds_python/src")
import lds.tracking.plotting

def main(argv):
    parser = argparse.ArgumentParser()
    # parser.add_argument("--smoothing_results_number", type=int, default=27148369,
    # parser.add_argument("--smoothing_results_number", type=int, default=72370328,
    # parser.add_argument("--smoothing_results_number", type=int, default=31620178,
    parser.add_argument("--smoothing_results_number", type=int, default=41359254,
                        help="smoothing results numbe")
    parser.add_argument("--first_sample_plot", type=int, default=0,
                        help="start position to plot")
    parser.add_argument("--number_samples_plot", type=int, default=100000,
                        help="number of samples to plot")
    parser.add_argument("--sample_rate", type=float, default=40,
                        help="sample rate")
    parser.add_argument("--min_like", type=float, default=0.2,
                        help=("x and y of samples with likelihood lower than "
                              "min_like is set to nan"))
    parser.add_argument("--variable", type=str, default="pos2D",
                        help="variable to plot: pos2D, pos, vel, acc")
    parser.add_argument("--color_measured", type=str, default="black",
                        help="color for measured markers")
    parser.add_argument("--color_filtered", type=str, default="red",
                        help="color for filtered markers")
    parser.add_argument("--color_smoothed", type=str, default="green",
                        help="color for smoothed markers")
    parser.add_argument("--symbol_x", type=str, default="circle",
                        help="color for x markers")
    parser.add_argument("--symbol_y", type=str, default="circle-open",
                        help="color for y markers")
    parser.add_argument("--smoothing_results_filename_pattern", type=str,
                        default="../../results/{:08d}_smoothing.pickle",
                        help="smoothed data filename pattern")
    parser.add_argument("--fig_filename_pattern", type=str,
                        default="../../figures/{:08d}_smoothing_{:s}_firstSamplePlotted{:d}_numberSamplesPlotted{:d}.{:s}",
                        help="figure filename pattern")

    args = parser.parse_args()

    smoothing_results_number = args.smoothing_results_number
    first_sample_plot = args.first_sample_plot
    number_samples_plot = args.number_samples_plot
    sample_rate = args.sample_rate
    min_like = args.min_like
    variable = args.variable
    color_measured = args.color_measured
    color_filtered = args.color_filtered
    color_smoothed = args.color_smoothed
    symbol_x = args.symbol_x
    symbol_y = args.symbol_y
    smoothing_results_filename_pattern = args.smoothing_results_filename_pattern
    fig_filename_pattern = args.fig_filename_pattern

    smoothing_results_filename = args.smoothing_results_filename_pattern.format(
        smoothing_results_number)
    with open(smoothing_results_filename, "rb") as f:
        smoothing_results = pickle.load(f)
    time = np.transpose(smoothing_results["time"])
    measurements = smoothing_results["measurements"]
    filter_res = smoothing_results["filter_res"]
    smooth_res = smoothing_results["smooth_res"]
    dt = 1.0/sample_rate

    slice_keep = slice(first_sample_plot,
                       first_sample_plot+number_samples_plot, 1)
    time = time[slice_keep]
    measurements = measurements[slice_keep, :]
    filtered_means = filter_res["xnn"][:, 0, slice_keep]
    filtered_stds = np.transpose(np.sqrt(np.diagonal(a=filter_res["Vnn"], axis1=0, axis2=1)))[:, slice_keep]
    smoothed_means = smooth_res["xnN"][:, 0, slice_keep]
    smoothed_stds = np.transpose(np.sqrt(np.diagonal(a=smooth_res["VnN"], axis1=0, axis2=1)))[:, slice_keep]

    y_vel_fd = np.zeros_like(measurements)
    y_vel_fd[:, 1:] = (measurements[:, 1:] - measurements[:, :-1])/dt
    y_vel_fd[:, 0] = y_vel_fd[:, 1]
    y_acc_fd = np.zeros_like(y_vel_fd)
    y_acc_fd[:, 1:] = (y_vel_fd[:, 1:] - y_vel_fd[:, :-1])/dt
    y_acc_fd[:, 0] = y_acc_fd[:, 1]

    if variable == "pos2D":
        fig = lds.tracking.plotting.get_fig_mfs_positions_2D(
            time=time,
            measurements=measurements,
            filtered_means=filtered_means[(0, 3), :],
            smoothed_means=smoothed_means[(0, 3), :],
        )
    elif variable == "pos":
        fig = lds.tracking.plotting.get_fig_mfdfs_kinematics_1D(
            time=time,
            yaxis_title="position (pixels)",
            measurements=measurements,
            filtered_means=filtered_means[(0, 3), :],
            filtered_stds=filtered_stds[(0, 3), :],
            smoothed_means=smoothed_means[(0, 3), :],
            smoothed_stds=smoothed_stds[(0, 3), :],
        )
    elif variable == "vel":
        fig = lds.tracking.plotting.get_fig_mfdfs_kinematics_1D(
            time=time,
            yaxis_title="velocity (pixels/sec)",
            finite_differences=y_vel_fd,
            filtered_means=filtered_means[(1, 4), :],
            filtered_stds=filtered_stds[(1, 4), :],
            smoothed_means=smoothed_means[(1, 4), :],
            smoothed_stds=smoothed_stds[(1, 4), :],
        )
    elif variable == "acc":
        fig = lds.tracking.plotting.get_fig_mfdfs_kinematics_1D(
            time=time,
            yaxis_title="acceleration pixels/sec^2",
            finite_differences=y_acc_fd,
            filtered_means=filtered_means[(2, 5), :],
            filtered_stds=filtered_stds[(2, 5), :],
            smoothed_means=smoothed_means[(2, 5), :],
            smoothed_stds=smoothed_stds[(2, 5), :],
        )
    else:
        raise ValueError("variable={:s} is invalid. It should be: pos2D, pos, vel, acc".format(variable))

    fig.write_image(fig_filename_pattern.format(smoothing_results_number,
                                                variable,
                                                first_sample_plot,
                                                number_samples_plot,
                                                "png"))
    fig.write_html(fig_filename_pattern.format(smoothing_results_number,
                                                variable,
                                                first_sample_plot,
                                                number_samples_plot,
                                               "html"))
    breakpoint()

if __name__ == "__main__":
    main(sys.argv)
