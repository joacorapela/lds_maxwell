import sys
import numpy as np
import pandas as pd
import datetime
import argparse
import plotly.graph_objects as go

sys.path.append("../../../../lds_python/src")
import lds.tracking.plotting

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--first_sample", type=int, default=122000,
                        help="start position to smooth")
    parser.add_argument("--number_samples", type=int, default=200000,
                        help="number of samples to smooth")
    parser.add_argument("--first_sample_plot", type=int, default=0,
                        help="start position to plot")
    parser.add_argument("--number_samples_plot", type=int, default=100000,
                        help="number of samples to plot")
    parser.add_argument("--sample_rate", type=float, default=40,
                        help="sample rate")
    parser.add_argument("--min_like", type=float, default=0.2,
                        help=("x and y of samples with likelihood lower than "
                              "min_like is set to nan"))
    parser.add_argument("--variable", type=str, default="pos",
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
    parser.add_argument("--smoothed_data_filename_pattern", type=str,
                        default="../../results/smoothed_results_firstSample{:d}_numberOfSamples{:d}_minLike{:.02f}.csv",
                        help="smoothed data filename pattern")
    parser.add_argument("--fig_filename_pattern", type=str,
                        default="../../figures/smoothed_results_{:s}_firstSample{:d}_numberOfSamples{:d}_minLike{:.02f}_plotFromSample{:d}To{:d}.{:s}",
                        help="figure filename pattern")

    args = parser.parse_args()

    first_sample = args.first_sample
    number_samples = args.number_samples
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
    smoothed_data_filename_pattern = args.smoothed_data_filename_pattern
    fig_filename_pattern = args.fig_filename_pattern

    smoothed_data_filename = args.smoothed_data_filename_pattern.format(
        first_sample, number_samples, min_like)
    smoothed_data = pd.read_csv(smoothed_data_filename)
    y = np.transpose(smoothed_data[["pos1", "pos2"]].to_numpy())
    dt = 1.0/sample_rate

    slice_keep = slice(first_sample_plot,
                       first_sample_plot+number_samples_plot, 1)
    y = y[:,slice_keep]
    smoothed_data = smoothed_data.iloc[slice_keep,:]

    y_vel_fd = np.zeros_like(y)
    y_vel_fd[:, 1:] = (y[:, 1:] - y[:, :-1])/dt
    y_vel_fd[:, 0] = y_vel_fd[:, 1]
    y_acc_fd = np.zeros_like(y_vel_fd)
    y_acc_fd[:, 1:] = (y_vel_fd[:, 1:] - y_vel_fd[:, :-1])/dt
    y_acc_fd[:, 0] = y_acc_fd[:, 1]

    if variable == "pos2D":
        fig = lds.tracking.plotting.get_fig_mfs_positions_2D(
            time=smoothed_data["time"],
            measured_x=y[0, :],
            measured_y=y[1, :],
            filtered_x=smoothed_data["fpos1"],
            filtered_y=smoothed_data["fpos2"],
            smoothed_x=smoothed_data["spos1"],
            smoothed_y=smoothed_data["spos2"],
        )
    elif variable == "pos":
        fig = lds.tracking.plotting.get_fig_mfdfs_kinematics_1D(
            time=smoothed_data["time"],
            yaxis_title="position (pixels)",
            measured_x=y[0, :],
            measured_y=y[1, :],
            filtered_x=smoothed_data["fpos1"],
            filtered_y=smoothed_data["fpos2"],
            smoothed_x=smoothed_data["spos1"],
            smoothed_y=smoothed_data["spos2"],
        )
    elif variable == "vel":
        fig = lds.tracking.plotting.get_fig_mfdfs_kinematics_1D(
            time=smoothed_data["time"],
            yaxis_title="velocity (pixels/sec)",
            fd_x=y_vel_fd[0, :],
            fd_y=y_vel_fd[1, :],
            filtered_x=smoothed_data["fvel1"],
            filtered_y=smoothed_data["fvel2"],
            smoothed_x=smoothed_data["svel1"],
            smoothed_y=smoothed_data["svel2"],
        )
    elif variable == "acc":
        fig = lds.tracking.plotting.get_fig_mfdfs_kinematics_1D(
            time=smoothed_data["time"],
            yaxis_title="acceleration pixels/sec^2",
            fd_x=y_acc_fd[0, :],
            fd_y=y_acc_fd[1, :],
            filtered_x=smoothed_data["facc1"],
            filtered_y=smoothed_data["facc2"],
            smoothed_x=smoothed_data["sacc1"],
            smoothed_y=smoothed_data["sacc2"],
        )
    else:
        raise ValueError("variable={:s} is invalid. It should be: pos2D, pos, vel, acc".format(variable))

    fig.write_image(fig_filename_pattern.format(variable, first_sample,
                                                number_samples, min_like,
                                                first_sample_plot,
                                                number_samples_plot,
                                                "png"))
    fig.write_html(fig_filename_pattern.format(variable, first_sample,
                                               number_samples, min_like,
                                                first_sample_plot,
                                                number_samples_plot,
                                               "html"))
    breakpoint()

if __name__ == "__main__":
    main(sys.argv)
