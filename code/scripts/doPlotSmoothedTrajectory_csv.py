import sys
import pdb
import numpy as np
import pandas as pd
import datetime
import argparse
import plotly.graph_objects as go

def main(argv):
    parser = argparse.ArgumentParser()
#     parser.add_argument("--first_sample", type=int, default=200000,
#                         help="start position to smooth")
#     parser.add_argument("--number_samples", type=int, default=10000,
#                         help="number of samples to smooth")
#     parser.add_argument("--sample_rate", type=float, default=40,
#                         help="sample rate")
    parser.add_argument("--variable", type=str, default="pos",
                        help="variable to plot: pos, vel, acc")
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
    parser.add_argument("--smoothed_data_filename", type=str,
#                         default="../../results/smoothed_results_firstSample{:d}_numberOfSamples{:d}.csv",
                        default="../../results/mouse_1120297_Stage_2_cleaned_smoothed.csv",
                        help="smoothed data filename pattern")
    parser.add_argument("--fig_filename_pattern", type=str,
                        # default="../../figures/smoothed_results_{:s}_firstSample{:d}_numberOfSamples{:d}.{:s}",
                        default="../../figures/mouse_1120297_Stage_2_cleaned_smoothed_{:s}.{:s}",
                        help="figure filename pattern")

    args = parser.parse_args()

#     first_sample = args.first_sample
#     number_samples = args.number_samples
#     sample_rate = args.sample_rate
    variable = args.variable
    color_measured = args.color_measured
    color_filtered = args.color_filtered
    color_smoothed = args.color_smoothed
    symbol_x = args.symbol_x
    symbol_y = args.symbol_y
    smoothed_data_filename = args.smoothed_data_filename
    fig_filename_pattern = args.fig_filename_pattern

#     smoothed_data_filename = args.smoothed_data_filename_pattern.format(
#         first_sample, number_samples)
    smoothed_data = pd.read_csv(smoothed_data_filename)
    y = np.transpose(smoothed_data[["pos1", "pos2"]].to_numpy())
    dt = np.diff(smoothed_data["time"]).mean()

    y_vel_fd = np.zeros_like(y)
    y_vel_fd[:, 1:] = (y[:, 1:] - y[:, :-1])/dt
    y_vel_fd[:, 0] = y_vel_fd[:, 1]
    y_acc_fd = np.zeros_like(y_vel_fd)
    y_acc_fd[:, 1:] = (y_vel_fd[:, 1:] - y_vel_fd[:, :-1])/dt
    y_acc_fd[:, 0] = y_acc_fd[:, 1]

    fig = go.Figure()
    if variable == "pos":
        trace_fd = go.Scatter(x=y[0, :], y=y[1, :],
                              mode="markers",
                              marker={"color": color_measured},
                              customdata=smoothed_data["time"],
                              hovertemplate="<b>x:</b>%{x:.3f}<br><b>y</b>:%{y:.3f}<br><b>time</b>:%{customdata} sec",
                              name="measured",
                              showlegend=True,
                              )
        trace_filtered = go.Scatter(x=smoothed_data["fpos1"],
                                    y=smoothed_data["fpos2"],
                                    mode="markers",
                                    marker={"color": color_filtered},
                                    customdata=smoothed_data["time"],
                                    hovertemplate="<b>x:</b>%{x:.3f}<br><b>y</b>:%{y:.3f}<br><b>time</b>:%{customdata} sec",
                                    name="filtered",
                                    showlegend=True,
                                    )
        trace_smoothed = go.Scatter(x=smoothed_data["spos1"],
                                    y=smoothed_data["spos2"],
                                    mode="markers",
                                    marker={"color": color_smoothed},
                                    customdata=smoothed_data["time"],
                                    hovertemplate="<b>x:</b>%{x:.3f}<br><b>y</b>:%{y:.3f}<br><b>time</b>:%{customdata} sec",
                                    name="smoothed",
                                    showlegend=True,
                                    )
        fig.add_trace(trace_fd)
        fig.add_trace(trace_filtered)
        fig.add_trace(trace_smoothed)
        fig.update_layout(xaxis_title="x (pixels)", yaxis_title="y (pixels)",
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
    elif variable == "vel":
        trace_fd_x = go.Scatter(x=smoothed_data["time"],
                                 y=y_vel_fd[0, :],
                                 mode="lines+markers",
                                 marker={"color": color_measured},
                                 name="finite diff x",
                                 showlegend=True,
                                 )
        trace_fd_y = go.Scatter(x=smoothed_data["time"],
                                 y=y_vel_fd[1, :],
                                 mode="lines+markers",
                                 marker={"color": color_measured},
                                 name="finite diff y",
                                 showlegend=True,
                                 )
        trace_filtered_x = go.Scatter(x=smoothed_data["time"],
                                      y=smoothed_data["fvel1"],
                                      mode="lines+markers",
                                      marker={"color": color_filtered,
                                              "symbol": symbol_x},
                                      name="filtered x",
                                      showlegend=True,
                                      )
        trace_filtered_y = go.Scatter(x=smoothed_data["time"],
                                      y=smoothed_data["fvel2"],
                                      mode="lines+markers",
                                      marker={"color": color_filtered,
                                              "symbol": symbol_y},
                                      name="filtered y",
                                      showlegend=True,
                                      )
        trace_smoothed_x = go.Scatter(x=smoothed_data["time"],
                                      y=smoothed_data["svel1"],
                                      mode="lines+markers",
                                      marker={"color": color_smoothed,
                                              "symbol": symbol_x},
                                      name="smoothed x",
                                      showlegend=True,
                                      )
        trace_smoothed_y = go.Scatter(x=smoothed_data["time"],
                                      y=smoothed_data["svel2"],
                                      mode="lines+markers",
                                      marker={"color": color_smoothed,
                                              "symbol": symbol_y},
                                      name="smoothed y",
                                      showlegend=True,
                                      )
        fig.add_trace(trace_fd_x)
        fig.add_trace(trace_fd_y)
        fig.add_trace(trace_filtered_x)
        fig.add_trace(trace_filtered_y)
        fig.add_trace(trace_smoothed_x)
        fig.add_trace(trace_smoothed_y)
        fig.update_layout(xaxis_title="time", yaxis_title="velocity",
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
    elif variable == "acc":
        trace_fd_x = go.Scatter(x=smoothed_data["time"],
                                 y=y_acc_fd[0, :],
                                 mode="lines+markers",
                                 marker={"color": color_measured},
                                 name="finite diff x",
                                 showlegend=True,
                                 )
        trace_fd_y = go.Scatter(x=smoothed_data["time"],
                                 y=y_acc_fd[1, :],
                                 mode="lines+markers",
                                 marker={"color": color_measured},
                                 name="finite diff y",
                                 showlegend=True,
                                 )
        trace_filtered_x = go.Scatter(x=smoothed_data["time"],
                                      y=smoothed_data["facc1"],
                                      mode="lines+markers",
                                      marker={"color": color_filtered,
                                              "symbol": symbol_x},
                                      name="filtered x",
                                      showlegend=True,
                                      )
        trace_filtered_y = go.Scatter(x=smoothed_data["time"],
                                      y=smoothed_data["facc2"],
                                      mode="lines+markers",
                                      marker={"color": color_filtered,
                                              "symbol": symbol_y},
                                      name="filtered y",
                                      showlegend=True,
                                      )
        trace_smoothed_x = go.Scatter(x=smoothed_data["time"],
                                      y=smoothed_data["sacc1"],
                                      mode="lines+markers",
                                      marker={"color": color_smoothed,
                                              "symbol": symbol_x},
                                      name="smoothed x",
                                      showlegend=True,
                                      )
        trace_smoothed_y = go.Scatter(x=smoothed_data["time"],
                                      y=smoothed_data["sacc2"],
                                      mode="lines+markers",
                                      marker={"color": color_smoothed,
                                              "symbol": symbol_y},
                                      name="smoothed y",
                                      showlegend=True,
                                      )
        fig.add_trace(trace_fd_x)
        fig.add_trace(trace_fd_y)
        fig.add_trace(trace_filtered_x)
        fig.add_trace(trace_filtered_y)
        fig.add_trace(trace_smoothed_x)
        fig.add_trace(trace_smoothed_y)
        fig.update_layout(xaxis_title="time", yaxis_title="acceleration",
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
    else:
        raise ValueError("variable={:s} is invalid. It should be: pos, vel, acc".format(variable))

    fig.write_image(fig_filename_pattern.format(variable, "png"))
    fig.write_html(fig_filename_pattern.format(variable, "html"))
    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)
