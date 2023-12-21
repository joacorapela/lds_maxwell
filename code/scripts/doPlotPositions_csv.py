import sys
import pdb
import numpy as np
import pandas as pd
import argparse
import plotly.graph_objects as go


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_filename", type=str,
                        # default="../../data/mouse_1120297_corrected_DLC_data.h5",
                        default="../../data/mouse_1120297_Stage_2_cleaned_data.csv",
                        help="data filename")
#     parser.add_argument("--body_part", type=str, default="bodycenter",
#                         help="body part to track")
#     parser.add_argument("--first_sample", type=int, default=200000,
#                         help="first sample")
#     parser.add_argument("--number_samples", type=int, default=10000,
#                         help="number of samples to plot")
#     parser.add_argument("--sample_rate", type=float, default=40,
#                         help="sample rate")
    parser.add_argument("--colorscale", type=str, default="Rainbow",
                        help="colorscale name")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filename pattern",
                        default="../../figures/positions_mouse_1120297_Stage_2_cleaned_data.{:s}")

    args = parser.parse_args()

    data_filename = args.data_filename
#     body_part = args.body_part
#     first_sample = args.first_sample
#     number_samples = args.number_samples
#     sample_rate = args.sample_rate
    colorscale = args.colorscale
    fig_filename_pattern = args.fig_filename_pattern

#     df = pd.read_hdf(data_filename)
    df = pd.read_csv(data_filename)
#     scorer=df.columns.get_level_values(0)[0]
#     pos = np.transpose(df[scorer][body_part][["x", "y"]].to_numpy())
    pos = np.transpose(df.iloc[2:, 28:30].to_numpy()) # bodycenter x, y
    pos = pos.astype(np.double)
#     number_samples = pos.shape[1]

#     if number_samples is None:
#         number_samples = pos.shape[1]

#     pos = pos[:, first_sample:(first_sample+number_samples)]
    N = pos.shape[1]
#     dt = 1.0/sample_rate
#     time = np.arange(first_sample*dt, (first_sample+number_samples)*dt, dt)
    time = df.iloc[2:, -1].to_numpy()
    dt = np.diff(time).mean()

    fig = go.Figure()
    trace_mes = go.Scatter(x=pos[0, :], y=pos[1, :],
                           mode="markers",
                           marker={"color": time,
                                   "colorscale": colorscale,
                                   "colorbar": {"title": "Time"},
                                  },
                           customdata=time,
                           hovertemplate="<b>x:</b>%{x:.3f}<br><b>y</b>:%{y:.3f}<br><b>time</b>:%{customdata} sec",
                           name="measured",
                           showlegend=False,
                          )
    fig.add_trace(trace_mes)
    fig.update_layout(xaxis_title="x (pixels)",
                      yaxis_title="y (pixels)",
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')
    fig.show()
    fig.write_image(fig_filename_pattern.format("png"))
    fig.write_html(fig_filename_pattern.format("html"))
    pdb.set_trace()


if __name__ == "__main__":
    main(sys.argv)
